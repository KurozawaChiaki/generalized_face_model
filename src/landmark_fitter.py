import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional
from pathlib import Path

from libs.sculptor.model.sculptor import SCULPTOR_layer
from flame_pytorch import FLAME

import mvlm


class LandmarkFitter:
    def __init__(
        self,
        target_path: str,
        sculptor_paradict_path: str,
        general_config: dict,
        sculptor_scale: float = 1000,
        fitting_method: str = "mesh",
        output_dir: str = "fitted_results",
        base_iterations: int = 1000,
        lr: float = 0.001,
    ):
        # File paths
        self.target_path = target_path
        self.output_dir = Path(output_dir)
        self.sculptor_path = sculptor_paradict_path

        # Fitting iteration parameters
        self.base_iterations = base_iterations
        self.lr = lr

        # Fitting method
        self.fitting_method = fitting_method

        # Sculptor scale
        self.sculptor_scale = sculptor_scale

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # General config
        self.general_config = general_config

        # Global translation
        self.centroid = None

    def rotation6d_to_matrix(
        self,
        rotation6d_tensor: torch.Tensor = None,
        rotation6d_array: np.ndarray = None,
    ) -> torch.Tensor:
        """
        Convert 6D rotation representation to 3x3 rotation matrix.
        """
        if rotation6d_tensor is not None:
            rotation6d = rotation6d_tensor
        elif rotation6d_array is not None:
            rotation6d = torch.tensor(
                rotation6d_array, dtype=torch.float32, device=self.device
            )
        else:
            raise ValueError(
                "Either rotation6d_tensor or rotation6d_array must be provided"
            )

        x_raw = rotation6d[0:3]
        y_raw = rotation6d[3:6]

        x = F.normalize(x_raw, dim=-1, eps=1e-8)
        y = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
        y = F.normalize(y, dim=-1, eps=1e-8)
        z = torch.cross(x, y, dim=-1)

        matrix = torch.stack([x, y, z], dim=-1)

        return matrix

    def compute_initial_translation_and_rotation(
        self, sculptor_model, target_landmarks, initial_landmarks
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the initial translation and rotation of the SCULPTOR model.
        """
        landmark_tensor: torch.Tensor = torch.tensor(
            initial_landmarks,
            dtype=sculptor_model.dtype,
            device=self.device,
        )

        target_landmarks_tensor = torch.tensor(
            target_landmarks, dtype=sculptor_model.dtype, device=self.device
        )

        translation = torch.zeros((3, 1), requires_grad=True, device=self.device)
        rotation = torch.tensor(
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0], requires_grad=True, device=self.device
        )

        optimizer = torch.optim.LBFGS(
            [translation, rotation], lr=1.0, max_iter=100, line_search_fn="strong_wolfe"
        )

        pbar = tqdm(range(self.base_iterations // 10))

        def closure():
            optimizer.zero_grad()

            rotation_tensor = self.rotation6d_to_matrix(rotation6d_tensor=rotation)
            fitted_landmarks = torch.matmul(
                rotation_tensor, landmark_tensor.transpose(0, 1)
            )  # (N, 3) * (3, 3) + (3,) = (N, 3)
            fitted_landmarks = fitted_landmarks + translation
            fitted_landmarks = fitted_landmarks.transpose(0, 1)

            loss = torch.sum((target_landmarks_tensor - fitted_landmarks) ** 2)
            loss.backward()

            pbar.set_description(f"Loss: {loss.item():.6f}")

            return loss

        for iter_num in pbar:
            optimizer.step(closure)

        with torch.no_grad():
            translation_result = translation.detach().cpu().numpy()
            rotation_result = rotation.detach().cpu().numpy()

        return translation_result, rotation_result

    def fit_shape_and_expression(
        self,
        sculptor_model: SCULPTOR_layer,
        beta_s,
        pose_s,
        jaw_s,
        translation,
        rotation,
        sculptor_landmarks,
        target_landmarks,
        sparse_matrix,
    ):
        target_landmarks_tensor = torch.tensor(
            target_landmarks, dtype=sculptor_model.dtype, device=self.device
        )

        translation_tensor = torch.tensor(
            translation, dtype=sculptor_model.dtype, device=self.device
        )
        sparse_matrix_tensor = torch.tensor(
            sparse_matrix, dtype=sculptor_model.dtype, device=self.device
        )
        rotation_tensor = self.rotation6d_to_matrix(rotation6d_array=rotation)

        beta = beta_s.clone().detach().requires_grad_(True)
        pose = pose_s.clone().detach().requires_grad_(True)
        jaw = jaw_s.clone().detach().requires_grad_(True)

        optimizer = torch.optim.LBFGS(
            [beta, pose, jaw], lr=1.0, max_iter=100, line_search_fn="strong_wolfe"
        )

        pbar = tqdm(range(self.base_iterations // 10))

        def closure():
            optimizer.zero_grad()

            vertices = sculptor_model(beta, pose, jaw)
            _, vertices = sculptor_model.split_mesh(vertices)
            vertices = vertices.squeeze(0)  # (N, 3)

            # Initial Processing
            initial_rotation_matrix = self.rotation6d_to_matrix(
                rotation6d_array=np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
            )
            vertices = torch.matmul(vertices, initial_rotation_matrix)
            vertices = vertices - torch.tensor(
                self.centroid, dtype=sculptor_model.dtype, device=self.device
            )
            fitted_landmarks = torch.matmul(
                torch.tensor(
                    sparse_matrix, dtype=sculptor_model.dtype, device=self.device
                ),
                vertices,
            )  # (M, N) * (N, 3) = (M, 3)

            # Fitted translation and rotation
            fitted_landmarks = torch.matmul(
                rotation_tensor, fitted_landmarks.transpose(0, 1)
            )
            fitted_landmarks = fitted_landmarks + translation_tensor
            fitted_landmarks = fitted_landmarks.transpose(0, 1)

            loss = torch.sum((target_landmarks_tensor - fitted_landmarks) ** 2)
            loss.backward()

            pbar.set_description(f"Loss: {loss.item():.6f}")

            return loss

        for iter_num in pbar:
            optimizer.step(closure)

        with torch.no_grad():
            beta_result = beta.detach().cpu().numpy()
            pose_result = pose.detach().cpu().numpy()
            jaw_result = jaw.detach().cpu().numpy()
        return beta_result, pose_result, jaw_result

    def get_translated_mesh(
        self, sculptor_model, beta_s, pose_s, jaw_s, translation, rotation
    ):
        mesh, _ = self.sculptor_to_trimesh(beta_s, pose_s, jaw_s, sculptor_model)
        vertices = np.matmul(
            self.rotation6d_to_matrix(rotation6d_array=rotation).cpu().numpy(),
            mesh.vertices.T,
        )
        vertices = vertices + translation
        mesh = trimesh.Trimesh(vertices=vertices.T, faces=mesh.faces)
        return mesh

    def sculptor_to_trimesh(self, beta_s, pose_s, jaw_s, sculptor_model):
        initial_rotation_6d = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        initial_rotation_matrix = self.rotation6d_to_matrix(
            rotation6d_array=initial_rotation_6d
        )

        vertices = sculptor_model(beta_s, pose_s, jaw_s)
        vertices = torch.matmul(vertices, initial_rotation_matrix)

        skull_vertices, face_vertices = sculptor_model.split_mesh(vertices)

        skull_vertices = skull_vertices.squeeze(0).detach().cpu().numpy()
        face_vertices = face_vertices.squeeze(0).detach().cpu().numpy()
        vertices = vertices.squeeze(0).detach().cpu().numpy()

        # Get facial mesh
        face_mesh = sculptor_model.facialmesh_face
        # Default number of vertices is 4 for every face, transform into 3
        temp_vertices_index = face_mesh[:, 0]
        temp_vertices_index = temp_vertices_index.reshape(face_mesh.shape[0], 1)
        additional_meshes = np.concatenate(
            [temp_vertices_index, face_mesh[:, 2:]], axis=1
        )
        face_mesh = np.concatenate([face_mesh[:, 0:3], additional_meshes], axis=0)

        face_trimesh = trimesh.Trimesh(vertices=face_vertices, faces=face_mesh)

        if self.centroid is None:
            self.centroid = face_trimesh.centroid
        face_vertices = face_vertices - self.centroid
        skull_vertices = skull_vertices - self.centroid

        face_trimesh = trimesh.Trimesh(
            vertices=face_vertices, faces=face_mesh, process=False, maintain_order=True
        )
        skull_trimesh = trimesh.Trimesh(
            vertices=skull_vertices,
            faces=sculptor_model.skullmesh_face,
            process=False,
        )

        return face_trimesh, skull_trimesh

    def fit_to_mesh(self):
        # Load target mesh and save as a temporary .obj file for mvlm
        temp_path = Path("./temp")
        target_mesh = trimesh.load(self.target_path)
        target_vertices = target_mesh.vertices
        target_faces = target_mesh.faces

        # Apply scale to target vertices
        scale_matrix = np.eye(3) * self.sculptor_scale
        target_vertices = np.matmul(scale_matrix, target_vertices.T).T
        target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)

        target_centroid = target_mesh.centroid
        target_vertices = target_vertices - target_centroid
        target_mesh = trimesh.Trimesh(vertices=target_vertices, faces=target_faces)

        # Save target mesh as a temporary .obj file
        os.makedirs(temp_path, exist_ok=True)
        target_mesh.export(temp_path / "target_mesh.obj")

        # Get target landmarks
        print("Getting target landmarks...")
        mvlm_pipeline = mvlm.pipeline.create_pipeline("face_alignment")
        target_landmarks = mvlm_pipeline.predict_one_file(temp_path / "target_mesh.obj")
        os.remove(temp_path / "target_mesh.obj")

        # Load SCULPTOR model
        sculptor_model = SCULPTOR_layer(self.sculptor_path).to(self.device)

        # Prepare sculptor parameters
        num_joints = self.general_config["sculptor_config"]["num_joints"]
        n_shape = self.general_config["sculptor_config"]["n_shape"]
        beta_s = torch.tensor(
            np.zeros((1, n_shape)) + 1e-5, dtype=torch.float32, device=self.device
        )
        pose_s = torch.tensor(
            np.tile(
                np.zeros(3 * (num_joints + 1)).reshape(1, 3 * (num_joints + 1)),
                (1, 1),
            ),
            dtype=torch.float32,
            device=self.device,
        )
        jaw_s = torch.tensor(
            np.zeros((1, 3, 1)), dtype=torch.float32, device=self.device
        )

        initial_face_trimesh, _ = self.sculptor_to_trimesh(
            beta_s, pose_s, jaw_s, sculptor_model
        )
        initial_face_trimesh.export(temp_path / "initial_face_trimesh.obj")

        print("Getting initial landmarks...")
        initial_landmarks = mvlm_pipeline.predict_one_file(
            temp_path / "initial_face_trimesh.obj"
        )
        os.remove(temp_path / "initial_face_trimesh.obj")

        sculptor_landmarks = []
        closest_points, distances, tri_indices = trimesh.proximity.closest_point(
            initial_face_trimesh, initial_landmarks
        )

        for closest_point, tri_idx in zip(closest_points, tri_indices):
            landmark_info = {
                "vertices_idx": initial_face_trimesh.faces[tri_idx],
                "weights": [],
            }
            triangle = np.expand_dims(initial_face_trimesh.triangles[tri_idx], axis=0)
            closest_point = np.expand_dims(closest_point, axis=0)

            weights = trimesh.triangles.points_to_barycentric(triangle, closest_point)
            weights = weights.squeeze(0)
            landmark_info["weights"] = weights
            sculptor_landmarks.append(landmark_info)

        sparse_matrix = np.zeros(
            (len(target_landmarks), len(initial_face_trimesh.vertices))  # (3M, N)
        )  # [i, j] = weight of j-th sculptor vertex for i-th target landmark
        for i, landmark_info in enumerate(sculptor_landmarks):
            sparse_matrix[i, landmark_info["vertices_idx"]] = landmark_info["weights"]

        translation, rotation = self.compute_initial_translation_and_rotation(
            sculptor_model, target_landmarks, initial_landmarks
        )

        print(f"Translation: {translation}")
        print(f"Rotation: {rotation}")

        translated_mesh = self.get_translated_mesh(
            sculptor_model, beta_s, pose_s, jaw_s, translation, rotation
        )
        # translated_mesh.export(temp_path / "translated_mesh.obj")

        beta_result, pose_result, jaw_result = self.fit_shape_and_expression(
            sculptor_model,
            beta_s,
            pose_s,
            jaw_s,
            translation,
            rotation,
            sculptor_landmarks,
            target_landmarks,
            sparse_matrix,
        )
        beta_result = torch.tensor(
            beta_result, dtype=sculptor_model.dtype, device=self.device
        )
        pose_result = torch.tensor(
            pose_result, dtype=sculptor_model.dtype, device=self.device
        )
        jaw_result = torch.tensor(
            jaw_result, dtype=sculptor_model.dtype, device=self.device
        )

        os.removedirs(temp_path)
        result_mesh = self.get_translated_mesh(
            sculptor_model,
            beta_result,
            pose_result,
            jaw_result,
            translation,
            rotation,
        )
        result_mesh.export(self.output_dir / "result_mesh.obj")

    def fit(self):
        if self.fitting_method == "mesh":
            self.fit_to_mesh()
        elif self.fitting_method == "parameter":
            pass
        else:
            raise ValueError(f"Invalid fitting method: {self.fitting_method}")
