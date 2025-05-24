import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional

from dependencies.sculptor.model.sculptor import SCULPTOR_layer

class SculptorFitter():
    def __init__(self, 
                 target_mesh_path: str,
                 sculptor_paradict_path: str,
                 output_dir: str="fitted_results",
                 num_iterations: int=200,
                 lr: float=0.01,
                 shape_reg_weight: float=0.001,
                 pose_reg_weight: float=0.001,
                 n_shape: int=60):
        print(f"Target mesh: {target_mesh_path}")
        print(f"SCULPTOR params: {sculptor_paradict_path}")
        print(f"Output directory: {output_dir}")
        print(f"Iterations: {num_iterations}, LR: {lr}")
    
        os.makedirs(output_dir, exist_ok=True)

        # Load target mesh
        try:
            self.target_vertices = self.load_target_mesh_vertices(target_mesh_path) # Shape: (N_target_verts, 3)
        except Exception as e:
            print(f"Error loading target mesh '{target_mesh_path}': {e}")
            return
        print(f"Target vertices: {self.target_vertices.shape}")

        # Initialize SCULPTOR model
        try:
            self.sculptor_model = SCULPTOR_layer(sculptor_paradict_path)
        except Exception as e:
            print(f"Error initializing SCULPTOR_layer (possibly related to '{sculptor_paradict_path}'): {e}")
            return

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.sculptor_model.to(self.device)
        self.target_vertices = self.target_vertices.to(self.device, dtype=self.sculptor_model.dtype)

        # Initialize parameters to optimize
        self.n_shape = n_shape
        self.num_joints = 1

        self.shape_reg_weight = shape_reg_weight
        self.pose_reg_weight = pose_reg_weight
        self.lr = lr
        self.num_iterations = num_iterations

    
    def load_target_mesh_vertices(self, mesh_path: str) -> torch.Tensor:
        '''
        Loads mesh vertices using trimesh.
        '''
        mesh = trimesh.load_mesh(mesh_path, process=False)
        vertices = mesh.vertices.astype(np.float32)
        
        return torch.from_numpy(vertices)
        

    def compute_approximate_scale(self) -> np.ndarray:
        '''
        Compute the approximate scale of the target mesh.
        '''
        template_vertices = self.sculptor_model.template_face
        template_vertices = torch.tensor(template_vertices.unsqueeze(0), dtype=self.sculptor_model.dtype, device=self.device)
        
        scale = torch.tensor(1.0, dtype=self.sculptor_model.dtype, requires_grad=True, device=self.device)
        eye = torch.tensor(np.eye(3), dtype=self.sculptor_model.dtype, requires_grad=False, device=self.device)

        optimizer = torch.optim.AdamW([scale], lr=self.lr)

        pbar = tqdm(range(self.num_iterations))
        for iter_num in pbar:
            optimizer.zero_grad()

            fitted_mesh_vertices = self.target_vertices * scale * eye

            loss = self.custom_chamfer_distance(fitted_mesh_vertices, self.target_vertices)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.6f}")

        with torch.no_grad():
            scale_result = scale.detach().cpu().numpy()

        return scale_result
    

    def rotation6d_to_matrix(self, rotation6d: torch.Tensor) -> torch.Tensor:
        '''
        Convert 6D rotation representation to 3x3 rotation matrix.
        '''
        x_raw = rotation6d[0:3]
        y_raw = rotation6d[3:6]

        x = F.normalize(x_raw, dim=-1, eps=1e-8)
        y = y_raw - (x * y_raw).sum(dim=-1, keepdim=True) * x
        y = F.normalize(y, dim=-1, eps=1e-8)
        z = torch.cross(x, y, dim=-1)

        matrix = torch.stack([x, y, z], dim=-1)

        return matrix
        

    def compute_initial_translation_and_rotation(self) -> tuple[np.ndarray, np.ndarray]:
        '''
        Compute the initial translation and rotation of the SCULPTOR model.
        '''
        template_vertices = self.sculptor_model.template_face
        template_vertices = torch.tensor(template_vertices.unsqueeze(0), dtype=self.sculptor_model.dtype, device=self.device)

        translation = torch.zeros((3), dtype=self.sculptor_model.dtype, requires_grad=True, device=self.device)
        rotation = torch.zeros((6), dtype=self.sculptor_model.dtype, requires_grad=True, device=self.device)

        optimizer = torch.optim.AdamW([translation, rotation], lr=self.lr)

        pbar = tqdm(range(self.num_iterations))
        for iter_num in pbar:
            optimizer.zero_grad()

            rotation_matrix = self.rotation6d_to_matrix(rotation)
            # (N, 3) * (3, 3) + (3,) = (N, 3)
            fitted_mesh_vertices = self.target_vertices * rotation_matrix + translation

            loss = self.custom_chamfer_distance(fitted_mesh_vertices, self.target_vertices)

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.6f}")
        
        with torch.no_grad():
            translation_result = translation.detach().cpu().numpy()
            rotation_result = rotation.detach().cpu().numpy()

        return translation_result, rotation_result

    
    def custom_chamfer_distance(self, 
                                p1: torch.Tensor, 
                                p2: torch.Tensor, 
                                p1_lengths: Optional[torch.Tensor]=None, 
                                p2_lengths: Optional[torch.Tensor]=None, 
                                device: str="cuda") -> torch.Tensor:
        """
        Custom Chamfer distance implementation.
        
        p1: (B, N, 3) tensor of N points
        p2: (B, M, 3) tensor of M points
        p1_lengths: (B,) tensor of actual number of points in p1, if padded
        p2_lengths: (B,) tensor of actual number of points in p2, if padded
        
        Returns: scalar
        """
        p1 = p1.to(device)
        p2 = p2.to(device)

        # diff[b, i, j, :] = p1[b, i, :] - p2[b, j, :]
        diff = p1.unsqueeze(2) - p2.unsqueeze(1)  # (B, N, M, 3)
        dist_sq = torch.sum(torch.pow(diff, 2), dim=3)  # (B, N, M)
    
        # Mask invalid distances if lengths are provided (for padded batches)
        if p1_lengths is not None:
            mask_p1 = torch.arange(p1.shape[1], device=p1.device)[None, :] >= p1_lengths[:, None] # (B, N)
            dist_sq.masked_fill_(mask_p1[:, :, None].expand_as(dist_sq), float('inf'))
        if p2_lengths is not None:
            mask_p2 = torch.arange(p2.shape[1], device=p2.device)[None, :] >= p2_lengths[:, None] # (B, M)
            dist_sq.masked_fill_(mask_p2[:, None, :].expand_as(dist_sq), float('inf'))

        dist_p1_p2, _ = torch.min(dist_sq, dim=2)  # (B, N), min dist from each point in p1 to p2
        dist_p2_p1, _ = torch.min(dist_sq, dim=1)  # (B, M), min dist from each point in p2 to p1

        # Mask invalid points before mean
        if p1_lengths is not None:
            dist_p1_p2.masked_fill_(mask_p1, 0.0) # Fill with 0, sum then divide by actual length
            loss_p1_p2 = torch.sum(dist_p1_p2, dim=1) / p1_lengths.float()
        else:
            loss_p1_p2 = torch.mean(dist_p1_p2, dim=1)

        if p2_lengths is not None:
            dist_p2_p1.masked_fill_(mask_p2, 0.0)
            loss_p2_p1 = torch.sum(dist_p2_p1, dim=1) / p2_lengths.float()
        else:
            loss_p2_p1 = torch.mean(dist_p2_p1, dim=1)
        
        chamfer_loss = torch.mean(loss_p1_p2 + loss_p2_p1)
    
        # Handle cases where N or M might be zero (though unlikely for meshes)
        if p1.shape[1] == 0 and p2.shape[1] == 0:
            return torch.tensor(0.0, device=p1.device, dtype=p1.dtype)
        if p1.shape[1] == 0: # only p2 has points
            return torch.mean(loss_p2_p1) if p2_lengths is None else torch.mean(loss_p2_p1[p2_lengths > 0])
        if p2.shape[1] == 0: # only p1 has points
            return torch.mean(loss_p1_p2) if p1_lengths is None else torch.mean(loss_p1_p2[p1_lengths > 0])
        
        return chamfer_loss
    

    def fit(self) -> tuple[dict[str, np.ndarray], torch.Tensor]:
        # TODO: determine the order of fitting
        initial_translation, initial_rotation = self.compute_initial_translation_and_rotation()
        scale: np.ndarray = self.compute_approximate_scale()
        print(f"Scale: {scale}")
        print(f"Initial translation: {initial_translation}")
        print(f"Initial rotation: {initial_rotation}")

        self.target_vertices = self.target_vertices * scale * self.rotation6d_to_matrix(initial_rotation) + initial_translation

        beta_s = torch.tensor(np.zeros((1, self.n_shape)) + 1e-5, dtype=self.sculptor_model.dtype, requires_grad=True)
        pose_theta = torch.tensor(np.tile(np.zeros(3 * (self.num_joints + 1)).reshape(1, 3 * (self.num_joints + 1)),(1, 1)), dtype=self.sculptor_model.dtype, requires_grad=True)
        jaw_offset = torch.tensor(np.zeros((1, 3, 1)), dtype=self.sculptor_model.dtype, requires_grad=True)

        # Optimizer
        optimizer = torch.optim.AdamW([beta_s, pose_theta, jaw_offset], lr=self.lr)

        print("Starting optimization...")
        # Optimization loop
        pbar: tqdm = tqdm(range(self.num_iterations))
        for iter_num in pbar:
            optimizer.zero_grad()

            fitted_mesh_vertices = self.sculptor_model(beta_s, pose_theta, jaw_offset)

            loss_chamfer: torch.Tensor = self.custom_chamfer_distance(fitted_mesh_vertices, self.target_vertices)
        
            loss_reg_shape: torch.Tensor = torch.mean(torch.square(beta_s))
            loss_reg_pose: torch.Tensor = torch.mean(torch.square(pose_theta)) # Regularize pose parameters
            
            loss = loss_chamfer + self.shape_reg_weight * loss_reg_shape + self.pose_reg_weight * loss_reg_pose

            loss.backward()
            optimizer.step()

            pbar.set_description(f"Loss: {loss.item():.6f}")

        with torch.no_grad():
            optimized_params_dict: dict[str, np.ndarray] = {
                'beta_s': beta_s.detach().cpu().numpy(),
                'pose_theta': pose_theta.detach().cpu().numpy(),
                'jaw_offset': jaw_offset.detach().cpu().numpy()
            }
            final_fitted_mesh_verts: torch.Tensor = fitted_mesh_vertices.detach().cpu()

        return optimized_params_dict, final_fitted_mesh_verts


if __name__ == "__main__":
    # TODO: test codes
    pass