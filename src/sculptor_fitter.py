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

            fitted_mesh_vertices = template_vertices * scale * eye

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
            fitted_mesh_vertices = template_vertices * rotation_matrix + translation

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
        dist_sq = torch.sum(diff**2, dim=3)  # (B, N, M)
    
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
    

    def fit(self):
        initial_translation, initial_rotation = self.compute_initial_translation_and_rotation()
        scale = self.compute_approximate_scale()

'''
# --- Main Fitting Function ---
def fit_sculptor_to_target(target_mesh_path,
                           paradict_path,
                           output_dir="fitted_results",
                           num_iterations=200,
                           lr=0.01,
                           shape_reg_weight=0.001,
                           pose_reg_weight=0.001):
    """
    Fits the SCULPTOR model to a target mesh.
    """
    print(f"Target mesh: {target_mesh_path}")
    print(f"SCULPTOR params: {paradict_path}")
    print(f"Output directory: {output_dir}")
    print(f"Iterations: {num_iterations}, LR: {lr}")
    
    os.makedirs(output_dir, exist_ok=True)

    # Load target mesh
    try:
        target_vertices = load_target_mesh_vertices(target_mesh_path) # Shape: (N_target_verts, 3)
    except Exception as e:
        print(f"Error loading target mesh '{target_mesh_path}': {e}")
        return None, None
    print(f"Target vertices: {target_vertices.shape}")
        
    target_vertices = target_vertices.unsqueeze(0) # Add batch dimension (B=1)

    # Initialize SCULPTOR model
    try:
        sculptor_model = SCULPTOR_layer(paradict_path)
    except Exception as e:
        print(f"Error initializing SCULPTOR_layer (possibly related to '{paradict_path}'): {e}")
        return None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    sculptor_model.to(device)
    target_vertices = target_vertices.to(device, dtype=sculptor_model.dtype)

    # Initialize parameters to optimize
    num_shape_params = sculptor_model.shape_dirs.shape[2]
    num_joints = 1

    beta_s = torch.tensor(np.zeros((1, num_shape_params)) + 1e-5, dtype=sculptor_model.dtype, requires_grad=True)
    pose_theta = torch.tensor(np.tile(np.zeros(3 * (num_joints + 1)).reshape(1, 3 * (num_joints + 1)),(1, 1)), dtype=sculptor_model.dtype, requires_grad=True)
    jaw_offset = torch.tensor(np.zeros((1, 3, 1)), dtype=sculptor_model.dtype, requires_grad=True)

    # Optimizer
    optimizer = torch.optim.AdamW([beta_s, pose_theta, jaw_offset], lr=lr)

    print("Starting optimization...")
    # Optimization loop
    for iter_num in range(num_iterations):
        optimizer.zero_grad()

        fitted_mesh_vertices = sculptor_model(beta_s, pose_theta, jaw_offset)

        loss_chamfer = custom_chamfer_distance(fitted_mesh_vertices, target_vertices)
        
        loss_reg_shape = torch.mean(beta_s**2)
        loss_reg_pose = torch.mean(pose_theta**2) # Regularize pose parameters
        
        loss = loss_chamfer + shape_reg_weight * loss_reg_shape + pose_reg_weight * loss_reg_pose

        loss.backward()
        optimizer.step()

        if iter_num % 10 == 0 or iter_num == num_iterations -1 :
            print(f"Iter {iter_num:04d}/{num_iterations-1} | Loss: {loss.item():.6f} "
                  f"(Chamfer: {loss_chamfer.item():.6f}, ShapeReg: {loss_reg_shape.item():.6f}, PoseReg: {loss_reg_pose.item():.6f})")

    # Get final results
    optimized_params_dict = {
        'beta_s': beta_s.detach().cpu().numpy(),
        'pose_theta': pose_theta.detach().cpu().numpy(),
        'jaw_offset': jaw_offset.detach().cpu().numpy()
    }
    final_fitted_mesh_verts = sculptor_model(beta_s, pose_theta, jaw_offset).detach().cpu()

    # Save optimized parameters
    params_path = os.path.join(output_dir, "optimized_sculptor_params.npz")
    np.savez(params_path, **optimized_params_dict)
    print(f"Saved optimized parameters to {params_path}")

    # Save the fitted facial mesh
    num_skull_verts = sculptor_model.template_skull.shape[0]
    facial_mesh_vertices_fitted = final_fitted_mesh_verts[0, num_skull_verts:].numpy()
    # sculptor_model.facialmesh_face contains 0-indexed faces for the facial part
    facial_mesh_faces = sculptor_model.facialmesh_face 
    
    output_face_obj_path = os.path.join(output_dir, "fitted_face_mesh.obj")
    save_mesh_to_obj(facial_mesh_vertices_fitted, facial_mesh_faces, output_face_obj_path)
    print(f"Saved fitted facial mesh to {output_face_obj_path}")

    # Save the full fitted mesh (skull + face)
    full_fitted_vertices = final_fitted_mesh_verts[0].numpy()
    skull_faces = sculptor_model.skullmesh_face
    # Offset face indices by number of skull vertices for the combined mesh
    face_faces_offset = sculptor_model.facialmesh_face + num_skull_verts 
    full_faces = np.vstack([skull_faces, face_faces_offset])
    
    output_full_obj_path = os.path.join(output_dir, "fitted_full_mesh.obj")
    save_mesh_to_obj(full_fitted_vertices, full_faces, output_full_obj_path)
    print(f"Saved full fitted mesh (skull+face) to {output_full_obj_path}")

    print("Fitting complete.")
    return optimized_params_dict, final_fitted_mesh_verts


def main():
    target_mesh = './data/test/target_mesh.ply'
    paradict_path = './dependencies/sculptor/model/paradict.npy'
    output_dir = './data/test/fitted_results'
    
    num_iterations = 200
    lr = 0.01
    shape_reg = 0.001
    pose_reg = 0.001

    fit_sculptor_to_target(
        target_mesh_path=target_mesh,
        paradict_path=paradict_path,
        output_dir=output_dir,
        num_iterations=num_iterations,
        lr=lr,
        shape_reg_weight=shape_reg,
        pose_reg_weight=pose_reg
    )

if __name__ == '__main__':
    main() 
'''