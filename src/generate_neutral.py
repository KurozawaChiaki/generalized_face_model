import os
import numpy as np
import torch
from psbody.mesh import Mesh
import trimesh
import argparse

import sys
sys.path.append('.')
from dependencies.FLAME_PyTorch.flame_pytorch.flame import FLAME

def generate_neutral_mesh(output_path, device='cuda'):
    sculptor_para = np.load('model/paradict.npy',allow_pickle=True).item()

    template_skull = sculptor_para['template_skull']
    template_face = sculptor_para['template_face']

    face_meshes = sculptor_para['facialmesh_face']
    skull_meshes = sculptor_para['skullmesh_face']

    temp_vertices_index = face_meshes[:, 0]
    temp_vertices_index = temp_vertices_index.reshape(face_meshes.shape[0], 1)
    additional_meshes = np.concatenate([temp_vertices_index, face_meshes[:, 2:]], axis=1)
    face_meshes = np.concatenate([face_meshes[:, 0:3], additional_meshes], axis=0)

    result_mesh = Mesh(v=template_face, f=face_meshes)
    result_mesh.write_ply(os.path.join(output_path, 'skin.ply'))

    result_mesh = Mesh(v=template_skull, f=skull_meshes)
    result_mesh.write_ply(os.path.join(output_path, 'skull.ply'))

    config = {
        "flame_model_path": "./model/generic_model.pkl",
        "static_landmark_embedding_path": "./model/flame_static_embedding.pkl",
        "dynamic_landmark_embedding_path": "./model/flame_dynamic_embedding.npy",
        "shape_params": 100,
        "expression_params": 50,
        "pose_params": 6,
        "use_face_contour": True,
        "use_3D_translation": True,
        "optimize_eyeballpose": True,
        "optimize_neckpose": True,
        "batch_size": 1,
        "ring_margin": 0.5,
        "num_worker": 4,
        "ring_loss_weight": 1.0,
    }
    config = argparse.Namespace(**config)
    flame = FLAME(config).to(device)
    trans = torch.zeros(1, 3, requires_grad=True, device=device)
    pose = torch.zeros(1, config.pose_params, requires_grad=True, device=device)
    shape = torch.zeros(1, config.shape_params, requires_grad=True, device=device)
    expression = torch.zeros(1, config.expression_params, requires_grad=True, device=device)
    if config.optimize_neckpose:
        neck_pose = torch.zeros(1, 3, requires_grad=False, device=device)
    else:
        neck_pose = None
            
    if config.optimize_eyeballpose:
        eye_pose = torch.zeros(1, 6, requires_grad=False, device=device)
    else:
        eye_pose = None

    with torch.no_grad():
        vertices, _ = flame(shape, expression, pose, neck_pose, eye_pose, transl=trans)
        vertices = vertices.squeeze(0).cpu().numpy()
        faces = flame.faces

    print(f"vertices: {vertices.shape}, faces: {faces.shape}")

    result_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    result_mesh.export(os.path.join(output_path, 'flame_neutral.ply'))

if __name__ == '__main__':
    generate_neutral_mesh('./sculptor_utils/results/')