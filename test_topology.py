import json
import torch
import numpy as np
import os
import trimesh
from tqdm import tqdm
import torch.nn.functional as F
from typing import Optional
import pyrender

from libs.sculptor.model.sculptor import SCULPTOR_layer
from flame_pytorch.flame import FLAME
from flame_pytorch.config import get_config


def main():
    output_dir = "./testing"
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    general_config = json.load(open("./config/general.json"))

    config = get_config()
    config.shape_params = 300
    config.expression_params = 100
    config.flame_model_path = "./models/flame/generic_model.pkl"
    config.static_landmark_embedding_path = "./models/flame/flame_static_embedding.pkl"
    config.dynamic_landmark_embedding_path = (
        "./models/flame/flame_dynamic_embedding.npy"
    )
    config.batch_size = 1

    flame_model = FLAME(config).to(device)

    trans = torch.zeros(1, 3, requires_grad=True, device=device)
    pose = torch.zeros(1, config.pose_params, requires_grad=True, device=device)
    shape = torch.zeros(1, config.shape_params, requires_grad=True, device=device)
    expression = torch.zeros(
        1, config.expression_params, requires_grad=True, device=device
    )

    with torch.no_grad():
        flame_vertices, _ = flame_model(
            shape_params=shape,
            expression_params=expression,
            pose_params=pose,
            transl=trans,
        )

    flame_vertices = flame_vertices.cpu().numpy().squeeze(0)
    flame_faces = flame_model.faces
    flame_color = np.ones([flame_vertices.shape[0], 4]) * [0.3, 0.3, 0.3, 0.8]
    flame_mesh = trimesh.Trimesh(flame_vertices, flame_faces, vertex_colors=flame_color)

    flame_centroid = flame_mesh.centroid
    scale = 1000
    flame_vertices = np.matmul(flame_vertices - flame_centroid, scale * np.eye(3))
    flame_mesh = trimesh.Trimesh(flame_vertices, flame_faces, vertex_colors=flame_color)
    flame_mesh.export("./testing/flame_mesh.ply")

    sculptor_model = SCULPTOR_layer(general_config["sculptor_config"]["model_path"])
    sculptor_vertices = sculptor_model.template_face
    sculptor_skull_vertices = sculptor_model.template_skull
    sculptor_faces = sculptor_model.facialmesh_face
    sculptor_skull_faces = sculptor_model.skullmesh_face

    temp_vertices_index = sculptor_faces[:, 0].reshape(sculptor_faces.shape[0], 1)
    additional_meshes = np.concatenate(
        [temp_vertices_index, sculptor_faces[:, 2:]], axis=1
    )
    sculptor_faces = np.concatenate([sculptor_faces[:, 0:3], additional_meshes], axis=0)

    sculptor_color = np.ones([flame_vertices.shape[0], 4]) * [0.6, 0.6, 0.6, 0.8]

    sculptor_mesh = trimesh.Trimesh(
        sculptor_vertices, sculptor_faces, vertex_colors=sculptor_color, process=False
    )
    sculptor_centroid = sculptor_mesh.centroid
    print(f"Sculptor centroid: {sculptor_centroid.shape}")
    sculptor_vertices = sculptor_vertices - sculptor_centroid

    sculptor_mesh = trimesh.Trimesh(
        sculptor_vertices, sculptor_faces, vertex_colors=sculptor_color, process=False
    )
    sculptor_skull_mesh = trimesh.Trimesh(
        sculptor_skull_vertices,
        sculptor_skull_faces,
        vertex_colors=sculptor_color,
        process=False,
    )

    sculptor_mesh.export("./testing/sculptor_mesh.obj")
    sculptor_skull_mesh.export("./testing/sculptor_skull_mesh.obj")

    sculptor_mesh_render = pyrender.Mesh.from_trimesh(sculptor_mesh)


if __name__ == "__main__":
    main()
