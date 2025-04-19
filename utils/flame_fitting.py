import os
import numpy as np
import torch
from tqdm import tqdm
from psbody.mesh import Mesh
from config import model_path

# Import FLAME from PyTorch implementation
from flame_pytorch import FLAME, get_config

def fit_3D_mesh(batch_size, target_3d_mesh_fname, weights, device='cuda', show_fitting=True):
    '''
    Fit FLAME to 3D mesh in correspondence to the FLAME mesh (i.e. same number of vertices, same mesh topology)
    :param target_3d_mesh_fname:    target 3D mesh filename
    :param model_fname:             saved FLAME model
    :param weights:                 weights of the individual objective functions
    :param device:                  device to run optimization on ('cuda' or 'cpu')
    :return: a mesh with the fitting results
    '''

    # read the given mesh, and put its vertices into a tensor
    mesh_fnames = os.listdir(target_3d_mesh_fname)
    mesh_fnames.sort()
    target_meshes = [Mesh(filename=os.path.join(target_3d_mesh_fname, f)) for f in mesh_fnames]

    vertices_nparray = np.array([m.v for m in target_meshes])
    target_vertices = torch.tensor(vertices_nparray, dtype=torch.float32).to(device)

    # Set up FLAME model
    config = get_config()
    config.flame_model_path = model_path.flame_model_path
    config.static_landmark_embedding_path = model_path.flame_static_embedding_path
    config.dynamic_landmark_embedding_path = model_path.flame_dynamic_embedding_path
    config.batch_size = batch_size

    flame_model = FLAME(config).to(device)

    # Initialize parameters to optimize
    trans = torch.zeros(batch_size, 3, requires_grad=True, device=device)
    pose = torch.zeros(batch_size, config.pose_params, requires_grad=True, device=device)
    shape = torch.zeros(batch_size, config.shape_params, requires_grad=True, device=device)
    expression = torch.zeros(batch_size, config.expression_params, requires_grad=True, device=device)

    if config.optimize_neckpose:
        neck_pose = torch.zeros(batch_size, 3, requires_grad=False, device=device)
    else:
        neck_pose = None
            
    if config.optimize_eyeballpose:
        eye_pose = torch.zeros(batch_size, 6, requires_grad=False, device=device)
    else:
        eye_pose = None
    
    # First optimize only the global transformation
    # rigid_optimizer = LBFGS([trans, pose], lr=1.0, max_iter=100, line_search_fn='strong_wolfe')
    rigid_optimizer = torch.optim.Adam([trans, pose], lr=0.001)

    def rigid_closure():
        rigid_optimizer.zero_grad()
        
        vertices, _ = flame_model(shape, expression, pose, neck_pose, eye_pose, transl=trans)
        
        # Compute data term (vertex distance)
        loss = torch.sum((vertices - target_vertices) ** 2)
        
        loss.backward()
        pbar.set_description(f"Loss: {loss.item():.6f}")
        return loss

    print('STAGE 1: Optimize rigid transformation')
    pbar = tqdm(range(5000))
    for i in pbar:
        rigid_optimizer.step(rigid_closure)
        
    # Then optimize all parameters
    # optimizer = LBFGS([trans, pose, shape, expression], lr=1.0, max_iter=100, line_search_fn='strong_wolfe')
    optimizer = torch.optim.Adam([trans, pose, shape, expression], lr=0.001)

    def closure():
        optimizer.zero_grad()
        
        vertices, _ = flame_model(shape, expression, pose, neck_pose, eye_pose, transl=trans)
        
        # Compute data term (vertex distance)
        mesh_dist = torch.sum((vertices - target_vertices) ** 2)
        
        # Regularization terms
        neck_pose_reg = torch.sum(neck_pose ** 2)
        jaw_pose_reg = torch.sum(pose ** 2)
        eyeballs_pose_reg = torch.sum(eye_pose ** 2)
        shape_reg = torch.sum(shape ** 2)
        exp_reg = torch.sum(expression ** 2)
        
        # Weighted loss
        loss = weights['data'] * mesh_dist + \
               weights['shape'] * shape_reg + \
               weights['expr'] * exp_reg + \
               weights['neck_pose'] * neck_pose_reg + \
               weights['jaw_pose'] * jaw_pose_reg + \
               weights['eyeballs_pose'] * eyeballs_pose_reg
        
        loss.backward()
        pbar.set_description(f"Loss: {loss.item():.6f}")
        return loss

    print('STAGE 2: Optimize model parameters')

    pbar = tqdm(range(10000))
    for i in pbar:
        optimizer.step(closure)

    print('Fitting done')

    # Get the final mesh
    with torch.no_grad():
        neutral_pose = torch.zeros(batch_size, config.pose_params, device=device)
        vertices, _ = flame_model(shape, expression, neutral_pose, neck_pose, eye_pose)
        vertices = vertices.cpu().numpy()

    # TODO: Visualize the result mesh
    '''
    if show_fitting:
        # Visualize fitting
        mv = MeshViewer()
        fitting_mesh = Mesh(vertices, flame_model.faces)
        fitting_mesh.set_vertex_colors('light sky blue')

        mv.set_static_meshes([target_mesh, fitting_mesh])
        input('Press key to continue')
    '''

    return vertices, flame_model.faces, shape, expression

def run_corresponding_mesh_fitting():
    # target 3D mesh in dense vertex-correspondence to the model
    target_mesh_path = '../data/FaMoS/'

    # Output filename
    out_mesh_fname = '../processed_data/'

    weights = {}
    # Weight of the data term
    weights['data'] = 1000.0
    # Weight of the shape regularizer (the lower, the less shape is constrained)
    weights['shape'] = 1e-4
    # Weight of the expression regularizer (the lower, the less expression is constrained)
    weights['expr'] = 1e-4
    # Weight of the neck pose (i.e. neck rotation around the neck) regularizer (the lower, the less neck pose is constrained)
    weights['neck_pose'] = 1e-4
    # Weight of the jaw pose (i.e. jaw rotation for opening the mouth) regularizer (the lower, the less jaw pose is constrained)
    weights['jaw_pose'] = 1e-4
    # Weight of the eyeball pose (i.e. eyeball rotations) regularizer (the lower, the less eyeballs pose is constrained)
    weights['eyeballs_pose'] = 1e-4

    """
    # Show landmark fitting (default: red = target landmarks, blue = fitting landmarks)
    show_fitting = True
    """

    # Check if CUDA is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    subject_list = os.listdir(target_mesh_path)
    subject_list.sort()

    for subject in subject_list:
        print(f"Processing subject: {subject}")
        exp_list = os.listdir(os.path.join(target_mesh_path, subject))
        exp_list.sort()
        for exp in exp_list:
            print(f"Processing expression: {exp}")
            frame_count = len(os.listdir(os.path.join(target_mesh_path, subject, exp)))
            result_vertices, result_faces, result_shape, result_exp = fit_3D_mesh(frame_count, os.path.join(target_mesh_path, subject, exp), weights, device=device)
            result_shape = result_shape.detach().cpu().numpy()
            result_exp = result_exp.detach().cpu().numpy()

            if not os.path.exists(os.path.dirname(os.path.join(out_mesh_fname, subject, exp))):
                os.makedirs(os.path.dirname(os.path.join(out_mesh_fname, subject, exp)))
            
            for i in range(frame_count):
                result_mesh = Mesh(v=result_vertices[i], f=result_faces)
                result_mesh.write_ply(os.path.join(out_mesh_fname, subject, exp, f'mesh_{i}.ply'))
                np.savez(os.path.join(out_mesh_fname, subject, exp, f'parameters_{i}.npz'), shape=result_shape[i], expression=result_exp[i])

if __name__ == '__main__':
    run_corresponding_mesh_fitting() 