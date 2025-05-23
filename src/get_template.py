import os
import numpy as np
import torch
from tqdm import tqdm
from scipy.spatial import cKDTree
import pickle
from psbody.mesh import Mesh
import trimesh

import sys
sys.path.append('.')
from dependencies.sculptor.model.sculptor import SCULPTOR_layer

def get_template():
    '''
    sculptor_para = np.load('model/paradict.npy',allow_pickle=True).item()
    template_skull = sculptor_para['template_skull']
    template_face = sculptor_para['template_face']
    '''
    sculptor_model = SCULPTOR_layer('./model/paradict.npy')
    template_skull = sculptor_model.template_skull
    template_face = sculptor_model.template_face

    print(template_skull.shape)
    print(template_face.shape)

if __name__ == '__main__':
    get_template()