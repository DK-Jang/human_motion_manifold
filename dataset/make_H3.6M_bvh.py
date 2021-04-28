import sys
import os
import numpy as np
sys.path.append('../utils_motion')
sys.path.append('../')
from Animation import Animation
from Quaternions import Quaternions
from BVH import save
from mocap_dataset import MocapDataset
from skeleton import Skeleton

skeleton_H36M = Skeleton(offsets=[
       [   0.      ,    0.      ,    0.      ],
       [-132.948591,    0.      ,    0.      ],
       [   0.      , -442.894612,    0.      ],
       [   0.      , -454.206447,    0.      ],
       [   0.      ,    0.      ,  162.767078],
       [   0.      ,    0.      ,   74.999437],
       [ 132.948826,    0.      ,    0.      ],
       [   0.      , -442.894413,    0.      ],
       [   0.      , -454.20659 ,    0.      ],
       [   0.      ,    0.      ,  162.767426],
       [   0.      ,    0.      ,   74.999948],
       [   0.      ,    0.1     ,    0.      ],
       [   0.      ,  233.383263,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  121.134938,    0.      ],
       [   0.      ,  115.002227,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  151.034226,    0.      ],
       [   0.      ,  278.882773,    0.      ],
       [   0.      ,  251.733451,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,    0.      ,   99.999627],
       [   0.      ,  100.000188,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,  257.077681,    0.      ],
       [   0.      ,  151.031437,    0.      ],
       [   0.      ,  278.892924,    0.      ],
       [   0.      ,  251.72868 ,    0.      ],
       [   0.      ,    0.      ,    0.      ],
       [   0.      ,    0.      ,   99.999888],
       [   0.      ,  137.499922,    0.      ],
       [   0.      ,    0.      ,    0.      ]
    ],
    parents=[-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30],
    joints_left=[1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31],
    joints_right=[6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23])

dataset_path_H36M = './dataset_h36m.npz'
dataset_H36M = MocapDataset(dataset_path_H36M, skeleton_H36M, fps=50)
dataset_H36M.mirror()
dataset_H36M.compute_exp_angles()
dataset_H36M.downsample(2)

offsets = skeleton_H36M.offsets().numpy()
parents = skeleton_H36M.parents()
joints_left = skeleton_H36M.joints_left()
joints_right = skeleton_H36M.joints_right()

# npz to bvh
orients = Quaternions.id(1)
orients_final = np.array([[1,0,0,0]]).repeat(len(offsets), axis=0)
orients.qs = np.append(orients.qs, orients_final, axis=0)
dataset_dir = './bvh'

for subject in dataset_H36M._data.keys():
    for action in list(dataset_H36M._data[subject].keys()):
        fnum = dataset_H36M._data[subject][action]['rotations'].shape[0]
        positions = offsets[np.newaxis].repeat(fnum, axis=0)
        rotations = np.zeros((fnum, len(orients), 3))

        positions[:, 0:1] = dataset_H36M._data[subject][action]['trajectory'][:, np.newaxis, :]
        rotations = dataset_H36M._data[subject][action]['rotations']
        rotations_Quat = Quaternions(rotations)
        anim = Animation(rotations_Quat, positions, orients, offsets, parents)
        filename = action + '.bvh'
        filepath = os.path.join(dataset_dir, subject, filename)

        try:
            if not(os.path.isdir(os.path.join(dataset_dir, subject))):
                os.makedirs(os.path.join(dataset_dir, subject))
        except OSError:
            pass

        save(filepath, anim)













