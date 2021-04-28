import sys
import os
import numpy as np
import h5py
sys.path.append('./utils_motion')
from Animation import Animation, positions_global
from Quaternions import Quaternions
from BVH import save
from skeleton import Skeleton
import argparse

offsets = np.array([
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
    ], dtype='float64') * 0.01

parents = np.array([-1,  0,  1,  2,  3,  4,  0,  6,  7,  8,  9,  0, 11, 12, 13, 14, 12,
       16, 17, 18, 19, 20, 19, 22, 12, 24, 25, 26, 27, 28, 27, 30], dtype='int64')
joints_left = np.array([1, 2, 3, 4, 5, 24, 25, 26, 27, 28, 29, 30, 31], dtype='int64')
joints_right = np.array([6, 7, 8, 9, 10, 16, 17, 18, 19, 20, 21, 22, 23], dtype='int64')

orients = Quaternions.id(1)
orients_final = np.array([[1,0,0,0]]).repeat(len(offsets), axis=0)
orients.qs = np.append(orients.qs, orients_final, axis=0)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--bvh_dir', 
                        type=str, 
                        default='./pretrained/output/recon/bvh')
    parser.add_argument('--hdf5_path', 
                        type=str, 
                        default='./pretrained/output/recon/m_recon.hdf5')
    args = parser.parse_args()

    file_dir = args.bvh_dir
    for jj in range(60):    # # of test motions: 60
        with h5py.File(args.hdf5_path, 'r') as h5f:
            rotations = h5f['batch{0}'.format(jj + 1)][:]   # (fnum, n_joint, 4)
            rotations = rotations[:-10]  # drop the last few frames
            fnum = rotations.shape[0]
            positions = offsets[np.newaxis].repeat(fnum, axis=0)
        
        rotations_Quat = Quaternions(rotations)
        anim = Animation(rotations_Quat, positions, orients, offsets, parents)

        xyz = positions_global(anim)
        height_offset = np.min(xyz[:, :, 1]) # Min height
        positions[:, :, 1] -= height_offset
        anim.positions = positions

        filename = 'batch{0}.bvh'.format(jj+1)
        filepath = os.path.join(file_dir, filename)

        try:
            if not(os.path.isdir(file_dir)):
                print("Creating directory: {}".format(file_dir))
                os.makedirs(file_dir)
        except OSError:
            pass

        save(filepath, anim, frametime=1.0/24.0)














