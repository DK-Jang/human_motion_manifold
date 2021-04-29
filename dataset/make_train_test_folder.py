import sys
sys.path.append('../')
from mocap_dataset import MocapDataset
from skeleton import Skeleton
import numpy as np
import os

# Set to True for validation, set to False for testing
perform_validation = True

if perform_validation:
    subjects_train_H36M = ['S1', 'S7', 'S8', 'S9', 'S11']
    subjects_valid_H36M = ['S6']
    subjects_test_H36M = ['S5']
else:
    subjects_train_H36M = ['S1', 'S6', 'S7', 'S8', 'S9', 'S11']
    subjects_valid_H36M = []
    subjects_test_H36M = ['S5']

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

########################################################################
##----------------------- Setup dataset ------------------------------##
########################################################################

# Setup datasets
sequences_train = []
sequences_valid = []
sequences_test = []
n_discarded = 0
sequence_length = 150

for subject in subjects_train_H36M:
    for action in dataset_H36M[subject].keys():
        if dataset_H36M[subject][action]['rotations'].shape[0] < sequence_length:
            n_discarded += 1
            continue
        sequences_train.append((subject, action))

for subject in subjects_valid_H36M:
    for action in dataset_H36M[subject].keys():
        if dataset_H36M[subject][action]['rotations'].shape[0] < sequence_length:
            n_discarded += 1
            continue
        sequences_valid.append((subject, action))

for subject in subjects_test_H36M:
    for action in dataset_H36M[subject].keys():
        if dataset_H36M[subject][action]['rotations'].shape[0] < sequence_length:
            n_discarded += 1
            continue
        sequences_test.append((subject, action))

print('%d sequences were discarded for being too short.' % n_discarded)
print('Training on %d sequences, validating on %d sequences, testing on %d sequences.'
      % (len(sequences_train), len(sequences_valid), len(sequences_test)))

dataset_H36M.compute_positions_quat()
exp_mean, exp_std, exp_dimTouse, exp_dimToignore = dataset_H36M.normalized_stats(sequences_train, 'rotations_exp')
pos_mean, pos_std, pos_dimTouse, pos_dimToignore = dataset_H36M.normalized_stats(sequences_train, 'positions_local')
dataset_H36M.normalize_data('rotations_exp', exp_mean, exp_std, exp_dimTouse)
dataset_H36M.normalize_data('positions_local', pos_mean, pos_std, pos_dimTouse)

np.save('./' + 'exp_mean.npy', exp_mean)
np.save('./' + 'exp_std.npy', exp_std)
np.save('./' + 'pos_mean.npy', pos_mean)
np.save('./' + 'pos_std.npy', pos_std)
np.save('./' + 'exp_dimTouse.npy', exp_dimTouse)
np.save('./' + 'pos_dimTouse.npy', pos_dimTouse)

# For exp coordinate
train_dataset_dir = './exp/train/'
valid_dataset_dir = './exp/valid/'
test_dataset_dir = './exp/test/'

for i, (subject, action) in enumerate(sequences_train):
    data = dataset_H36M._data[subject][action]['rotations_exp']
    if not os.path.exists(train_dataset_dir):
        os.makedirs(train_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(train_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_valid):
    data = dataset_H36M._data[subject][action]['rotations_exp']
    if not os.path.exists(valid_dataset_dir):
        os.makedirs(valid_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(valid_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_test):
    data = dataset_H36M._data[subject][action]['rotations_exp']
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(test_dataset_dir, filename)
    np.save(filename, data)

# For local position
train_dataset_dir = './pos/train/'
valid_dataset_dir = './pos/valid/'
test_dataset_dir = './pos/test/'

for i, (subject, action) in enumerate(sequences_train):
    data = dataset_H36M._data[subject][action]['positions_local']
    if not os.path.exists(train_dataset_dir):
        os.makedirs(train_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(train_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_valid):
    data = dataset_H36M._data[subject][action]['positions_local']
    if not os.path.exists(valid_dataset_dir):
        os.makedirs(valid_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(valid_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_test):
    data = dataset_H36M._data[subject][action]['positions_local']
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(test_dataset_dir, filename)
    np.save(filename, data)

# For trajectory
train_dataset_dir = './trajectory/train/'
valid_dataset_dir = './trajectory/valid/'
test_dataset_dir = './trajectory/test/'

for i, (subject, action) in enumerate(sequences_train):
    data = dataset_H36M._data[subject][action]['trajectory']
    data[:, [0, 2]] = 0
    if not os.path.exists(train_dataset_dir):
        os.makedirs(train_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(train_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_valid):
    data = dataset_H36M._data[subject][action]['trajectory']
    data[:, [0, 2]] = 0
    if not os.path.exists(valid_dataset_dir):
        os.makedirs(valid_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(valid_dataset_dir, filename)
    np.save(filename, data)

for i, (subject, action) in enumerate(sequences_test):
    data = dataset_H36M._data[subject][action]['trajectory']
    data[:, [0, 2]] = 0
    if not os.path.exists(test_dataset_dir):
        os.makedirs(test_dataset_dir)
    filename = subject + '_' + action + '.npy'
    filename = os.path.join(test_dataset_dir, filename)
    np.save(filename, data)
