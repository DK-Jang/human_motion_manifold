import numpy as np
import torch
from utils import qeuler_np, qfix, qexp_np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MocapDataset:
    def __init__(self, path, skeleton, fps):
        self._data = self._load(path)
        self._fps = fps
        self._use_gpu = False
        self._skeleton = skeleton

    def cuda(self):
        self._use_gpu = True
        self._skeleton.cuda()
        return self
        
    def _load(self, path):
        result = {}
        data = np.load(path, 'r', allow_pickle=True)
        for i, (trajectory, rotations, subject, action) in enumerate(zip(data['trajectories'],
                                                                         data['rotations'],
                                                                         data['subjects'],
                                                                         data['actions'])):
            if subject not in result:
                result[subject] = {}
            
            result[subject][action] = {
                'rotations': rotations,
                'trajectory': trajectory
            }
        return result
        
    def downsample(self, factor, keep_strides=True):
        """
        Downsample this dataset by an integer factor, keeping all strides of the data
        if keep_strides is True.
        The frame rate must be divisible by the given factor.
        The sequences will be replaced by their downsampled versions, whose actions
        will have '_d0', ... '_dn' appended to their names.
        """
        assert self._fps % factor == 0
        
        for subject in self._data.keys():
            new_actions = {}
            for action in list(self._data[subject].keys()):
                for idx in range(factor):
                    tup = {}
                    for k in self._data[subject][action].keys():
                        tup[k] = self._data[subject][action][k][idx::factor]
                    new_actions[action + '_d' + str(idx)] = tup
                    if not keep_strides:
                        break
            self._data[subject] = new_actions
            
        self._fps //= factor
        
    def _mirror_sequence(self, sequence):
        mirrored_rotations = sequence['rotations'].copy()
        mirrored_trajectory = sequence['trajectory'].copy()
        
        joints_left = self._skeleton.joints_left()
        joints_right = self._skeleton.joints_right()
        
        # Flip left/right joints
        mirrored_rotations[:, joints_left] = sequence['rotations'][:, joints_right]
        mirrored_rotations[:, joints_right] = sequence['rotations'][:, joints_left]
        
        mirrored_rotations[:, :, [2, 3]] *= -1
        mirrored_trajectory[:, 0] *= -1

        return {
            'rotations': qfix(mirrored_rotations),
            'trajectory': mirrored_trajectory
        }
    
    def mirror(self):
        """
        Perform data augmentation by mirroring every sequence in the dataset.
        The mirrored sequences will have '_m' appended to the action name.
        """
        for subject in self._data.keys():
            for action in list(self._data[subject].keys()):
                if '_m' in action:
                    continue
                self._data[subject][action + '_m'] = self._mirror_sequence(self._data[subject][action])
                
    def compute_euler_angles(self, order):
        for subject in self._data.values():
            for action in subject.values():
                action['rotations_euler'] = qeuler_np(action['rotations'], order, use_gpu=self._use_gpu)

    def compute_exp_angles(self):
        for subject in self._data.values():
            for action in subject.values():
                action['rotations_exp'] = qexp_np(action['rotations'], use_gpu=self._use_gpu)
                
    def compute_positions_quat(self):   # joint angle (quaternion) to positions
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0).to(device)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0).to(device)
                action['positions_world'] = self._skeleton.forward_kinematics_quat(rotations, trajectory).squeeze(0).cpu().numpy()
                
                # Absolute translations across the XY plane are removed here -> only Y axis (height)
                trajectory[:, :, [0, 2]] = 0
                action['positions_local'] = self._skeleton.forward_kinematics_quat(rotations, trajectory).squeeze(0).cpu().numpy()

    def compute_positions_exp(self):    # joint angle (exponential coordiante) to positions
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations_exp'].astype('float32')).unsqueeze(0).to(device)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0).to(device)
                action['positions_world'] = self._skeleton.forward_kinematics_exp(rotations, trajectory).squeeze(
                    0).cpu().numpy()

                # Absolute translations across the XY plane are removed here
                trajectory[:, :, [0, 2]] = 0
                action['positions_local'] = self._skeleton.forward_kinematics_exp(rotations, trajectory).squeeze(
                    0).cpu().numpy()

    def compute_transformation_quat(self):
        for subject in self._data.values():
            for action in subject.values():
                rotations = torch.from_numpy(action['rotations'].astype('float32')).unsqueeze(0).to(device)
                trajectory = torch.from_numpy(action['trajectory'].astype('float32')).unsqueeze(0).to(device)
                action['transformations_world'] = self._skeleton.compute_trasformation_matrix(rotations, trajectory).squeeze(0).cpu().numpy()

                # Absolute translations across the XY plane are removed here
                trajectory[:, :, [0, 2]] = 0
                action['transformations_local'] = self._skeleton.compute_trasformation_matrix(rotations, trajectory).squeeze(0).cpu().numpy()

    def normalized_stats(self, sequences, representation):
        """
        Normalize the datasets.

        Args
            sequences: list of motion sequence's name
            representation: 'rotations', 'rotations_exp', 'positions_world', 'positions_local', 'transformations'
        Returns
            data_mean:
            data_std:
            dim_to_use: vector with dimensions not used by the model
            dim_to_ignore: vector with dimensions used by the model
        """
        for i, (subject, action) in enumerate(sequences):
            if i == 0:
                data = self._data[subject][action][representation]
            else:
                data = np.append(data, self._data[subject][action][representation], axis=0)

        data = np.reshape(data, (data.shape[0], -1))
        data_mean = np.mean(data, axis=0).astype(np.float32)
        data_std = np.std(data, axis=0).astype(np.float32)

        dim_to_ignore = []
        dim_to_use = []

        dim_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
        dim_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
        data_std[dim_to_ignore] = 1.0

        return data_mean, data_std, dim_to_use, dim_to_ignore

    def normalize_data(self, representation, data_mean, data_std, dim_to_use=None):
        eps = 1e-15

        for subject in self._data.values():
            for action in subject.values():
                data = np.reshape(action[representation], (action[representation].shape[0], -1))
                norm_data = (np.divide((data - data_mean), data_std+eps)).astype(np.float32)
                if dim_to_use:
                    norm_data = norm_data[:, dim_to_use]
                action[representation] = np.reshape(norm_data, (action[representation].shape[0], -1,
                                                                action[representation].shape[-1]))
                
    def __getitem__(self, key):
        return self._data[key]

    def subjects(self):
        return self._data.keys()

    def subject_actions(self, subject):
        return self._data[subject].keys()

    def all_actions(self):
        result = []
        for subject, actions in self._data.items():
            for action in actions.keys():
                result.append((subject, action))
        return result

    def fps(self):
        return self._fps

    def skeleton(self):
        return self._skeleton