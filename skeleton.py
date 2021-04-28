import torch
import numpy as np
from utils import qmul_np, qmul, qrot, expmap2rotmat_tensor, quat2rotmat
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Skeleton:
    def __init__(self, offsets, parents, joints_left=None, joints_right=None):
        assert len(offsets) == len(parents)

        self._offsets = torch.FloatTensor(offsets).to(device)
        self._parents = np.array(parents)
        self._joints_left =  np.array(joints_left)
        self._joints_right = np.array(joints_right)
        self._compute_metadata()

    def num_joints(self):
        return self._offsets.shape[0]

    def offsets(self):
        return self._offsets

    def parents(self):
        return self._parents

    def has_children(self):
        return self._has_children

    def children(self):
        return self._children

    def remove_joints(self, joints_to_remove, dataset):
        """
        Remove the joints specified in 'joints_to_remove', both from the
        skeleton definition and from the dataset (which is modified in place).
        The rotations of removed joints are propagated along the kinematic chain.
        """
        valid_joints = []
        for joint in range(len(self._parents)):
            if joint not in joints_to_remove:
                valid_joints.append(joint)

        # Update all transformations in the dataset
        for subject in dataset.subjects():
            for action in dataset[subject].keys():
                rotations = dataset[subject][action]['rotations']
                for joint in joints_to_remove:
                    for child in self._children[joint]:
                        rotations[:, child] = qmul_np(rotations[:, joint], rotations[:, child])
                    rotations[:, joint] = [1, 0, 0, 0] # Identity
                dataset[subject][action]['rotations'] = rotations[:, valid_joints]

        index_offsets = np.zeros(len(self._parents), dtype=int)
        new_parents = []
        for i, parent in enumerate(self._parents):
            if i not in joints_to_remove:
                new_parents.append(parent - index_offsets[parent])
            else:
                index_offsets[i:] += 1
        self._parents = np.array(new_parents)

        self._offsets = self._offsets[valid_joints]
        self._compute_metadata()

    def forward_kinematics_quat(self, rotations, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rotations_world.append(rotations[:, :, 0])
            else:
                positions_world.append(qrot(rotations_world[self._parents[i]], expanded_offsets[:, :, i]) \
                                       + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(qmul(rotations_world[self._parents[i]], rotations[:, :, i]))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def forward_kinematics_exp(self, angles, root_positions):
        """
        Perform forward kinematics using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- angles: (N, L, J, 3) tensor of exp map describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        if len(angles.shape) == 3:
            angles = angles.view(angles.shape[0], angles.shape[1], -1, 3)

        assert len(angles.shape) == 4
        assert angles.shape[-1] == 3

        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(angles.shape[0], angles.shape[1],
                                                self._offsets.shape[0], self._offsets.shape[1])

        if not angles.shape[0] == root_positions.shape[0]:
            root_positions = torch.zeros(angles.shape[0], angles.shape[1], 3).to(device)
            # root_positions = root_positions.new(angles.size()).zero_()

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):

            if self._parents[i] == -1:
                positions_world.append(root_positions)
                r_root = angles[:, :, 0]
                thisRotation = expmap2rotmat_tensor(r_root)
                rotations_world.append(thisRotation)     # (batch_size, seq, 9)
            else:
                r = angles[:, :, i]
                thisRotation = expmap2rotmat_tensor(r)  # (batch_size, seq, 9)
                thisRotation_ = thisRotation.view(-1, 3, 3)
                offset_ = expanded_offsets[:, :, i].view(-1, 1, 3)
                positions_world.append((torch.bmm(offset_, rotations_world[self._parents[i]].view(-1, 3, 3)))
                                       .view(angles.shape[0], angles.shape[1], 3) + positions_world[self._parents[i]])
                if self._has_children[i]:
                    rotations_world.append(( torch.bmm(thisRotation_, rotations_world[self._parents[i]].view(-1, 3, 3)))
                                           .view(angles.shape[0], angles.shape[1], 9))
                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)

        return torch.stack(positions_world, dim=3).permute(0, 1, 3, 2)

    def compute_trasformation_matrix(self, rotations, root_positions):
        """
        Compute global homogeneous transformation matrices using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length, J = number of joints):
         -- rotations: (N, L, J, 4) tensor of unit quaternions describing the local rotations of each joint.
         -- root_positions: (N, L, 3) tensor describing the root joint positions.
        """
        assert len(rotations.shape) == 4
        assert rotations.shape[-1] == 4

        transformations_mat_world = []
        positions_world = []
        rotations_world = []

        expanded_offsets = self._offsets.expand(rotations.shape[0], rotations.shape[1],
                                                   self._offsets.shape[0], self._offsets.shape[1])

        # Parallelize along the batch and time dimensions
        for i in range(self._offsets.shape[0]):
            if self._parents[i] == -1:
                positions_world.append(root_positions)
                rot = rotations[:, :, 0]
                rot_mat = quat2rotmat(rot)  # (N, L, 3, 3)
                rotations_world.append(rot_mat)
                tl_mat = root_positions.unsqueeze(-1)
                tr_mat = torch.cat((rot_mat, tl_mat), -1)   # (N, L, 3, 4)
                transformations_mat_world.append(tr_mat.view(rotations.shape[0], rotations.shape[1], 12))

            else:
                rot = rotations[:, :, i]
                rot_mat = quat2rotmat(rot).view(-1, 3, 3)   # (N*L, 3, 3)
                offset_ = expanded_offsets[:, :, i].view(-1, 1, 3)

                pos_world = (torch.bmm(offset_, rotations_world[self._parents[i]].view(-1, 3, 3))).view(rotations.shape[0], rotations.shape[1], 3) + positions_world[self._parents[i]]
                positions_world.append(pos_world)
                tl_mat = pos_world.unsqueeze(-1)

                if self._has_children[i]:
                    rot_world = ( torch.bmm(rot_mat, rotations_world[self._parents[i]].view(-1, 3, 3))).view(rotations.shape[0], rotations.shape[1], 3, 3)
                    rotations_world.append(rot_world)

                else:
                    # This joint is a terminal node -> it would be useless to compute the transformation
                    rotations_world.append(None)
                    rot_world = torch.eye(3).expand(rotations.shape[0], rotations.shape[1], 3, 3).to(device)

                tr_mat = torch.cat((rot_world, tl_mat), -1)
                transformations_mat_world.append(tr_mat.view(rotations.shape[0], rotations.shape[1], 12))

        return torch.stack(transformations_mat_world, dim=3).permute(0, 1, 3, 2)

    def joints_left(self):
        return self._joints_left

    def joints_right(self):
        return self._joints_right

    def _compute_metadata(self):
        self._has_children = np.zeros(len(self._parents)).astype(bool)
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._has_children[parent] = True

        self._children = []
        for i, parent in enumerate(self._parents):
            self._children.append([])
        for i, parent in enumerate(self._parents):
            if parent != -1:
                self._children[parent].append(i)
