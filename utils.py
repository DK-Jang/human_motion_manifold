import numpy as np
import torch
import torch.nn.init as init
from torch.optim import lr_scheduler
import h5py
import os
import math
import yaml
import shutil
import random
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

##################################################################################
# Quaternion
##################################################################################
def qmul(q, r):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)

def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1+epsilon, 1-epsilon))
    elif order == 'zxy':
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2*(q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1+epsilon, 1-epsilon))
    elif order == 'yxz':
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1+epsilon, 1-epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2*(q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2*(q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2*(q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1+epsilon, 1-epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2*(q2 * q2 + q3 * q3))
    # else:
    #     raise

    return torch.stack((x, y, z), dim=1).view(original_shape)

def qexp(q, epsilon=1e-8):
    """
        Convert quaternion(s) q to exponential map
        Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
        Returns a tensor of shape (*, 3).

        denote the axis of rotation by unit vector r0, the angle by theta
        q is of the form (cos(theta/2), r0*sin(theta/2))
        r is of the form r0*theta
    """

    # pi = torch.from_numpy(np.pi).to(device)
    assert q.shape[-1] == 4
    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    if (np.abs(torch.norm(q, 2, 1)-1) > 1e-3).all():
        raise(ValueError, "quat2expmap: input quaternion is not norm 1")

    # w = q[:, 0]
    # v = q[:, 1:]
    # theta = torch.acos(w) * 2.0
    # r0 = torch.div(v, (torch.norm(v, 2, 1) + epsilon).view(-1, 1))

    sinhalftheta = torch.norm(q[:, 1:], 2, 1)
    coshalftheta = q[:, 0]
    r0 = torch.div(q[:, 1:], (sinhalftheta + epsilon).view(-1, 1))
    theta = 2 * torch.atan2( sinhalftheta, coshalftheta )
    theta = torch.remainder( theta + 2*np.pi, 2*np.pi )

    idx = []
    for index, item in enumerate(theta):
        if item > np.pi:
            idx.append(index)

    theta[idx] = 2 * np.pi - theta[idx]
    r0[idx] = -r0[idx]

    r = torch.mul(r0, theta.view(-1, 1))

    return r.view(original_shape)

def qmul_np(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return qmul(q, r).numpy()

def qrot_np(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return qrot(q, v).numpy()

def qeuler_np(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qeuler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qeuler(q, order, epsilon).numpy()

def qexp_np(q, epsilon=1e-8, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return qexp(q, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return qexp(q, epsilon).numpy()

def qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:]*q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0)%2).astype(bool)
    result[1:][mask] *= -1
    return result

def expmap_to_quaternion(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5*theta).reshape(-1, 1)
    xyz = 0.5*np.sinc(0.5*theta/np.pi)*e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)

def quat2rotmat(q):
    """
        Compute rotation matrices using the given trajectory and local rotations.
        Arguments (where N = batch size, L = sequence length):
         -- q: (N, L, 4) tensor of unit quaternions describing the local rotations of joint.
         -- q: (*, 4), where * denotes any number of dimensions.
        Returns
         -- R: (N, L, 3, 3) rotation matrices
         -- R: (*, 3, 3), where * denotes any number of dimensions.
    """
    check = torch.norm(q, 2, 2) - 1
    if len(torch.nonzero(check > 1e-3)) > 0:
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    original_shape = list(q.size())
    original_shape[-1] = 3
    original_shape.append(3)    # (*, 4) -> (*, 3, 3)

    q = q.reshape(-1, 4)

    w, x, y, z = q[:,0], q[:,1], q[:,2], q[:,3]
    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w*x, w*y, w*z
    xy, xz, yz = x*y, x*z, y*z

    # ---------------- check out the column major and row major!! ------------------#
    # rotMat = torch.stack([w2 + x2 - y2 - z2, 2*xy - 2*wz, 2*wy + 2*xz,
    #                       2*wz + 2*xy, w2 - x2 + y2 - z2, 2*yz - 2*wx,
    #                       2*xz - 2*wy, 2*wx + 2*yz, w2 - x2 - y2 + z2], dim=1).reshape(original_shape)
    rotMat = torch.stack([w2 + x2 - y2 - z2, 2*wz + 2*xy, 2*xz - 2*wy,
                          2*xy - 2*wz, w2 - x2 + y2 - z2, 2*wx + 2*yz,
                          2*wy + 2*xz, 2*yz - 2*wx, w2 - x2 - y2 + z2], dim=1).reshape(original_shape)
    return rotMat

def quat2expmap(q):
    """
      Converts a quaternion to an exponential map
      Matlab port to python for evaluation purposes
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

      Args
        q: 1x4 quaternion, w, x, y, z
      Returns
        r: 1x3 exponential map
      Raises
        ValueError if the l2 norm of the quaternion is not close to 1
    """

    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


##################################################################################
# Exp, Euler and Rotation matrix
##################################################################################
def euler_to_quaternion(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x/2), np.sin(x/2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y/2), np.zeros_like(y), np.sin(y/2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z/2), np.zeros_like(z), np.zeros_like(z), np.sin(z/2)), axis=1)

    result = None
    for coord in order:
        if coord == 'x':
            r = rx
        elif coord == 'y':
            r = ry
        elif coord == 'z':
            r = rz
        # else:
        #     raise
        if result is None:
            result = r
        else:
            result = qmul_np(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ['xyz', 'yzx', 'zxy']:
        result *= -1

    return result.reshape(original_shape)

def rotmat2euler(R):
    """
      Converts a rotation matrix to Euler angles
      Matlab port to python for evaluation purposes
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

      Args
        R: a 3x3 rotation matrix
      Returns
        eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul

def rotmat2quat(R):
    """
      Converts a rotation matrix to a quaternion
      Matlab port to python for evaluation purposes
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

      Args
        R: 3x3 rotation matrix
      Returns
        q: 1x4 quaternion
    """
    rotdiff = R - R.T;

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps);

    costheta = (np.trace(R) - 1) / 2;

    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R));


def expmap2rotmat(r):
    """
      Converts an exponential map angle to a rotation matrix
      Matlab port to python for evaluation purposes
      I believe this is also called Rodrigues' formula
      https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

      Args
        r: 1x3 exponential map
      Returns
        R: 3x3 rotation matrix
    """
    if r.size == 3:     # for each data
        theta = np.linalg.norm(r)
        r0 = np.divide(r, theta + np.finfo(np.float32).eps)
        r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
        r0x = r0x - r0x.T
        R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x)
    else:               # for many data
        theta = np.linalg.norm(r, axis=1, keepdims=True)
        r0 = np.divide(r, theta + np.finfo(np.float32).eps)

        r0x = np.zeros((r.shape[0], 9))
        r0x[:, 1] = -r0[:, 2]
        r0x[:, 2] = r0[:, 1]
        r0x[:, 3] = r0[:, 2]
        r0x[:, 5] = -r0[:, 0]
        r0x[:, 6] = -r0[:, 1]
        r0x[:, 7] = r0[:, 0]

        mm_r0x_r0x = np.zeros((r.shape[0], 9))
        mm_r0x_r0x[:, 0] = - r0[:, 2] ** 2 - r0[:, 1] ** 2
        mm_r0x_r0x[:, 1] = r0[:, 0] * r0[:, 1]
        mm_r0x_r0x[:, 2] = r0[:, 0] * r0[:, 2]
        mm_r0x_r0x[:, 3] = r0[:, 0] * r0[:, 1]
        mm_r0x_r0x[:, 4] = - r0[:, 2] ** 2 - r0[:, 0] ** 2
        mm_r0x_r0x[:, 5] = r0[:, 1] * r0[:, 2]
        mm_r0x_r0x[:, 6] = r0[:, 0] * r0[:, 2]
        mm_r0x_r0x[:, 7] = r0[:, 1] * r0[:, 2]
        mm_r0x_r0x[:, 8] = - r0[:, 1] ** 2 - r0[:, 0] ** 2

        eye = np.zeros((r.shape[0], 9))
        eye[:, 0] = 1
        eye[:, 4] = 1
        eye[:, 8] = 1

        expanded_theta = np.repeat(theta, 9, axis=1)

        R = eye + np.sin(expanded_theta) * r0x + (1 - np.cos(expanded_theta)) * mm_r0x_r0x

    return R.reshape(r.shape[0], 3, 3)


def expmap2rotmat_tensor(r):
    """
    Args
        r: (batch_size x seq x 3) exponential map
    Returns
        R: 3x3 rotation matrix -> 9
    """
    theta = torch.norm(r, 2, 2)     # (batch_size x seq)
    eps = 1e-15
    theta = (theta + eps).unsqueeze(2)
    # theta = theta.expand(theta.shape[0], theta.shape[1], 3)     # (batch_size x seq x 3)
    r0 = torch.div(r, theta)

    # column major or row major check!!!!
    r0x = torch.zeros(theta.shape[0], theta.shape[1], 9).to(device)
    # r0x[:, :, 1] = -r0[:, :, 2]
    # r0x[:, :, 2] = r0[:, :, 1]
    # r0x[:, :, 3] = r0[:, :, 2]
    # r0x[:, :, 5] = -r0[:, :, 0]
    # r0x[:, :, 6] = -r0[:, :, 1]
    # r0x[:, :, 7] = r0[:, :, 0]
    r0x[:, :, 1] = r0[:, :, 2]
    r0x[:, :, 2] = -r0[:, :, 1]
    r0x[:, :, 3] = -r0[:, :, 2]
    r0x[:, :, 5] = r0[:, :, 0]
    r0x[:, :, 6] = r0[:, :, 1]
    r0x[:, :, 7] = -r0[:, :, 0]

    mm_r0x_r0x = torch.zeros(theta.shape[0], theta.shape[1], 9).to(device)
    mm_r0x_r0x[:, :, 0] = - r0[:, :, 2]**2 - r0[:, :, 1]**2
    mm_r0x_r0x[:, :, 1] = r0[:, :, 0] * r0[:, :, 1]
    mm_r0x_r0x[:, :, 2] = r0[:, :, 0] * r0[:, :, 2]
    mm_r0x_r0x[:, :, 3] = r0[:, :, 0] * r0[:, :, 1]
    mm_r0x_r0x[:, :, 4] = - r0[:, :, 2]**2 - r0[:, :, 0]**2
    mm_r0x_r0x[:, :, 5] = r0[:, :, 1] * r0[:, :, 2]
    mm_r0x_r0x[:, :, 6] = r0[:, :, 0] * r0[:, :, 2]
    mm_r0x_r0x[:, :, 7] = r0[:, :, 1] * r0[:, :, 2]
    mm_r0x_r0x[:, :, 8] = - r0[:, :, 1]**2 - r0[:, :, 0]**2

    eye = torch.zeros(theta.shape[0], theta.shape[1], 9).to(device)
    eye[:, :, 0] = 1
    eye[:, :, 4] = 1
    eye[:, :, 8] = 1

    expanded_theta = theta.expand(theta.shape[0], theta.shape[1], 9)
    R = eye + torch.sin(expanded_theta) * r0x + (1 - torch.cos(expanded_theta)) * mm_r0x_r0x
    return R


##################################################################################
# data utils
##################################################################################
def prepare_next_batch(seq_len, batch_size, dataset, representation, sequences):
    global subject, action

    probs = []
    for i, (subject, action) in enumerate(sequences):
        probs.append(dataset[subject][action][representation].shape[0])
    probs = np.array(probs) / np.sum(probs)
    nj = dataset[subject][action][representation].shape[1]
    dim = nj * dataset[subject][action][representation].shape[2]

    # The memory layout of the batches is: rotations or positions | translations
    buffer_data = np.zeros((batch_size, seq_len, dim), dtype='float32')
    buffer_data_flip = np.zeros((batch_size, seq_len, dim), dtype='float32')
    buffer_root_trajectory = np.zeros((batch_size, seq_len, 3), dtype='float32')
    buffer_root_trajectory_flip = np.zeros((batch_size, seq_len, 3), dtype='float32')

    pseudo_passes = (len(sequences) + batch_size - 1) // batch_size  # Round in excess
    for p in range(pseudo_passes):
        idxs = np.random.choice(len(sequences), size=batch_size, replace=True, p=probs)
        for i, (subject, action) in enumerate(np.array(sequences)[idxs]):
            # Pick a random chunk
            full_seq_length = dataset[subject][action][representation].shape[0]

            max_index = full_seq_length - seq_len - 1
            start_idx = np.random.randint(1, max_index)    # 가끔 first idx는 모션이 무너진 경우가 있음
            end_idx = start_idx + seq_len

            buffer_data[i, :, :dim] = dataset[subject][action][representation][start_idx:end_idx]\
                .reshape(seq_len, -1)
            buffer_data_flip[i, :, :dim] = np.flip(buffer_data[i, :, :dim], 0)

            buffer_root_trajectory[i, :, :3] = dataset[subject][action]['trajectory'][start_idx:end_idx].reshape(
                seq_len, -1)
            buffer_root_trajectory_flip[i, :, :3] = np.flip(buffer_root_trajectory[i, :, :3], 0)

        yield buffer_data, buffer_data_flip, buffer_root_trajectory, buffer_root_trajectory_flip

def prepare_specific_action(seq_len, batch_size, dataset, representation, sequences, spe_action):
    # permute entries at random
    specific_sequences = []
    for i in range(len(sequences)):
        if spe_action+'_' in sequences[i][1] and not '_m_' in sequences[i][1]:
            specific_sequences.append(sequences[i])

    nj = dataset[specific_sequences[0][0]][specific_sequences[0][1]][representation].shape[1]
    dim = nj * dataset[specific_sequences[0][0]][specific_sequences[0][1]][representation].shape[2]

    # The memory layout of the batches is: rotations or positions | translations
    buffer_data = np.zeros((batch_size, seq_len, dim), dtype='float32')
    buffer_data_flip = np.zeros((batch_size, seq_len, dim), dtype='float32')
    buffer_root_trajectory = np.zeros((batch_size, seq_len, 3), dtype='float32')
    buffer_root_trajectory_flip = np.zeros((batch_size, seq_len, 3), dtype='float32')

    pseudo_passes = (len(specific_sequences) + batch_size - 1) // batch_size  # Round in excess
    for p in range(pseudo_passes):
        idxs = np.random.choice(len(specific_sequences), size=batch_size, replace=True)
        for i, (subject, action) in enumerate(np.array(specific_sequences)[idxs]):
            # Pick a random chunk
            full_seq_length = dataset[subject][action][representation].shape[0]

            max_index = full_seq_length - seq_len - 1
            start_idx = np.random.randint(1, max_index)
            end_idx = start_idx + seq_len

            buffer_data[i, :, :dim] = dataset[subject][action][representation][start_idx:end_idx] \
                .reshape(seq_len, -1)
            buffer_data_flip[i, :, :dim] = np.flip(buffer_data[i, :, :dim], 0)

            buffer_root_trajectory[i, :, :3] = dataset[subject][action]['trajectory'][start_idx:end_idx].reshape(
                seq_len, -1)
            # buffer_root_trajectory[i, :, [0, 2]] = 0  # Absolute translations across the XY plane are removed here
            buffer_root_trajectory_flip[i, :, :3] = np.flip(buffer_root_trajectory[i, :, :3], 0)

        yield buffer_data, buffer_data_flip, buffer_root_trajectory, buffer_root_trajectory_flip

def unNormalizeData_tensor_batch(normalizedData, data_mean, data_std, dim_to_use=None):
    """
        Args
            normalizedData: (batch, seq, dim) matrix with normalized data,  dtype: tensor
            data_mean: vector of mean used to normalize the data,  dtype: numpy
            data_std: vector of standard deviation used to normalize the data,  dtype: numpy
            dimensions_to_use: vector with dimensions not used by the model dtype : list

        Returns
            origData: data originally used to
    """

    n_b = normalizedData.shape[0]
    T = normalizedData.shape[1]
    D = data_mean.shape[0]

    if dim_to_use is None:
        origData = normalizedData
    else:
        origData = torch.zeros(n_b, T, D).to(device)
        origData[:, :, dim_to_use] = normalizedData

    # potentially ineficient, but only done at once
    stdMat = data_std.repeat(n_b, T, 1)
    meanMat = data_mean.repeat(n_b, T, 1)
    origData = origData * stdMat + meanMat

    return origData

def slerp(p0, p1, t):
    """Spherical interpolation."""
    p0 = p0.data.cpu().numpy()
    p1 = p1.data.cpu().numpy()
    omega = np.arccos(np.dot(p0 / np.linalg.norm(p0), p1 / np.linalg.norm(p1)))
    so = np.sin(omega)
    return np.sin((1.0 - t) * omega) / so * p0 + np.sin(t * omega) / so * p1

def lerp(p0, p1, t):
    """Linear interpolation."""
    p0 = p0.data.cpu().numpy()
    p1 = p1.data.cpu().numpy()
    return (1.0 - t) * p0 + t * p1


##################################################################################
# learning utils
##################################################################################
def get_scheduler(optimizer, hyperparameters, iterations=-1):
    if 'lr_policy' not in hyperparameters or hyperparameters['lr_policy'] == 'constant':
        scheduler = None # constant scheduler
    elif hyperparameters['lr_policy'] == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=hyperparameters['step_size'],
                                        gamma=hyperparameters['gamma'], last_epoch=iterations)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', hyperparameters['lr_policy'])
    return scheduler

def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find('Linear') == 0) and hasattr(m, 'weight'):
            # print m.__class__.__name__
            if init_type == 'gaussian':
                init.normal_(m.weight.data, 0.0, 0.02)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
    return init_fun


##################################################################################
# evaluation
##################################################################################
def compute_bone_vector(skeleton, pos):
    """
        Compute bone vector
        Arguments (where N = batch size, L = sequence length, J = number of joints):
            -- pos: (N, L, J, 3) tensor describing the joint world positions.
    """
    skeleton_parents = skeleton.parents()
    position_world = pos
    bone_vector = []

    for i in range(position_world.shape[2]):
        if skeleton_parents[i] == -1:
            continue
        bone_vector.append(pos[:, :, i] - pos[:, :, skeleton_parents[i]])

    return torch.stack(bone_vector, dim=3).permute(0, 1, 3, 2)

def compute_local_angle(bone_vector):
    eps = 1e-15
    local_angle = []
    for i in range(bone_vector.shape[2]):
        if (i+1) == bone_vector.shape[2]:
            break
        dot_batch = torch.bmm(bone_vector[:, :, i+1].view(-1, 1, 3),
                              bone_vector[:, :, i].view(-1, 3, 1))
        dot_batch = dot_batch.view(-1)
        norm_v1 = torch.norm(bone_vector[:, :, i].view(-1, 3), 2, 1)
        norm_v2 = torch.norm(bone_vector[:, :, i+1].view(-1, 3), 2, 1)

        local_angle.append(torch.acos(dot_batch / (norm_v1*norm_v2 + eps)))

    return torch.stack(local_angle, dim=1).view(bone_vector.shape[0], bone_vector.shape[1], -1)

def compute_global_angle(bone_vector1, bone_vector2):
    eps = 1e-15
    global_angle = []
    for i in range(bone_vector1.shape[2]):
        dot_batch = torch.bmm(bone_vector1[:, :, i].view(-1, 1, 3),
                              bone_vector2[:, :, i].view(-1, 3, 1))
        dot_batch = dot_batch.view(-1)
        norm_v1 = torch.norm(bone_vector1[:, :, i].view(-1, 3), 2, 1)
        norm_v2 = torch.norm(bone_vector2[:, :, i].view(-1, 3), 2, 1)

        global_angle.append(torch.acos(dot_batch / (norm_v1*norm_v2 + eps)))

    return torch.stack(global_angle, dim=1).view(bone_vector1.shape[0], bone_vector1.shape[1], -1)

##################################################################################
# etc
##################################################################################
def get_config(config):
    with open(config, 'r') as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)

def initialize_path(args, config, save=True):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    config['tb_dir'] = os.path.join(config['main_dir'], "log")
    config['info_dir'] = os.path.join(config['main_dir'], "info")
    config['output_dir'] = os.path.join(config['main_dir'], "output")
    ensure_dirs([config['main_dir'], config['model_dir'], config['tb_dir'],
                 config['info_dir'], config['output_dir']])
    if save:
        shutil.copy(args.config, os.path.join(config['info_dir'], 'config.yaml'))

def ensure_dir(path):
    """
    create path by first checking its existence,
    :param paths: path
    :return:
    """
    if not os.path.exists(path):
        print("Create folder ", path)
        os.makedirs(path)
    else:
        print(path, " already exists.")

def ensure_dirs(paths):
    """
    create paths by first checking their existence
    :param paths: list of path
    :return:
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            ensure_dir(path)
    else:
        ensure_dir(paths)

# Get model list for resume
def get_model_list(dirname, key):
    if os.path.exists(dirname) is False:
        return None
    gen_models = [os.path.join(dirname, f) for f in os.listdir(dirname) if
                  os.path.isfile(os.path.join(dirname, f)) and key in f and ".pt" in f]
    if gen_models is None:
        return None
    gen_models.sort()
    last_model_name = gen_models[-1]
    return last_model_name

def save_motions(motion_data, f_dir):
    with h5py.File(f_dir, 'a') as hf:
        for i in np.arange(len(motion_data)):
            node_name = "batch{0}".format(i+1)
            hf.create_dataset(node_name, data=motion_data[i])
    print('save generated motions')

def prepare_sub_folder(output_directory):
    motion_directory = os.path.join(output_directory, 'motions')
    if not os.path.exists(motion_directory):
        print("Creating directory: {}".format(motion_directory))
        os.makedirs(motion_directory)
    checkpoint_directory = os.path.join(output_directory, 'checkpoints')
    if not os.path.exists(checkpoint_directory):
        print("Creating directory: {}".format(checkpoint_directory))
        os.makedirs(checkpoint_directory)
    return checkpoint_directory, motion_directory

def cycle(iterable):
    while True:
        for x in iterable:
            yield x

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)