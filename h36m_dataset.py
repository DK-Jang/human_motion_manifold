import os
from torch.utils.data import Dataset
import torch
import glob
import numpy as np

MOTION_EXTENSIONS = [
    '.bvh', '.npy', '.npz'
]

class H36MDataset(Dataset):
    def __init__(self, phase, config, specific_motion=None):
        super(H36MDataset, self).__init__()

        assert phase in ['train', 'valid', 'test']
        data_exp_path = os.path.join(config['data_dir'], 'exp', phase)
        data_pos_path = os.path.join(config['data_dir'], 'pos', phase)
        data_tra_path = os.path.join(config['data_dir'], 'trajectory',phase)

        motions_exp, motions_exp_name = make_dataset(data_exp_path, specific_motion)
        motions_pos, _ = make_dataset(data_pos_path, specific_motion)
        motions_tra, _ = make_dataset(data_tra_path, specific_motion) 

        if len(motions_exp) == 0:
            raise(RuntimeError("Found 0 motions_exp in: " + data_exp_path + "\n"
                               "Supported motion extensions are: " +
                               ",".join(MOTION_EXTENSIONS)))
        if len(motions_pos) == 0:
            raise(RuntimeError("Found 0 motions_pos in: " + data_pos_path + "\n"
                               "Supported motion extensions are: " +
                               ",".join(MOTION_EXTENSIONS)))
        if len(motions_tra) == 0:
            raise(RuntimeError("Found 0 motions_trajectory in: " + data_tra_path + "\n"
                               "Supported motion extensions are: " +
                               ",".join(MOTION_EXTENSIONS)))

        # self.data_root = data_root
        self.seq_len = config['seq_len']
        self.motions_exp = motions_exp
        self.motions_pos = motions_pos
        self.motions_tra = motions_tra
        self.motions_name = motions_exp_name

        self.exp_mean, self.exp_std, self.pos_mean, self.pos_std = get_meanstd(config)
        if config['dimTouse']:
            self.exp_dimTOuse, self.pos_dimTouse = get_dimTouse(config)

    def __getitem__(self, index):
        seq_len = self.seq_len
        path_exp = self.motions_exp[index]
        path_pos = self.motions_pos[index]
        path_tra = self.motions_tra[index]
        motion_exp = np.load(path_exp).astype(np.float32)
        motion_pos = np.load(path_pos).astype(np.float32)
        motion_tra = np.load(path_tra).astype(np.float32)

        max_index = motion_exp.shape[0] - seq_len - 1
        start_idx = np.random.randint(1, max_index)
        end_idx = start_idx + seq_len
        motion_exp = motion_exp[start_idx:end_idx].reshape((seq_len, -1))
        motion_exp_flip = np.flip(motion_exp, 0).copy()
        motion_pos = motion_pos[start_idx:end_idx].reshape((seq_len, -1))
        motion_pos_flip = np.flip(motion_pos, 0).copy()
        motion_tra = motion_tra[start_idx:end_idx].reshape((seq_len, -1))
        motion_tra_flip = np.flip(motion_tra, 0).copy()
        
        motion_name = path_exp.split(os.sep)[-1]
        motion_name = motion_name.split('.')[0]

        # return motion
        return {"motion_exp": motion_exp, "motion_exp_flip": motion_exp_flip,
                "motion_pos": motion_pos, "motion_pos_flip": motion_pos_flip,
                "motion_tra": motion_tra, "motion_tra_flip": motion_tra_flip,
                "motion_name": motion_name}

    def __len__(self):
        return len(self.motions_exp)


def get_meanstd(config):
    pos_mean_path = config['pos_mean_path']
    pos_std_path = config['pos_std_path']
    exp_mean_path = config['exp_mean_path']
    exp_std_path = config['exp_std_path']
    
    assert os.path.exists(pos_mean_path) and os.path.exists(pos_std_path) \
        and os.path.exists(exp_mean_path) and os.path.exists(exp_std_path)
    
    pos_mean = np.load(pos_mean_path)
    pos_std = np.load(pos_std_path)
    exp_mean = np.load(exp_mean_path)
    exp_std = np.load(exp_std_path)

    return exp_mean, exp_std, pos_mean, pos_std

def get_dimTouse(config):
    exp_dimTouse_path = config['exp_dimTouse_path']
    pos_dimTOuse_path = config['pos_dimTouse_path']

    assert os.path.exists(exp_dimTouse_path) and os.path.exists(pos_dimTOuse_path)
    
    exp_dimTouse = np.load(exp_dimTouse_path)
    pos_dimTOuse = np.load(pos_dimTOuse_path)

    return exp_dimTouse, pos_dimTOuse

def is_motion_file(filename):
    return any(filename.endswith(extension) for extension in MOTION_EXTENSIONS)

def make_dataset(dir, specific_motion=None):
    motions = []
    motions_name = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_motion_file(fname):
                path = os.path.join(root, fname)
                mname = fname.split('.')[0]

                if not specific_motion is None:    
                    if specific_motion + '_' in mname and not '_m_' in mname:
                        motions.append(path)
                        motions_name.append(mname)
                    else:
                        continue
                else:
                    motions.append(path)
                    motions_name.append(mname)

    motions = sorted(motions)
    motions_name = sorted(motions_name)

    return motions, motions_name


# def make_dataset(dir):
#     motions = []
#     assert os.path.isdir(dir), '%s is not a valid directory' % dir

#     for root, _, fnames in sorted(os.walk(dir)):
#         for fname in fnames:
#             if is_motion_file(fname):
#                 path = os.path.join(root, fname)
#                 motions.append(path)

#     return motions

# def collate_fn(batch, specific_motion='waiting'):
#     # batch = list(filter(lambda x:x is not None, batch))

#     for data in batch:
#         print(type(data['motion_exp']))
    
#     motion_exp = np.array([data['motion_exp'] for data in batch])
#     motion_exp_flip = np.array([data['motion_exp_flip'] for data in batch])
#     motion_pos = np.array([data['motion_pos'] for data in batch])
#     motion_pos_flip = np.array([data['motion_pos_flip'] for data in batch])
#     motion_tra = np.array([data['motion_tra'] for data in batch])
#     motion_tra_flip = np.array([data['motion_tra_flip'] for data in batch])
#     motion_name = np.array([data['motion_name'] for data in batch])

#     # print(len(motion_exp))

#     specific_idx = [i for i, name in enumerate(motion_name) \
#                     if specific_motion + '_' in name and not '_m_' in name]
    
#     return {"motion_exp": motion_exp[specific_idx], "motion_exp_flip": motion_exp_flip[specific_idx],
#             "motion_pos": motion_pos[specific_idx], "motion_pos_flip": motion_pos_flip[specific_idx],
#             "motion_tra": motion_tra[specific_idx], "motion_tra_flip": motion_tra_flip[specific_idx],
#             "motion_name": motion_name[specific_idx]}