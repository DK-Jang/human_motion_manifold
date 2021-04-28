import torch
import argparse
import os
import sys
from skeleton_h36m import skeleton_H36M
from trainer import Trainer
from h36m_dataset import H36MDataset
from torch.utils.data import DataLoader
from utils import get_config, save_motions, set_seed, ensure_dirs, ensure_dir, cycle

def initialize_path(args, config, save=True):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    config['output_dir'] = os.path.join(config['main_dir'], "output")
    ensure_dirs([config['main_dir'], config['model_dir'], config['output_dir']])

parser = argparse.ArgumentParser()
parser.add_argument('--config', 
                    type=str, 
                    default='pretrained/info/config.yaml',
                    help='Path to the config file.')
parser.add_argument('--output_representation', 
                    type=str, 
                    default='rotations',
                    help='rotations_exp or positions_world or rotations(quaternion)')
args = parser.parse_args()

config = get_config(args.config)
initialize_path(args, config)

print("Random Seed: ", 1777)
set_seed(1777)

# Setup model
trainer = Trainer(skeleton_H36M, config)
trainer.load_checkpoint()

# Create dataloader
dataset_test = H36MDataset('test', config)
test_loader = DataLoader(dataset_test, batch_size=60,
                         shuffle=True, num_workers=4)
test_loader = cycle(test_loader)

test_batch = next(test_loader)

# Setup dataset
exp_mean = dataset_test.exp_mean
exp_std = dataset_test.exp_std
exp_dimTouse = dataset_test.exp_dimTOuse

m_recon_np, m_recon_vel_np, m_gt_np = trainer.test_motion(test_batch, 
                                                          exp_mean, exp_std, exp_dimTouse,
                                                          out_representation=args.output_representation)

recon_dir = os.path.join(config['output_dir'], 'recon')
ensure_dir(recon_dir)
try:
    os.remove(os.path.join(recon_dir, 'm_recon.hdf5'))
    os.remove(os.path.join(recon_dir, 'm_recon_vel.hdf5'))
    os.remove(os.path.join(recon_dir, 'm_gt.hdf5'))
except OSError:
    pass

save_motions(m_recon_np, recon_dir + '/m_recon.hdf5')
save_motions(m_recon_vel_np, recon_dir + '/m_recon_vel.hdf5')
save_motions(m_gt_np, recon_dir + '/m_gt.hdf5')
