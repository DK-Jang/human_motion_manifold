import torch
import argparse
import os
import sys
from skeleton_h36m import skeleton_H36M
from trainer import Trainer
from h36m_dataset import H36MDataset
from utils import get_config, save_motions, set_seed, ensure_dirs, ensure_dir

def initialize_path(args, config, save=True):
    config['main_dir'] = os.path.join('.', config['name'])
    config['model_dir'] = os.path.join(config['main_dir'], "pth")
    config['output_dir'] = os.path.join(config['main_dir'], "output")
    ensure_dirs([config['main_dir'], config['model_dir'], config['output_dir']])

parser = argparse.ArgumentParser()
parser.add_argument('--config', 
                    type=str, 
                    default='pretrianed/info/config.yaml',
                    help='Path to the config file.')
args = parser.parse_args()

config = get_config(args.config)
initialize_path(args, config)

print("Random Seed: ", 1777)
set_seed(1777)

# Setup model
trainer = Trainer(skeleton_H36M, config)
trainer.load_checkpoint()

# Create dataloader
dataset_test = H36MDataset('train', config)

# Setup dataset
exp_mean = dataset_test.exp_mean
exp_std = dataset_test.exp_std
exp_dimTouse = dataset_test.exp_dimTOuse

m_recon_np, m_recon_vel_np = trainer.random_sample(60, exp_mean, exp_std, exp_dimTouse)

random_sample_dir = os.path.join(config['output_dir'], 'random_sample')
ensure_dir(random_sample_dir)
try:
    os.remove(os.path.join(random_sample_dir, 'm_recon.hdf5'))
    os.remove(os.path.join(random_sample_dir, 'm_recon_vel.hdf5'))
except OSError:
    pass

# represented as quaternion 
save_motions(m_recon_np, os.path.join(random_sample_dir, 'm_recon.hdf5'))
save_motions(m_recon_vel_np, os.path.join(random_sample_dir, 'm_recon_vel.hdf5'))
