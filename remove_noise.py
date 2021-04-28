import torch
import argparse
import os
import sys
import numpy as np
from torch.utils.data import DataLoader
from skeleton_h36m import skeleton_H36M
from trainer import MotionManifold
from h36m_dataset import H36MDataset
from utils import get_config, cycle, save_motions, save_motions_csv
import multiprocessing

torch.manual_seed(1238)
# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    parser.add_argument('--model_dir', type=str, help="saved model parameters dir")
    opts = parser.parse_args()
        
    opts.config = './no_concat_z_reg/config.yaml'
    # opts.output_motion_path = '/no_concat_z_reg/motions'
    opts.output_motion_path = './no_concat_z_reg/motions/noise'
    opts.model_dir = './no_concat_z_reg/checkpoints'
    

    # Load experiment setting
    config = get_config(opts.config)

    # Setup model
    trainer = MotionManifold(skeleton_H36M, config)
    print(trainer.gen)      # print networks' architecture
    trainer.load_weights(opts.model_dir)
    trainer.to(device)

    # Create dataloader
    dataset_test = H36MDataset('train', config)
    test_loader = DataLoader(dataset_test, batch_size=30, shuffle=True, num_workers=2)
    test_loader = cycle(test_loader)
    test_batch = next(test_loader)

    # Setup dataset
    exp_mean = dataset_test.exp_mean
    exp_std = dataset_test.exp_std
    exp_dimTouse = dataset_test.exp_dimTOuse

    m_recon_np, m_recon_vel_np, m_, m_noise, root_trj = trainer.remove_noise(test_batch, exp_mean, exp_std, exp_dimTouse)
    
    try:
        os.remove(opts.output_motion_path + '/m_recon.hdf5')
        os.remove(opts.output_motion_path + '/m_recon_vel.hdf5')
        os.remove(opts.output_motion_path + '/m_.hdf5')
        os.remove(opts.output_motion_path + '/m_noise.hdf5')
        os.remove(opts.output_motion_path + '/root_trj.hdf5')
    except OSError:
        pass
            
    save_motions(m_recon_np, opts.output_motion_path + '/m_recon.hdf5')
    save_motions(m_recon_vel_np, opts.output_motion_path + '/m_recon_vel.hdf5')
    save_motions(m_, opts.output_motion_path + '/m_.hdf5')
    save_motions(m_noise, opts.output_motion_path + '/m_noise.hdf5')
    save_motions(root_trj, opts.output_motion_path + '/root_trj.hdf5')

if __name__ == '__main__':
    main()

