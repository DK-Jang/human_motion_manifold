import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import os
import h5py
import numpy as np
from model import MotionGen, MotionDis, Gaussian_P_Z
from utils import unNormalizeData_tensor_batch, get_model_list, \
    get_scheduler, lerp, slerp, prepare_next_batch, expmap_to_quaternion
import logging
logger = logging.getLogger(__name__)


class Trainer(nn.Module):
    def __init__(self, skeleton, config):
        super(Trainer, self).__init__()
        self.skeleton = skeleton
        self.lr = config['lr']
        self.lr_decay = config['lr_decay']
        self.gradient_clip = config['gradient_clip']
        self.seq_len = config['seq_len']
        self.n_joints = config['n_joints']
        self.z_dim = config['z_dim']
        self.model_dir = config['model_dir']

        # Initiate the networks
        self.gen = MotionGen(config['input_size'], self.seq_len, config['z_dim'], config['gen'])
        self.dis = MotionDis(config['input_size'], config['dis'])

        # Setup the optimizers
        gen_params = list(self.gen.parameters())
        dis_params = list(self.dis.parameters())

        self.gen_opt = torch.optim.Adam([p for p in gen_params if p.requires_grad], lr=self.lr)
        self.dis_opt = torch.optim.Adam([p for p in dis_params if p.requires_grad], lr=self.lr)

        self.gen_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.gen_opt, 0.999)
        self.dis_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.dis_opt, 0.999)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            self.gen.to(self.device)
            self.dis.to(self.device)

    def forward(self, m, m_flip):
        self.eval()
        z = self.gen.encode(m)
        m_recon_flip, m_recon_vel_flip = self.gen.decode(z)
        self.train()
        return m_recon_flip, m_recon_vel_flip

    def recon_criterion(self, input_seq, target_seq):
        # 3 for exp. 4 for quaternion
        return torch.mean((input_seq.view(input_seq.shape[0], input_seq.shape[1], -1, 3)
                           - target_seq.view(target_seq.shape[0], target_seq.shape[1], -1, 3)).norm(dim=3))

    def fk_criterion(self, input_exp, target_pos, root_trajectory,
                           exp_mean, exp_std, exp_dimTouse,
                           pos_mean, pos_std, pos_dimTouse):
        """ Compute forward kinematics for fk loss
            
            Args:
                (where N = batch size, L = sequence length, J = number of joints):
                input_exp: (N, L, J, 3) tensor describing the joint exp coordinates.
                target_pos: (N, L, J, 3) tensor describing the joint pos coordinates.
            Returns:
                loss_fk
        """

        input_un = unNormalizeData_tensor_batch(input_exp, exp_mean, exp_std, exp_dimTouse)

        input_un = input_un.view(input_un.shape[0], input_un.shape[1], -1, 3)
        input_pos = self.skeleton.forward_kinematics_exp(input_un, root_trajectory)
        input_pos = input_pos.contiguous().view(input_pos.shape[0], input_pos.shape[1], -1)
        nm_input_pos = torch.div((input_pos - pos_mean), pos_std)
        nm_input_pos = nm_input_pos[:, :, pos_dimTouse]

        # Euclidean distance (with joint-wise square root, not MSE!)
        nm_input_pos = nm_input_pos.view(nm_input_pos.shape[0], nm_input_pos.shape[1], -1, 3)
        nm_target_pos = target_pos.view(target_pos.shape[0], target_pos.shape[1], -1, 3)

        loss_fk = torch.mean((nm_input_pos - nm_target_pos).norm(dim=3))

        return loss_fk

    def __compute_MMD(self, z):
        p_z = Gaussian_P_Z(z.size(1))
        sample_Pz = p_z.sample(z.size(0))
        # sample_Pz = sample_Pz.to(z.data.get_device())
        sample_Pz = sample_Pz.to(self.device)

        n_ = z.size(0)
        C_base = 2. * z.size(1) * 1
        z_dot_z = torch.mm(sample_Pz, sample_Pz.transpose(0, 1))
        z_tilde_dot_z_tilde = torch.mm(z, z.transpose(0, 1))
        z_dot_z_tilde = torch.mm(sample_Pz, z.transpose(0, 1))

        dist_z_z = (torch.unsqueeze(torch.diagonal(z_dot_z, 0), 1) \
                    + torch.unsqueeze(torch.diagonal(z_dot_z, 0), 0)) \
                    - 2 * z_dot_z

        dist_z_tilde_z_tilde = (torch.unsqueeze(torch.diagonal(z_tilde_dot_z_tilde, 0), 1)
                                + torch.unsqueeze(torch.diagonal(z_tilde_dot_z_tilde, 0), 0)) \
                                - 2 * z_tilde_dot_z_tilde

        dist_z_z_tilde = (torch.unsqueeze(torch.diagonal(z_dot_z, 0), 1)
                          + torch.unsqueeze(torch.diagonal(z_tilde_dot_z_tilde, 0), 0)) \
                          - 2 * z_dot_z_tilde

        loss_z = 0
        # for scale in [1.0]:
        for scale in [.1, .2, .5, 1., 2., 5., 10.]:
            C = C_base * scale

            k_z = \
                C / (C + dist_z_z + 1e-8)
            k_z_tilde = \
                C / (C + dist_z_tilde_z_tilde + 1e-8)
            k_z_z_tilde = \
                C / (C + dist_z_z_tilde + 1e-8)

            loss = 1 / (n_ * (n_ - 1)) * torch.sum(k_z) \
                   + 1 / (n_ * (n_ - 1)) * torch.sum(k_z_tilde) \
                   - 2 / (n_ * n_) * torch.sum(k_z_z_tilde)

            loss_z += loss
        loss_z = loss_z.mean()

        return loss_z

    def gen_update(self, data, exp_mean, exp_std, exp_dimTouse, \
                   pos_mean, pos_std, pos_dimTouse, config):
        self.gen.train()
        eps = 1e-15

        # motion data
        m_exp = data['motion_exp'].to(self.device).detach()     # input: (batch, seq, dim)
        m_exp_flip = data['motion_exp_flip'].to(self.device).detach()
        m_pos_flip = data['motion_pos_flip'].to(self.device).detach()
        m_tra_flip = data['motion_tra_flip'].to(self.device).detach()

        # mean, std
        exp_mean = torch.from_numpy(exp_mean).to(self.device).detach()
        exp_std = torch.from_numpy(exp_std).to(self.device).detach()
        pos_mean = torch.from_numpy(pos_mean).to(self.device).detach()
        pos_std = torch.from_numpy(pos_std).to(self.device).detach()

        # encode
        z = self.gen.encode(m_exp)

        # decode
        m_exp_recon_flip, m_exp_recon_vel_flip = self.gen.decode(z=z)
    
        # encode again
        z_recon = self.gen.encode(torch.flip(m_exp_recon_flip, [1]))
        z_recon_vel = self.gen.encode(torch.flip(m_exp_recon_vel_flip, [1]))

        # update loss metric
        losses_gen = {}

        # reconstruction loss
        losses_gen['loss_gen_recon'] = config['recon_x_w'] * self.recon_criterion(m_exp_recon_flip, m_exp_flip)
        losses_gen['loss_gen_recon_vel'] = config['recon_x_w'] * self.recon_criterion(m_exp_recon_vel_flip, m_exp_flip)

        # regularizer loss
        losses_gen['loss_gen_z'] = config['recon_z_w'] * self.__compute_MMD(z)

        # GAN loss
        if config['gan_w'] > 0:
            losses_gen['loss_gen_adv'] = config['gan_w'] * self.dis.calc_gen_loss(m_exp_recon_flip)
            losses_gen['loss_gen_adv_vel'] = config['gan_w'] * self.dis.calc_gen_loss(m_exp_recon_vel_flip)

        # latent regression loss
        if config['recon_z_reg_w'] > 0:
            losses_gen['loss_gen_z_recon'] = config['recon_z_reg_w'] * torch.mean(torch.abs(z_recon - z))
            losses_gen['loss_gen_z_recon_vel'] = config['recon_z_reg_w'] * torch.mean(torch.abs(z_recon_vel - z))

        # FK loss
        if config['recon_fk_w'] > 0:
            losses_gen['loss_gen_fk'] = config['recon_fk_w'] * \
                                        self.fk_criterion(m_exp_recon_flip, m_pos_flip, m_tra_flip,
                                                          exp_mean, exp_std, exp_dimTouse, 
                                                          pos_mean, pos_std, pos_dimTouse)
            losses_gen['loss_gen_fk_vel'] = config['recon_fk_w'] * \
                                            self.fk_criterion(m_exp_recon_vel_flip, m_pos_flip, m_tra_flip,
                                                              exp_mean, exp_std, exp_dimTouse, 
                                                              pos_mean, pos_std, pos_dimTouse)

        loss_gen_total = sum(losses_gen.values())
        self.gen_opt.zero_grad()
        loss_gen_total.backward()
        self.gen_opt.step()

        return losses_gen

    def dis_update(self, data, config):
        self.dis.train()

        # data
        motions_exp = data['motion_exp'].to(self.device).detach()
        motions_exp_flip = data['motion_exp_flip'].to(self.device).detach()

        # encode
        z = self.gen.encode(motions_exp)
        
        # decode
        m_recon_flip, m_recon_vel_flip = self.gen.decode(z)

        # update loss metric
        losses_dis = {}

        # D loss
        losses_dis['loss_dis_rot'] = config['gan_w'] * self.dis.calc_dis_loss(m_recon_flip.detach(), motions_exp_flip)
        losses_dis['loss_dis_vel'] = config['gan_w'] * self.dis.calc_dis_loss(m_recon_vel_flip.detach(), motions_exp_flip)

        loss_dis_total = sum(losses_dis.values())
        self.dis_opt.zero_grad()
        loss_dis_total.backward()
        self.dis_opt.step()

        return losses_dis

    def test_loss (self, data, exp_mean, exp_std, exp_dimTouse, \
                   pos_mean, pos_std, pos_dimTouse):
        self.gen.eval()
        with torch.no_grad():
            # motion data
            m_exp = data['motion_exp'].to(self.device).detach()     # input: (batch, seq, dim)
            m_exp_flip = data['motion_exp_flip'].to(self.device).detach()
            m_pos_flip = data['motion_pos_flip'].to(self.device).detach()
            m_tra_flip = data['motion_tra_flip'].to(self.device).detach()

            # mean, std
            exp_mean = torch.from_numpy(exp_mean).to(self.device).detach()
            exp_std = torch.from_numpy(exp_std).to(self.device).detach()
            pos_mean = torch.from_numpy(pos_mean).to(self.device).detach()
            pos_std = torch.from_numpy(pos_std).to(self.device).detach()

            # encode
            z = self.gen.encode(m_exp)

            # decode
            m_exp_recon_flip, m_exp_recon_vel_flip = self.gen.decode(z=z)

            # encode again
            z_recon = self.gen.encode(torch.flip(m_exp_recon_flip, [1]))
            z_recon_vel = self.gen.encode(torch.flip(m_exp_recon_vel_flip, [1]))

            # update loss metric
            losses_gen = {}

            # # reconstruction loss
            losses_gen['loss_gen_recon'] = self.recon_criterion(m_exp_recon_flip, m_exp_flip)
            losses_gen['loss_gen_recon_vel'] = self.recon_criterion(m_exp_recon_vel_flip, m_exp_flip)

            # # regularizer loss
            losses_gen['loss_gen_z'] = self.__compute_MMD(z)

            # latent regression loss
            losses_gen['loss_gen_z_recon'] = torch.mean(torch.abs(z_recon - z))
            losses_gen['loss_gen_z_recon_vel'] = torch.mean(torch.abs(z_recon_vel - z))

        return losses_gen

    def test_motion(self, data, exp_mean, exp_std, exp_dimTouse, out_representation='rotations'):
        self.gen.eval()
        with torch.no_grad():
            m_ = data['motion_exp'].to(self.device).detach()
            m_flip = data['motion_exp_flip'].to(self.device).detach()   # input: (batch, seq, dim)
            root_trajectory_flip = data['motion_tra_flip'].to(self.device).detach()      # input: (batch, seq, dim)

            # mean, std
            exp_mean = torch.from_numpy(exp_mean).to(self.device).detach()
            exp_std = torch.from_numpy(exp_std).to(self.device).detach()

            # encode
            z = self.gen.encode(m_)

            # decode
            m_recon_flip, m_recon_vel_flip = self.gen.decode(z=z)

            # unnormalize
            m_recon_flip_un = unNormalizeData_tensor_batch(m_recon_flip, exp_mean, exp_std, exp_dimTouse)
            m_recon_vel_flip_un = unNormalizeData_tensor_batch(m_recon_vel_flip, exp_mean, exp_std, exp_dimTouse)
            m_flip_un = unNormalizeData_tensor_batch(m_flip, exp_mean, exp_std, exp_dimTouse)

            if out_representation == 'positions_world':
                # forward kinematics
                m_recon_flip_un = m_recon_flip_un.view(m_recon_flip_un.shape[0], m_recon_flip_un.shape[1], -1, 3)
                m_recon_flip_xyz = self.skeleton.forward_kinematics_exp(m_recon_flip_un, root_trajectory_flip)
                m_recon_vel_flip_un = m_recon_vel_flip_un.view(m_recon_vel_flip_un.shape[0], m_recon_vel_flip_un.shape[1], -1, 3)
                m_recon_vel_flip_xyz = self.skeleton.forward_kinematics_exp(m_recon_vel_flip_un, root_trajectory_flip)
                m_flip_un = m_flip_un.view(m_flip_un.shape[0], m_flip_un.shape[1], -1, 3)
                m_flip_xyz = self.skeleton.forward_kinematics_exp(m_flip_un, root_trajectory_flip)

                # reverse
                m_recon_flip_np = m_recon_flip_xyz.data.cpu().numpy()
                m_recon_np = np.flip(m_recon_flip_np, 1)
                m_recon_vel_flip_np = m_recon_vel_flip_xyz.data.cpu().numpy()
                m_recon_vel_np = np.flip(m_recon_vel_flip_np, 1)
                m_flip_np = m_flip_xyz.data.cpu().numpy()
                m_np = np.flip(m_flip_np, 1)

            elif out_representation == 'rotations_exp':     
                # reverse
                m_recon_flip_np = m_recon_flip_un.data.cpu().numpy()
                m_recon_np = np.flip(m_recon_flip_np, 1)
                m_recon_vel_flip_np = m_recon_vel_flip_un.data.cpu().numpy()
                m_recon_vel_np = np.flip(m_recon_vel_flip_np, 1)
                m_flip_np = m_flip_un.data.cpu().numpy()
                m_np = np.flip(m_flip_np, 1)
            
            elif out_representation == 'rotations':  # quaternion
                # reverse
                m_recon_flip_np = m_recon_flip_un.data.cpu().numpy()
                m_recon_np = np.flip(m_recon_flip_np, 1)
                m_recon_np = m_recon_np.reshape(m_recon_np.shape[0], m_recon_np.shape[1], -1, 3)
                m_recon_np = expmap_to_quaternion(m_recon_np)
                
                m_recon_vel_flip_np = m_recon_vel_flip_un.data.cpu().numpy()
                m_recon_vel_np = np.flip(m_recon_vel_flip_np, 1)
                m_recon_vel_np = m_recon_vel_np.reshape(m_recon_vel_np.shape[0], m_recon_vel_np.shape[1], -1, 3)
                m_recon_vel_np = expmap_to_quaternion(m_recon_vel_np)

                m_flip_np = m_flip_un.data.cpu().numpy()
                m_np = np.flip(m_flip_np, 1)
                m_np = m_np.reshape(m_np.shape[0], m_np.shape[1], -1, 3)
                m_np = expmap_to_quaternion(m_np)
            
            return m_recon_np, m_recon_vel_np, m_np
    
    def random_sample(self, n_samples, exp_mean, exp_std, exp_dimTouse, out_representation='rotations'):
        self.gen.eval()
        with torch.no_grad():
            # mean, std
            exp_mean = torch.from_numpy(exp_mean).to(self.device).detach()
            exp_std = torch.from_numpy(exp_std).to(self.device).detach()

            # for sampling
            p_z = Gaussian_P_Z(self.z_dim)
            sample_Pz = p_z.sample(n_samples)
            sample_Pz = sample_Pz.to(self.device)

            # decode
            m_recon_flip, m_recon_vel_flip = self.gen.decode(z=sample_Pz)

            m_recon_flip_un = unNormalizeData_tensor_batch(m_recon_flip, exp_mean, exp_std, exp_dimTouse)
            m_recon_vel_flip_un = unNormalizeData_tensor_batch(m_recon_vel_flip, exp_mean, exp_std, exp_dimTouse)

            # reverse
            m_recon_flip_np = m_recon_flip_un.data.cpu().numpy()
            m_recon_np = np.flip(m_recon_flip_np, 1)
            m_recon_np = m_recon_np.reshape(m_recon_np.shape[0], m_recon_np.shape[1], -1, 3)
            m_recon_np = expmap_to_quaternion(m_recon_np)
                
            m_recon_vel_flip_np = m_recon_vel_flip_un.data.cpu().numpy()
            m_recon_vel_np = np.flip(m_recon_vel_flip_np, 1)
            m_recon_vel_np = m_recon_vel_np.reshape(m_recon_vel_np.shape[0], m_recon_vel_np.shape[1], -1, 3)
            m_recon_vel_np = expmap_to_quaternion(m_recon_vel_np)

            

        return m_recon_np, m_recon_vel_np
    
    def remove_noise (self, data, exp_mean, exp_std, exp_dimTouse):
        self.gen.eval()
        with torch.no_grad():
            m_ = data['motion_exp'].to(self.device).detach()
            noise = torch.randint(low=0, high=2, size=m_.shape)
            m_noise = m_ * noise

            m_flip = data['motion_exp_flip'].to(self.device).detach()   # input: (batch, seq, dim)
            root_trajectory_flip = data['motion_tra_flip'].to(self.device).detach()      # input: (batch, seq, dim)

            # mean, std
            exp_mean = torch.from_numpy(exp_mean).to(self.device).detach()
            exp_std = torch.from_numpy(exp_std).to(self.device).detach()

            # encode
            z = self.gen.encode(m_noise)

            # decode
            m_recon_flip, m_recon_vel_flip = self.gen.decode(z=z)

            # unnormalize
            m_recon_flip_un = unNormalizeData_tensor_batch(m_recon_flip, exp_mean, exp_std, exp_dimTouse)
            m_recon_vel_flip_un = unNormalizeData_tensor_batch(m_recon_vel_flip, exp_mean, exp_std, exp_dimTouse)
            m_flip_un = unNormalizeData_tensor_batch(m_flip, exp_mean, exp_std, exp_dimTouse)
            m_noise_un = unNormalizeData_tensor_batch(m_noise, exp_mean, exp_std, exp_dimTouse)

            # reverse
            m_recon_flip_np = m_recon_flip_un.data.cpu().numpy()
            m_recon_np = np.flip(m_recon_flip_np, 1)
            m_recon_np = m_recon_np.reshape(m_recon_np.shape[0], m_recon_np.shape[1], -1, 3)
            m_recon_np = expmap_to_quaternion(m_recon_np)
                
            m_recon_vel_flip_np = m_recon_vel_flip_un.data.cpu().numpy()
            m_recon_vel_np = np.flip(m_recon_vel_flip_np, 1)
            m_recon_vel_np = m_recon_vel_np.reshape(m_recon_vel_np.shape[0], m_recon_vel_np.shape[1], -1, 3)
            m_recon_vel_np = expmap_to_quaternion(m_recon_vel_np)

            m_flip_np = m_flip_un.data.cpu().numpy()
            m_np = np.flip(m_flip_np, 1)
            m_np = m_np.reshape(m_np.shape[0], m_np.shape[1], -1, 3)
            m_np = expmap_to_quaternion(m_np)

            m_noise_np = m_noise_un.data.cpu().numpy()
            m_noise_np = m_noise_np.reshape(m_noise_np.shape[0], m_noise_np.shape[1], -1, 3)
            m_noise_np = expmap_to_quaternion(m_noise_np)

            root_trajectory_flip_np = root_trajectory_flip.data.cpu().numpy()
            root_trajectory = np.flip(root_trajectory_flip_np, 1)

        return m_recon_np, m_recon_vel_np, m_np, m_noise_np, root_trajectory

    def update_learning_rate(self):
        if self.gen_scheduler is not None:
            self.gen_scheduler.step()
        if self.dis_scheduler is not None:
            self.dis_scheduler.step()

    def save_checkpoint(self, epoch):
        gen_path = os.path.join(self.model_dir, 'gen_%04d.pt' % epoch)

        logger.info("saving %s", gen_path)
        torch.save({'gen': self.gen.state_dict()}, gen_path)
        
        print('Saved model at epoch %d' % epoch)

    def load_checkpoint(self, model_path=None):
        if not model_path:
            model_dir = self.model_dir
            model_path = get_model_list(model_dir, "gen")   # last model

        state_dict = torch.load(model_path, map_location=self.device)
        self.gen.load_state_dict(state_dict['gen'])

        epochs = int(model_path[-7:-3])
        print('Load from epoch %d' % epochs)

        return epochs
