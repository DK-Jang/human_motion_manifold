from torch import nn
from torch.autograd import Variable
import torch
import torch.nn.functional as F
import numpy as np

##################################################################################
# Generator
##################################################################################
class MotionGen(nn.Module):
    def __init__(self, input_size, seq_len, z_dim, params):
        super(MotionGen, self).__init__()
        # for network
        self.seq_len = seq_len
        self.input_size = input_size

        # for rnn
        rnn_dropout = params['rnn_dropout']
        num_units = params['num_units']
        n_layers = params['n_layers']
        concat = params['concat']   # bool
        
        # content encoder
        self.enc = Encoder_RNN(input_size, z_dim, self.seq_len, rnn_dropout, num_units,
                               n_layers)
        self.dec = Decoder_RNN(input_size, z_dim, self.seq_len, rnn_dropout, num_units,
                               n_layers, False, concat)
        self.dec_vel = Decoder_RNN(input_size, z_dim, self.seq_len, rnn_dropout, num_units, 
                                   n_layers, True, concat)

        self.apply(self._init_weights)

    def forward(self, motions, motions_flip):
        z = self.encode(motions)
        motions_recon_flip, motions_recon_vel_flip = self.decode(z)

        return motions_recon_flip, motions_recon_vel_flip, z

    def encode(self, motions):
        z = self.enc(motions, h=None)
        return z

    def decode(self, z=None):
        motions_recon_flip = []
        motions_recon_vel_flip = []
        batch_z = z
        dec_first_in = Variable(torch.zeros(z.shape[0], 1, self.input_size).to(batch_z.device))

        dec_out, h_dec = self.dec(x=dec_first_in, z=batch_z, h=None)
        dec_out_vel, h_dec_vel = self.dec_vel(x=dec_first_in, z=batch_z, h=None)
        motions_recon_flip.append(dec_out)
        motions_recon_vel_flip.append(dec_out_vel)

        for i in range(self.seq_len - 1):
            # Feed own output
            dec_out, h_dec = self.dec(x=dec_out, z=batch_z, h=h_dec)
            dec_out_vel, h_dec_vel = self.dec_vel(x=dec_out_vel, z=batch_z, h=h_dec_vel)
            motions_recon_flip.append(dec_out)
            motions_recon_vel_flip.append(dec_out_vel)

        motions_recon_flip = torch.cat(motions_recon_flip, dim=1)
        motions_recon_vel_flip = torch.cat(motions_recon_vel_flip, dim=1)

        return motions_recon_flip, motions_recon_vel_flip
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


#################################################################################
# Discriminator
##################################################################################
class MotionDis(nn.Module):
    # Motion manifold discriminator architecture
    def __init__(self, input_dim, params):
        super(MotionDis, self).__init__()
        n_layer = params['n_layer']
        dim = params['dim']
        norm = params['norm']
        activ = params['activ']
        pad_type = params['pad_type']
        self.drop = nn.Dropout(p=0.2)
        self.norm = params['norm']
        self.gan_type = params['gan_type']

        self.model = []
        self.model += [Conv1dBlock(input_dim, dim, 4, 2, 1, norm='none', activation=activ, pad_type=pad_type)]
        for i in range(n_layer - 1):
            self.model += [Conv1dBlock(dim, dim * 2, 4, 2, 1, norm=norm, activation=activ, pad_type=pad_type)]
            dim *= 2
        self.model += [nn.Conv1d(dim, 1, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)

        self.apply(self._init_weights)

    def forward(self, x):
        x = x.permute(0,2,1)        # (batch, dim, seq)
        if not self.norm == 'in':   # instant norm
            x = self.drop(x)
        logits = self.model(x)
        return logits

    def calc_dis_loss(self, input_fake, input_real):
        # calculate the loss to train D
        outs0 = self.forward(input_fake)
        outs1 = self.forward(input_real)
        loss = 0

        for it, (out0, out1) in enumerate(zip(outs0, outs1)):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 0) ** 2) + torch.mean((out1 - 1) ** 2)
            elif self.gan_type == 'nsgan':
                all0 = Variable(torch.zeros_like(out0.data).cuda(), requires_grad=False)
                all1 = Variable(torch.ones_like(out1.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all0) +
                                   F.binary_cross_entropy(F.sigmoid(out1), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss

    def calc_gen_loss(self, input_fake):
        # calculate the loss to train G
        outs0 = self.forward(input_fake)
        loss = 0
        for it, (out0) in enumerate(outs0):
            if self.gan_type == 'lsgan':
                loss += torch.mean((out0 - 1) ** 2)  # LSGAN
            elif self.gan_type == 'nsgan':
                all1 = Variable(torch.ones_like(out0.data).cuda(), requires_grad=False)
                loss += torch.mean(F.binary_cross_entropy(F.sigmoid(out0), all1))
            else:
                assert 0, "Unsupported GAN type: {}".format(self.gan_type)
        return loss
    
    def _init_weights(self, module):
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            nn.init.kaiming_normal_(module.weight.data, a=0, mode='fan_in')
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.Linear)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()


##################################################################################
# Encoder and Decoders
##################################################################################
class Encoder_RNN(nn.Module):
    def __init__(self, input_size, z_dim, seq_len, rnn_dropout, 
                 num_units, n_layers):
        super(Encoder_RNN, self).__init__()
        self.seq_len = seq_len

        self.rnn = nn.GRU(input_size=input_size, hidden_size=num_units,
                          num_layers=n_layers, dropout=rnn_dropout, batch_first=True)
        self.fc_h2z = LinearBlock(input_dim=num_units*n_layers, output_dim=z_dim,
                                  norm='none', activation='none')
        self.h0 = nn.Parameter(torch.zeros(n_layers, 1, num_units).normal_(std=0.01),
                               requires_grad=True)

    def forward(self, x, h=None):
        """
        Run a forward pass of this model.

        Arguments:
             -- x: input tensor of shape (N, L, J), where N is the batch size, L is the sequence length,
                   J is input size.
             -- h: hidden state. If None, it defaults to the learned initial state.
                   (num_layers * num_directions, batch, hidden_size)
             -- z: (batch_size, z_dim)
        """
        if h is None:
            h = self.h0.expand(-1, x.shape[0], -1).contiguous()

        x_out, h = self.rnn(x, h)
        h = h.contiguous().view(h.shape[1], h.shape[0] * h.shape[2])
        z = self.fc_h2z(h)

        return z


class Decoder_RNN(nn.Module):
    def __init__(self, input_size, z_dim, seq_len, rnn_dropout, num_units,
                 n_layers, residual, concat):
        """
        Construct a Encoder.
        Arguments:
            -- input_size:
            -- z: (batch_size, z_dim)
            -- output_size: number of actions.
            -- params: add a quaternion multiplication block on the RNN output to force
                       the network to model velocities instead of absolute rotations.
        """
        super(Decoder_RNN, self).__init__()

        self.seq_len = seq_len
        self.num_units = num_units
        self.residual = residual
        self.concat = concat

        if self.concat:
            self.rnn = nn.GRU(input_size=input_size+z_dim, hidden_size=num_units,
                              num_layers=n_layers, dropout=rnn_dropout, batch_first=True)
        else:
            self.rnn = nn.GRU(input_size=input_size, hidden_size=num_units,
                              num_layers=n_layers, dropout=rnn_dropout, batch_first=True)

        self.fc_z2h = LinearBlock(input_dim=z_dim, output_dim=num_units * n_layers,
                                  norm='none', activation='none')
        self.fc_rnn2out = LinearBlock(input_dim=num_units, output_dim=input_size,
                                      norm='none', activation='none')
        self.h0 = nn.Parameter(torch.zeros(n_layers, 1, num_units).normal_(std=0.01),
                               requires_grad=True)

    def forward(self, x, z=None, h=None):
        """
        Run a forward pass of this model.

        Arguments:
            -- x: input tensor of shape (N, L, J), where N is the batch size, L is the sequence length,
                  J is the number of joints, O is the number of actions.
            -- h: hidden state. If None, it defaults to the learned initial state.
                  (num_layers * num_directions, batch, hidden_size)
            -- z: (batch_size, z_dim)
        """

        if h is None:
            h = self.fc_z2h(z)
            h = h.contiguous().view(-1, z.shape[0], self.num_units)
        
        if self.concat:
            x_concat = torch.cat((x, z.unsqueeze(1).expand(-1, x.shape[1], -1)), dim=2)
            x_out, h = self.rnn(x_concat, h)
        else:
            x_out, h = self.rnn(x, h)
        x_out = self.fc_rnn2out(x_out)

        if self.residual:
            # Add the residual connection
            x_out = torch.add(x_out, x)

        return x_out, h


##################################################################################
# Basic Blocks
##################################################################################
class Conv1dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride,
                 padding=0, norm='none', activation='relu', pad_type='zero'):
        super(Conv1dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad1d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad1d(padding)
        elif pad_type == 'zero':
            self.pad = None     # just using default function
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            #self.norm = nn.InstanceNorm2d(norm_dim, track_running_stats=True)
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if pad_type == 'zero':
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride, padding,
                                  bias=self.use_bias)
        else:
            self.conv = nn.Conv1d(input_dim, output_dim, kernel_size, stride,
                                  bias=self.use_bias)

    @staticmethod
    def calc_samepad_size(input_dim, kernel_size, stride, dilation=1):
        # just for zero pad='same'
        return ((input_dim - 1) * stride - input_dim + kernel_size + (kernel_size-1)*(dilation-1)) / 2

    def forward(self, x):
        if self.pad:
            x = self.pad(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class LinearBlock(nn.Module):
    def __init__(self, input_dim, output_dim, norm='none', activation='relu'):
        super(LinearBlock, self).__init__()
        use_bias = True
        # initialize fully connected layer
        self.fc = nn.Linear(input_dim, output_dim, bias=use_bias)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm1d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm1d(norm_dim)
        elif norm == 'ln':
            self.norm = nn.LayerNorm(norm_dim)
        elif norm == 'none' or norm == 'sn':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

    def forward(self, x):
        out = self.fc(x)
        if self.norm:
            out = self.norm(out)
        if self.activation:
            out = self.activation(out)
        return out


##################################################################################
# etc
##################################################################################
class P_Z(object):
    def __init__(self, length):
        self.length = length

    def sample(self, batch_size):
        """
        return tensor ops(vars) generating sample from P_z
        """
        raise NotImplemented


class Gaussian_P_Z(P_Z):
    def __init__(self, length):
        super().__init__(length)  # 같은 P_Z을 상속받고 있으므로 super 함수를 쓴다

    def sample(self, batch_size):
        mean = np.zeros(self.length)
        # cov = np.identity(self.length) * 0.5    # CMU
        cov = np.identity(self.length)        # H36M
        s = np.random.multivariate_normal(mean, cov, batch_size).astype(np.float32)
        s = Variable(torch.from_numpy(s))
        return s
        # return tf.random_normal([batch_size,self.length])

    def sample_np(self, batch_size):
        mean = np.zeros(self.length)
        cov = np.identity(self.length)
        s = np.random.multivariate_normal(mean, cov, batch_size).astype(np.float32)
        return s