"""
Code by Christian Jacobsen, University of Michigan 2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


class _PadCyclic_1d(nn.Module):
    # pads the input with the right-most value on the left and the left-most value on the right
    def __init__(self, pad):
        super(_PadCyclic_1d, self).__init__()
        self.pad = pad

    def forward(self, x):
        shape = (np.shape(x)[0], np.shape(x)[1], self.pad)
        return torch.cat([x[:, :, -self.pad:].view(shape), x, x[:, :, :self.pad].view(shape)], -1)


class _Reshape(nn.Module):
    def __init__(self, shape):
        super(_Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(self.shape)


class _DenseBlockLayer_1d(nn.Sequential):
    def __init__(self, in_features, growth_rate, act=nn.ReLU(inplace=True)):
        super(_DenseBlockLayer_1d, self).__init__()

        self.add_module('cyclic_pad', _PadCyclic_1d(1))
        #self.add_module('norm', nn.BatchNorm1d(in_features))
        self.add_module('act', act)
        self.add_module('conv', nn.Conv1d(in_features, growth_rate,
            kernel_size=3, stride=1, padding=0, bias=False))

    def forward(self, x):
        return torch.cat([x, super(_DenseBlockLayer_1d, self).forward(x)], 1)


class _DenseBlock_1d(nn.Sequential):
    def __init__(self, n_layers, in_features, growth_rate, act=nn.ReLU(inplace=True)):
        super(_DenseBlock_1d, self).__init__()
        for i in range(n_layers):
            layer = _DenseBlockLayer_1d(
                in_features + i*growth_rate, growth_rate, act)
            self.add_module('denseblocklayer_1d_%d' % (i+1), layer)


class _EncodeBlock_1d(nn.Sequential):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        super(_EncodeBlock_1d, self).__init__()

        self.add_module('cyclic_pad1', _PadCyclic_1d(1))
        #self.add_module('norm1', nn.BatchNorm1d(in_features))
        self.add_module('act1', act)
        self.add_module('conv1', nn.Conv1d(in_features, out_features,
            kernel_size=3, stride=1, padding=0, bias=False))
        #self.add_module('norm2', nn.BatchNorm1d(out_features))
        self.add_module('act2', act)
        self.add_module('pool', nn.MaxPool1d(2, stride=2))


class _DecodeBlock_1d(nn.Sequential):
    def __init__(self, in_features, out_features, act=nn.ReLU(inplace=True)):
        super(_DecodeBlock_1d, self).__init__()

        self.add_module('cyclic_pad1', _PadCyclic_1d(1))
        #self.add_module('norm1', nn.BatchNorm1d(in_features))
        self.add_module('act1', act)
        self.add_module('conv1', nn.Conv1d(in_features, out_features,
            kernel_size=3, stride=1, padding=0, bias=False))
        #self.add_module('norm2', nn.BatchNorm1d(out_features))
        self.add_module('act2', act)
        self.add_module('upsample', nn.Upsample(
            scale_factor=2, mode='linear', align_corners=False))


class _FinalDecode_1d(nn.Sequential):
    def __init__(self, in_features, out_channels, act=nn.ReLU(inplace=True)):
        super(_FinalDecode_1d, self).__init__()

        self.add_module('cyclic_pad1', _PadCyclic_1d(1))
        #self.add_module('norm1', nn.BatchNorm1d(in_features))
        self.add_module('act1', act)
        self.add_module('conv1', nn.Conv1d(in_features, in_features//2,
            kernel_size=3, stride=1, padding=0, bias=False))
        #self.add_module('norm2', nn.BatchNorm1d(in_features//2))
        self.add_module('act2', act)
        self.add_module('upsample', nn.Upsample(
            scale_factor=2, mode='linear', align_corners=False))

class _DilationBlock_1d(nn.Sequential):
    def __init__(self, in_chan, out_chan, dil, act=nn.ReLU(inplace=True)):
        super(_DilationBlock_1d, self).__init__()

        self.add_module('cyclic_pad', _PadCyclic_1d(dil))
        #self.add_module('norm', nn.BatchNorm1d(in_chan))
        self.add_module('dil_conv', nn.Conv1d(in_chan, out_chan, kernel_size=3, dilation=dil))
        self.add_module('act', act)

class rom_class(nn.Module):
    def __init__(self, n_latent, tau=1, act=nn.ReLU()):

        super(rom_class, self).__init__()

        # set manually for testing
        data_channels = 1
        initial_features = 2
        K = 4  # growth rate
        self.n_latent = n_latent
        self.tau = tau # this is the lookback window size

        self.F = nn.Sequential()
        self.lv = nn.Parameter(torch.zeros((1, self.n_latent)))

        
        # input shape: (n_batch, n_lookback, n_x)
        d = [1, 2]
        channels = [self.tau, 32, 32]
        for (n, dil) in enumerate(d):
            dil_block = _DilationBlock_1d(channels[n], channels[n+1], dil, act)
            self.F.add_module('dilation_{}'.format(n+1), dil_block)

        self.F.add_module('final_conv', nn.Conv1d(channels[-1], 1, kernel_size=1, stride=1, dilation=1))
        self.F.add_module('final_reshape', _Reshape((-1, self.n_latent)))

    def forward(self, ztau):
        zmu, zlogvar=self.F(ztau), self.lv
        #C = 7 # correlation length.. estimated as maximum receptive field size (dilation 8)
        #weights = F.softmax(-zlogvar, dim=-1)
        #zmu = (zmu * weights).sum(dim=-1).view(-1, self.n_latent)
        #zlogvar = C / torch.sum(torch.exp(-zlogvar), dim=-1)
        #zlogvar = torch.log(zlogvar.view(-1, self.n_latent))
        #z=self._reparameterize(zmu, zlogvar)
        return zmu, zlogvar

    def _reparameterize(self, mu, logvar):
        std=logvar.mul(0.5).exp_()
        eps=torch.randn(*mu.size()).type_as(mu)
        return mu + std * eps

    def gaussian_log_prob(self, x, mu, logvar):
        return -0.5*((x-mu)**2/torch.exp(logvar) + math.log(2*math.pi) + logvar)

    def compute_kld(self, zmu, zlogvar):
        return torch.Tensor([0])#0.5*(zmu**2 + torch.exp(zlogvar) - 1 - zlogvar)
        #return 0.5*(zmu**2/torch.exp(self.prior_logvar) + torch.exp(zlogvar)/torch.exp(self.prior_logvar) - 1 - zlogvar + self.prior_logvar)#0.5*(2*math.log(0.25)- 0.5*torch.sum(zlogvar, 1) - 2 + 1/0.25*torch.sum(zlogvar.mul(0.5).exp_(), 1) + torch.sum((0.5-zmu)**2, 1))#

    def compute_kld_ss(self, zmu, zlogvar, zl):
        return 0.5*((zmu-zl)**2/torch.exp(self.zl_logvar) + torch.exp(zlogvar)/torch.exp(self.zl_logvar) - 1 - zlogvar + self.zl_logvar)

    def compute_loss(self, ztau, z):
        # in this case, we have xu = unlabeled data, xl = labeled data, zl = representation of labeled data
        # freebits = 0
        zmu, zlogvar=self.forward(ztau)
        l_rec=-torch.sum(self.gaussian_log_prob(z, zmu, zlogvar), 1)
        l_reg=0#self.compute_kld(zmu, zlogvar)
        # l_ss = -torch.sum(self.gaussian_log_prob(zl, zmu[np.shape(xu)[0]:,:], zlogvar[np.shape(xu)[0]:,:]), 1)
        # l_ss = -self.compute_kld_ss(zmu[np.shape(xu)[0]:,:], zlogvar[np.shape(xu)[0]:,:], zl)
        return zmu, zlogvar, z, xmu, xlogvar, l_rec, l_reg  # , l_ss

    def update_beta(self, beta, rec, nu, tau):
        def H(d):
            if d > 0:
                return 1.0
            else:
                return 0.0

        def f(b, d, t):
            return (1-H(d))*math.tanh(t*(b-1)) - H(d)

        return beta*math.exp(nu*f(beta, rec, tau)*rec)

    def compute_dis_score(self, p, z):
        # compute disentanglement score where p are true parameter samples and z are latent samples
        if p.is_cuda:
            p=p.cpu().detach().numpy()
            z=z.cpu().detach().numpy()
        else:
            p=p.detach().numpy()
            z=z.detach().numpy()

        score=0
        for i in range(z.shape[1]):
            m=np.concatenate((z[:, i].reshape((-1, 1)), p), axis=1)
            m=np.transpose(m)
            c=np.cov(m)
            score=np.max(np.abs(c[0, 1:]))/np.sum(np.abs(c[0, 1:])) + score

        return score / z.shape[1]
