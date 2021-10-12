# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:27:37 2021

@author: chris
"""

from burgers_rom_vae import *
from load_data_new import load_data_new
from load_data_sub import load_data_sub
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal
from mpl_toolkits import mplot3d
from matplotlib.animation import FuncAnimation

plt.close('all')

def vae_load(path):
    config = torch.load(path)
    data_channels = 3
    initial_features = config['initial_features']
    growth_rate = config['growth_rate']
    n_latent = config['n_latent']
    dense_blocks = config['dense_blocks']
    act = config['activations']
    VAE = burgers_rom_vae(n_latent, act = act)
    VAE.load_state_dict(config['model_state_dict'])
    loss_reg = config['l_reg']
    loss_rec = config['l_rec']
    beta_list = config['beta_final']
    #beta_list = 0
    return VAE, loss_reg, loss_rec, beta_list, config

n_latent = 8
n_ic = 10
ntrain = 1000
ntest = 1000

trials = np.arange(0, 1)
for trial in trials:
    load_path = './Burgers1D/ic_{}/n{}/VAE_{}'.format(n_ic,n_latent,trial)
    model_name = 'VAE_{}.pth'.format(trial)

    save_figs = True

    train_data_dir = 'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)#
    test_data_dir = 'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)#

    train_loader, train_stats = load_data_new(train_data_dir, n_ic, shuff=False)
    test_loader, test_stats = load_data_new(test_data_dir, n_ic, shuff=False)

    VAE, loss_reg, loss_rec, beta, config = vae_load(os.path.join(load_path, model_name))
    print('Rec loss: ', loss_rec[-1])
    print('Loss: ', loss_reg[-1] + loss_rec[-1])
        
    for n, (u, _, _) in enumerate(train_loader):
        sub_loader, sub_stats = load_data_sub(u, ntrain*n_ic, shuff=False)
        for m, (us, _) in enumerate(sub_loader):
            us = us.float()
            u = u.float()
            if n == 0:
                zmu, zlogvar, z, out_test, out_test_logvar = VAE.forward(us)
                in_test = us
        U = u
    # test data
    for n, (u, t) in enumerate(test_loader):
        sub_loader, sub_stats = load_data_sub(u, test*n_ic, shuff=False)
        us = us.float()
        u = u.float()
            if n == 0:
                #in_test = output
                zmu_test, zlogvar_test, z_test, out_test_test, _ = VAE.forward(us)
    print('z shape: ', np.shape(zmu))

    plt.figure(43)
    plt.plot(beta, lw=3)
    plt.xlabel('Epoch', fontsize = 22)
    plt.ylabel(r'$\beta$', fontsize = 22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if save_figs:
        plt.savefig(os.path.join(load_path, 'beta_{}.png'.format(trial)))

    in_test = in_test.detach().numpy()
    out_test = out_test.detach().numpy()
    out_test_logvar = out_test_logvar.detach().numpy()
    out_test_var = np.exp(out_test_logvar)
    out_test_var = np.tile(out_test_var,(ntrain,1))

    s = np.random.randint(0,n_ic) # random sample to show recon on
    in_test = in_test[s*ntrain:(s+1)*ntrain,0,:]
    out_test = out_test[s*ntrain:(s+1)*ntrain,0,:]

    plt.figure(4, figsize = (24, 20))
    plt.subplot(4,1,1)
    plt.imshow(np.transpose(in_test), cmap = 'jet')
    plt.title('Data Sample', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,2)
    plt.imshow(np.transpose(out_test), cmap = 'jet')
    plt.title('Reconstructed Mean', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,3)
    plt.imshow(np.transpose(in_test-out_test), cmap = 'jet')
    plt.title('Error in Mean', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,4)
    print(np.shape(out_test_var))
    plt.imshow(np.transpose(out_test_var), cmap='jet')
    plt.title('Reconstructed Variance', fontsize=16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    if save_figs:
        plt.savefig(os.path.join(load_path, 'recon_{}.png'.format(trial)))

    plt.figure(10)
    plt.plot(beta*loss_reg + loss_rec, 'r', label = r'Training Loss', lw=3)
    plt.plot(loss_reg + loss_rec, 'k', label = r'$L_{VAE}$', lw=3)
    plt.legend(prop={"size":16})
    plt.ylabel(r'Loss', fontsize=22)
    plt.xlabel(r'Epochs', fontsize=22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.ylim(-18,10)
    if save_figs:
        plt.savefig(os.path.join(load_path, 'training_losses_{}.png'.format(trial)))

    plt.figure(13)
    ind = config['epochs'] - 1250
    
    # plot all latent data correlations with gen parameters
    plt.figure(6, figsize = (13, 13))
    '''
    count = 0
    for i in range(n_latent):
        for j in range(kle):
            if j == 0:
                count = count + 2
            else:
                count = count + 1
            plt.subplot(n_latent+1, kle+1, count)
            if (i == 0) & (j==kle-1):
                plt.plot(g[:,j], zmu[:,i].detach().numpy(), 'k.', markersize = 1, label = 'Train Data')
                plt.plot(g_test[:,j],zmu_test[:,i].detach().numpy(),'r.', markersize = 1, label = 'Test Data')
                plt.legend(prop={"size":14}, markerscale = 3.)
            else:
                plt.plot(g[:,j], zmu[:,i].detach().numpy(), 'k.', markersize = 1)
                plt.plot(g_test[:,j],zmu_test[:,i].detach().numpy(),'r.', markersize = 1)
    '''            
    ztest = zmu.detach().numpy()
    zstd = np.exp(0.5*zlogvar.detach().numpy())
    plt.figure(335)
    plt.imshow(np.transpose(ztest[0:100,:]))
    plt.xlabel('t')
    plt.ylabel('z')
    plt.title('Latent Time Evolution')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'latent_time_field_{}.png'.format(trial)))
    
    fig = plt.figure(336)
    ax = plt.axes(xlim=(0,n_latent-1), ylim=(np.min(ztest-2*zstd), np.max(ztest+2*zstd)))
    ax.set_ylabel(r'$z_i$')
    ax.set_xlabel(r'$i$')
    ax.set_title('Latent Time Evolution')
    line, = ax.plot([], [], 'k')
    lines = []
    ls = ["-", "--", "--"]
    for i in range(3):
        line, = ax.plot([], [], 'k', linestyle = ls[i])
        lines.append(line)
    #plt.subplot(1,3,1)
    def init():
        for line in lines:
            line.set_data([], [])
        return lines
    def animate(i):
        #plt.plot(ztest[0,:] + 2*zstd[0,:], 'k--')
        #plt.plot(ztest[0,:] - 2*zstd[0,:], 'k--')
        x = np.arange(0,n_latent)
        y = ztest[i,:]
        yp = ztest[i,:]+2*zstd[i,:]
        ym = ztest[i,:]-2*zstd[i,:]
        xlist = [x, x, x]
        ylist = [y, yp, ym]
        for num, line in enumerate(lines):
            line.set_data(xlist[num],ylist[num])
        return lines

    anim = FuncAnimation(fig, animate, init_func = init, frames = ntrain, interval=10,blit=True)
    anim.save(os.path.join(load_path, 'latent_time_snapshots_{}.gif'.format(trial)), writer='imagemagick')
    plt.figure(337)
    plt.xlabel(r'$i$')
    plt.ylabel(r'$z_i$')
    plt.title('z(0)')
    plt.subplot(1,2,1)
    plt.plot(ztest[0,:], 'k', label=r'$z_0$')
    plt.plot(ztest[0,:] + 2*zstd[0,:], 'k--')
    plt.plot(ztest[0,:] - 2*zstd[0,:], 'k--')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$z_i$')
    plt.title('z(0)')
    plt.subplot(1,2,2)
    plt.plot(ztest[-1,:], 'r', label=r'$z_T$')
    plt.plot(ztest[-1,:] + 2*zstd[-1,:], 'r--')
    plt.plot(ztest[-1,:] - 2*zstd[-1,:], 'r--')
    #plt.legend()
    plt.xlabel(r'$i$')
    plt.ylabel(r'$z_i$')
    plt.title('z(T)')
    
    if save_figs:
        plt.savefig(os.path.join(load_path, 'latent_time_snapshots_{}.png'.format(trial)))
    ''' 
    kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
    for i in range(n_latent):
        v = np.concatenate((g[:,i], g_test[:,i]), axis = 0)
        kde.fit(np.reshape(v, (-1,1)))
        plotv = np.linspace(np.min(v), np.max(v), 1000)
        lp = kde.score_samples(np.reshape(plotv, (-1, 1)))
        plt.subplot(n_latent+1, kle+1, (n_latent*(kle+1) + i + 2))
        plt.xlabel(r'$\theta_{}$'.format(i+1), fontsize = 18)
        plt.plot(plotv, np.exp(lp), 'k--')
    ''' 
    plt.figure(339)
    kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
    for i in range(n_latent):
        v = np.concatenate((z[:,i].detach().numpy(), z_test[:,i].detach().numpy()), axis = 0)
        kde.fit(np.reshape(v, (-1,1)))
        plotv = np.linspace(np.min(v), np.max(v), 1000)
        lp = kde.score_samples(np.reshape(plotv, (-1, 1)))
        plt.subplot(n_latent+1, kle+1, i*(kle+1) + 1)
        plt.xlabel(r'$z_{}$'.format(i+1), fontsize = 18)
        plt.plot( np.exp(lp), plotv, 'k--')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'disentanglement_{}.png'.format(trial)))

    plt.figure(333, figsize=(20, 10))
    for i in range(n_latent):
        plt.subplot(1,n_latent,i+1)
        plt.plot(t,ztest[:,i],'k')
        plt.xlabel('t')
        plt.ylabel(r'$z_{}$'.format(i+1))
        plt.title('Latent Dim Over Time')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'latent_model_{}.png'.format(trial)))
        
    '''
    ztest = zmu.detach().numpy()
    ztest = ztest[:,0:2]
    g = g[:,0:2]
    #ztest = np.reshape(ztest, (-1, 1))
    n_samples = 100
    xv1 = np.linspace(-4, 4, 100)
    xv2 = xv1 + 0.
    [XM1, XM2] = np.meshgrid(xv1, xv2)
    fitM = np.concatenate((np.reshape(XM1, (-1,1)), np.reshape(XM2, (-1,1))), axis = 1)
    kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
    kde.fit(ztest)
    lp = kde.score_samples(fitM)
    kde.fit(g)
    lpg = kde.score_samples(fitM)
    prior = multivariate_normal([0, 0], [[1, 0], [0, 1]])
    plt.figure(7, figsize=(7,7))
    plt.contour(XM1, XM2, np.exp(np.reshape(lp, (n_samples, n_samples))), colors = 'red')
    plt.contour(XM1, XM2, np.exp(np.reshape(lpg, (n_samples, n_samples))), colors = 'blue')
    plt.contour(XM1, XM2, prior.pdf(np.dstack((XM1, XM2))), colors = 'black')
    plt.gca().set_aspect('equal') 
    plt.xlabel(r'$z_1$', fontsize = 18)
    plt.ylabel(r'$z_2$', fontsize = 18)
    plt.title('Aggregated Posterior - Prior Comparison (VAE)', fontsize = 16)
    proxy = [plt.Rectangle((0,0),1,1,fc = 'red'), plt.Rectangle((0,0),1,1,fc = 'blue'), plt.Rectangle((0,0),1,1,fc = 'black')]
    plt.legend(proxy, ['Aggregated Posterior', 'Generative Parameters', 'Prior'], prop={"size":16}, loc = 2)
    if save_figs:
        plt.savefig(os.path.join(load_path, 'agg_post_comparison_{}.png'.format(trial)))
    plt.figure(27)
    plt.plot(zmu[:,0].detach().numpy(), zmu[:,1].detach().numpy(), 'k.', markersize = 1, label = 'Train Data')
    plt.plot(zmu_test[:,0].detach().numpy(), zmu_test[:,1].detach().numpy(), 'r.', markersize = 1, label = 'Test Data')
    plt.xlabel(r'$z_1$')
    plt.ylabel(r'$z_2$')
    plt.title(r'Samples from $q_\phi(z)$')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'agg_post_samples_{}.png'.format(trial)))
    '''    
    plt.figure(334)
    plt.semilogy(config['gradient_list'], 'k')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Gradient Norm')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'grad_norm_{}.png'.format(trial)))

    plt.close('all')
