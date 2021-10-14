# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 11:27:37 2021

@author: chris
"""

from burgers_rom_vae import *
from rom_vae_dilated import *
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
    VAE = rom_vae_dilated(n_latent, act = act)
    VAE.load_state_dict(config['model_state_dict'])
    loss_reg = config['l_reg']
    loss_rec = config['l_rec']
    beta_list = config['beta_final']
    #beta_list = 0
    return VAE, loss_reg, loss_rec, beta_list, config

n_latent = 8 # latent space dimension
n_ic = 1   # number of initial conditions in dataset (for outer loop batches)
ntest = 1200    # number of time snapshots to test (the first ntrain were used in training)


trials = np.arange(3, 4)

for trial in trials:
    load_path = './Burgers1D/ic_{}/n{}/AE_{}'.format(n_ic,n_latent,trial)
    model_name = 'AE_{}.pth'.format(trial)

    save_figs = True
    save_gifs = True


    VAE, loss_reg, loss_rec, beta, config = vae_load(os.path.join(load_path, model_name))
    ntrain = config['nt'] # number of time snapshots trained on
    nstest = ntest - ntrain # the total number of unseen time snapshots

    train_data_dir = config['train_data_dir_u']#'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)#
    test_data_dir = config['test_data_dir']#'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)#
    train_loader, train_stats = load_data_new(train_data_dir, n_ic, ntrain, shuff=False)
    test_loader, test_stats = load_data_new(test_data_dir, n_ic, ntest, shuff=False)

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
    for n, (u, t, phi) in enumerate(test_loader):
        sub_loader, sub_stats = load_data_sub(u, ntest*n_ic, shuff=False)
        for m, (us, _) in enumerate(sub_loader):
            us = us.float()
            u = u.float()
            if n == 0:
                #in_test = output
                zmu_test, zlogvar_test, z_test, out_test_test, out_logvar_test_test = VAE.forward(us)
                in_test_test = us
    print('z shape: ', np.shape(zmu))
    print('t shape: ', np.shape(t))

    print('train_data shape', np.shape(in_test))
    print('test_data shape', np.shape(in_test_test))
    plt.figure(43)
    plt.plot(beta, lw=3)
    plt.xlabel('Epoch', fontsize = 22)
    plt.ylabel(r'$\beta$', fontsize = 22)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    if save_figs:
        plt.savefig(os.path.join(load_path, 'beta_{}.png'.format(trial)))
    
    #detach training data and reconstructions
    in_test = in_test.detach().numpy()
    out_test = out_test.detach().numpy()
    out_test_logvar = out_test_logvar.detach().numpy()
    out_test_var = np.exp(out_test_logvar)
    #out_test_var = np.tile(out_test_var,(ntrain,1))

    #detach testing data and predictions
    in_test_test = in_test_test.detach().numpy()
    out_test_test = out_test_test.detach().numpy()
    out_logvar_test_test = out_logvar_test_test.detach().numpy()
    out_var_test_test = np.exp(out_logvar_test_test)
    #out_var_test_test = np.tile(out_var_test_test,(ntest,1))

    
    train_test_error = np.mean((out_test-out_test_test[0:ntrain,0,:])**2)
    print('Average error between same data: ', train_test_error)

    # plot reconstruction of spatio-temporal fields
    s = np.random.randint(0,n_ic) # random sample to show recon on <<--------- random sample -----------
    
    in_test = in_test[s*ntrain:(s+1)*ntrain,0,:]
    out_test = out_test[s*ntrain:(s+1)*ntrain,0,:]
    out_test_var = np.sqrt(out_test_var[s*ntrain:(s+1)*ntrain,0,:])

    plt.figure(4, figsize = (20*ntrain/1200+4, 20))
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
    #print(np.shape(out_test_var))
    plt.imshow(np.transpose(out_test_var), cmap='jet')
    plt.title('Reconstructed Standard Deviation', fontsize=16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    if save_figs:
        plt.savefig(os.path.join(load_path, 'recon_{}.png'.format(trial)))

    # plot reconstruction of single time snapshots (on training samples)

    nx = np.shape(in_test)[-1]
    nxv = np.linspace(0,1,nx)
    plt.figure(478, figsize = (20, 6))
    plt.subplot(1,3,1)
    plt.plot(nxv, in_test[0,:], 'k', label=r'$u_0$')
    plt.plot(nxv, out_test[0,:], 'k--', label='Reconstruction Mean')
    plt.fill_between(nxv, out_test[0,:]+2*out_test_var[0,:],
            out_test[0,:]-2*out_test_var[0,:], color='black', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Reconstruction')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplot(1,3,2)
    plt.plot(nxv, in_test[ntrain//2,:], 'b', label=r'$u_{mid}$')
    plt.plot(nxv, out_test[ntrain//2,:], 'b--', label='Reconstruction Mean')
    plt.fill_between(nxv, out_test[ntrain//2,:]+2*out_test_var[ntrain//2,:],
            out_test[ntrain//2,:]-2*out_test_var[ntrain//2,:], color='blue', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Reconstruction')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplot(1,3,3)
    plt.plot(nxv, in_test[-1,:], 'r', label=r'$u_T$')
    plt.plot(nxv, out_test[-1,:], 'r--', label='Reconstruction Mean')
    plt.fill_between(nxv, out_test[-1,:]+2*out_test_var[-1,:],
            out_test[-1,:]-2*out_test_var[-1,:], color='red', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Reconstruction')
    plt.xlabel('x')
    plt.ylabel('u')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'recon_snapshots_{}.png'.format(trial)))
        
    #plot spatio-temporal field including prediction
    in_test = in_test_test[s*ntest:(s+1)*ntest,0,:]
    out_test = out_test_test[s*ntest:(s+1)*ntest,0,:]
    out_var_test = np.sqrt(out_var_test_test[s*ntest:(s+1)*ntest,0,:])

    plt.figure(485, figsize = (20*ntest/1200+4, 20))
    plt.subplot(4,1,1)
    plt.imshow(np.transpose(in_test), cmap = 'jet')
    plt.plot([ntrain,ntrain],[0,nx], 'k--', lw=3)
    plt.title('Data Sample', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,2)
    plt.imshow(np.transpose(out_test), cmap = 'jet')
    plt.plot([ntrain,ntrain],[0,nx], 'k--', lw=3)
    plt.title('Mean', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,3)
    plt.imshow(np.transpose(in_test-out_test), cmap = 'jet')
    plt.plot([ntrain,ntrain],[0,nx], 'k--', lw=3)
    plt.title('Error in Mean', fontsize = 16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    plt.subplot(4,1,4)
    #print(np.shape(out_test_var))
    plt.imshow(np.transpose(out_var_test), cmap='jet')
    plt.plot([ntrain,ntrain],[0,nx], 'k--', lw=3, label='Final Training Time')
    plt.legend()
    plt.title('Standard Deviation', fontsize=16)
    plt.xlabel('t')
    plt.ylabel('x')
    plt.colorbar()
    if save_figs:
        plt.savefig(os.path.join(load_path, 'prediction_{}.png'.format(trial)))

    #plot prediction of single time snapshots (on training samples)
    #       - This means that the first ntrain samples in timewere used in training, 
    #           but ntest-ntrain were not
    in_test = in_test_test[s*ntest+ntrain:(s+1)*ntest,0,:]
    out_test = out_test_test[s*ntest+ntrain:(s+1)*ntest,0,:]
    out_var_test = np.sqrt(out_var_test_test[s*ntest+ntrain:(s+1)*ntest,0,:])
    out_var_test_test = np.sqrt(out_var_test_test[s*ntest:(s+1)*ntest,0,:])

    nx = np.shape(in_test)[-1]
    nxv = np.linspace(0,1,nx)
    plt.figure(483, figsize = (20, 6))
    plt.subplot(1,3,1)
    plt.plot(nxv, in_test[0,:], 'k', label=r'$u_{T+1}$')
    plt.plot(nxv, out_test[0,:], 'k--', label='Prediction Mean')
    plt.fill_between(nxv, out_test[0,:]+2*out_var_test[0,:],
            out_test[0,:]-2*out_var_test[0,:], color='black', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Prediction')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplot(1,3,2)
    plt.plot(nxv, in_test[nstest//2,:], 'b', label=r'$u_{T+100}$')
    plt.plot(nxv, out_test[nstest//2,:], 'b--', label='Prediction Mean')
    plt.fill_between(nxv, out_test[nstest//2,:]+2*out_var_test[nstest//2,:],
            out_test[nstest//2,:]-2*out_var_test[nstest//2,:], color='blue', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Prediction')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.subplot(1,3,3)
    plt.plot(nxv, in_test[-1,:], 'r', label=r'$u_{T+200}$')
    plt.plot(nxv, out_test[-1,:], 'r--', label='Prediction Mean')
    plt.fill_between(nxv, out_test[-1,:]+2*out_var_test[-1,:],
            out_test[-1,:]-2*out_var_test[-1,:], color='red', alpha = 0.2, label = r'$\pm 2\sigma$')
    plt.legend()
    plt.title('Single Time Snapshot Prediction')
    plt.xlabel('x')
    plt.ylabel('u')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'time_prediction_snapshots_{}.png'.format(trial)))

    # plot the training losses over training epochs
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
    
    # visualize the latent-time field
    plt.figure(6, figsize = (13, 13))
    
    zmu = zmu.detach().numpy()
    zmu_test = zmu_test.detach().numpy()
    zstd = np.exp(0.5*zlogvar.detach().numpy())
    zstd_test = np.exp(0.5*zlogvar_test.detach().numpy())

    plt.figure(335)
    plt.imshow(np.transpose(zmu_test[s*ntrain:s*ntrain+50,:]))
    plt.xlabel('t')
    plt.ylabel('z')
    plt.title('Latent Time Evolution')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'latent_time_field_{}.png'.format(trial)))
    
    # plot evolutions of individual latent dimensions over time (gif)
    fig = plt.figure(336)
    ax = plt.axes(xlim=(0,n_latent-1), ylim=(np.min(zmu-2*zstd), np.max(zmu+2*zstd)))
    ax.set_ylabel(r'$z_i$')
    ax.set_xlabel(r'$i$')
    ax.set_title('Latent Time Evolution')
    line, = ax.plot([], [], 'k')
    fb = ax.fill_between([],[],[],lw=0, color='black', alpha = 0.15)
    lines = []
    ls = ["-", "--", "--"]
    #for i in range(3):
    #line, = ax.plot([], [], 'k', linestyle = ls[0])
    #    lines.append(line)
    #plt.subplot(1,3,1)
    def init():
        #for line in lines:
        line.set_data([], [])
        fb.set_verts([])
        return line, fb,
    def animate(i):
        i = i + s*ntest

        x = np.arange(0,n_latent)
        y = zmu_test[i,:]
        yp = y+2*zstd_test[i,:]
        ym = y-2*zstd_test[i,:]
        data = np.concatenate((x,np.flip(x)))
        datay = np.concatenate((yp,np.flip(ym)))
        data = np.vstack((data,datay))
        fb.set_verts([np.transpose(data)])
        #for num, line in enumerate(lines):
        line.set_data(x,y)
        return line, fb,
    if save_gifs:
        anim = FuncAnimation(fig, animate, init_func = init, frames = ntest, interval=10,blit=True)
        anim.save(os.path.join(load_path, 'latent_time_snapshots_{}.gif'.format(trial)), 
                writer='imagemagick')
    
    # plot evolution of reconstructed solution over time (gif) [training time only]
    in_test = in_test_test[s*ntest:(s+1)*ntest,0,:]
    out_test = out_test_test[s*ntest:(s+1)*ntest,0,:]
    fig = plt.figure(978)
    ax = plt.axes(xlim=(0,1), ylim=(-1.1, 1.1))
    ax.set_ylabel(r'$x$')
    ax.set_xlabel(r'$u(x)$')
    ax.set_title('Latent Time Evolution')
    line, = ax.plot([], [], 'k--')
    line_true, = ax.plot([], [], 'k')
    fb = ax.fill_between([],[],[],lw=0, color='black', alpha = 0.15)
    def init():
        #for line in lines:
        line.set_data([], [])
        line_true.set_data([], [])
        fb.set_verts([])
        return line, line_true, fb,
    def animate(i):
        i = i 

        x = nxv
        y = out_test[i,:]
        y_true = in_test[i,:]
        yp = y+2*np.sqrt(out_var_test_test[i,:])
        ym = y-2*np.sqrt(out_var_test_test[i,:])
        data = np.concatenate((x,np.flip(x)))
        datay = np.concatenate((yp,np.flip(ym)))
        data = np.vstack((data,datay))
        fb.set_verts([np.transpose(data)])
        #for num, line in enumerate(lines):
        line.set_data(x,y)
        line_true.set_data(x,y_true)
        return line, line_true, fb,
    if save_gifs:
        anim = FuncAnimation(fig, animate, init_func = init, frames = ntest, interval=10,blit=True)
        anim.save(os.path.join(load_path, 'data_snapshots_{}.gif'.format(trial)), 
                writer='imagemagick')

    # plot the initial and final training time snapshot distribution in latent space
    plt.figure(337, figsize = (10,4))
    plt.xlabel(r'$i$')
    plt.ylabel(r'$z_i$')
    plt.title('z(0)')
    plt.subplot(1,2,1)
    plt.plot(zmu[s*ntrain,:], 'k', label=r'$z_0$')
    plt.plot(zmu[s*ntrain,:] + 2*zstd[s*ntrain,:], 'k--')
    plt.plot(zmu[s*ntrain,:] - 2*zstd[s*ntrain,:], 'k--')
    plt.xlabel(r'$i$')
    plt.ylabel(r'$z_i$')
    plt.title('z(0)')
    plt.subplot(1,2,2)
    plt.plot(zmu[(s+1)*ntrain-1,:], 'r', label=r'$z_T$')
    plt.plot(zmu[(s+1)*ntrain-1,:] + 2*zstd[(s+1)*ntrain-1,:], 'r--')
    plt.plot(zmu[(s+1)*ntrain-1,:] - 2*zstd[(s+1)*ntrain-1,:], 'r--')
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
    plt.figure(339, figsize = (20, 5))
    kde = KernelDensity(bandwidth = 0.5, kernel = 'gaussian')
    for i in range(n_latent):
        v = np.concatenate((z[:,i].detach().numpy(), z_test[:,i].detach().numpy()), axis = 0)
        kde.fit(np.reshape(v, (-1,1)))
        plotv = np.linspace(np.min(v), np.max(v), 100)
        lp = kde.score_samples(np.reshape(plotv, (-1, 1)))
        plt.subplot(1, n_latent, i + 1)
        plt.xlabel(r'$z_{}$'.format(i+1), fontsize = 18)
        plt.plot( plotv, np.exp(lp), 'k--')
        plt.title('Agg Post Marginal')
    if save_figs:
        plt.savefig(os.path.join(load_path, 'disentanglement_{}.png'.format(trial)))

    plt.figure(333, figsize=(20, 5))
    for i in range(n_latent):
        plt.subplot(1,n_latent,i+1)
        plt.plot(t[s,0:ntrain],zmu[s*ntrain:(s+1)*ntrain,i],'k')
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
