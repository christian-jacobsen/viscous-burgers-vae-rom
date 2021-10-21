# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:41:10 2021

@author: Christian Jacobsen, University of Michigan
"""

from burgers_rom_vae import *
from rom_vae_dilated import *
from rom_class import *
from load_data_new import load_data_new
from load_data_rom import load_data_rom
from load_data_sub import load_data_sub
import torch
import numpy as np
import time


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
    return VAE, config

def rom_train(train_data_dir_u, test_data_dir, save_dir, filename, \
                       epochs, batch_size_u, test_batch_size, nt, tau_lookback, \
                       wd, lr_schedule, \
                       data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                       prior, activations, vae_path):

    sub_batch_size = nt//10
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
       
    train_loader_u, train_stats_u = load_data_new(train_data_dir_u, batch_size_u, nt) # unlabeled data
    
    VAE, config = vae_load(vae_path)
    VAE = VAE.to(device)
    ROM = rom_class(n_latent, tau_lookback, activations)
    ROM = ROM.to(device)
    optimizer = torch.optim.Adam(ROM.parameters(), lr=lr_schedule(0), weight_decay = wd)
    l_rec_list = np.zeros((epochs,))
    l_reg_list = np.zeros((epochs,))
    #l_ss_list = np.zeros((epochs,))
    ROM.train()

    print_epochs = 1
    t_start = time.time()
    for epoch in range(epochs):
        if epoch % print_epochs == 0:
            print('=======================================')
            print('Epoch: ', epoch) 
        optimizer.param_groups[0]['lr'] = lr_schedule(epoch) #update learning rate
        
        for n, (u, _, _) in enumerate(train_loader_u): # we will load all data in 1 batch here
            u = u.to(device, dtype=torch.float) 
            zmu, zlogvar, _, _, _ = VAE.forward(u.view((-1, 1, 128))) # get latent conditionals for all training data
            sub_loader, sub_stats = load_data_rom(zmu.cpu(), zlogvar.cpu(), tau_lookback, sub_batch_size)
            for m, (muzkT, lvzkT, muz, lvz) in enumerate(sub_loader):
                muzkT = muzkT.to(device, dtype=torch.float)
                lvzkT = lvzkT.to(device, dtype=torch.float)
                muz = muz.to(device, dtype=torch.float)
                lvz = lvz.to(device, dtype=torch.float)
                zkT = ROM._reparameterize(muzkT, lvzkT)
                if epoch==0: # compute initialized losses
                    _, _, l_rec, l_reg = ROM.compute_loss(zkT, muz, lvz)
                    l_rec_0 = torch.mean(l_rec)
                    l_reg_0 = torch.mean(l_reg)
                    #l_ss_0 = torch.mean(l_ss) 
                
                optimizer.zero_grad()
        
                _, _, l_rec, l_reg = ROM.compute_loss(zkT, muz, lvz)
                
                l_rec = torch.mean(l_rec)
                #l_ss = torch.mean(l_ss)
                    
                loss = (torch.mean(l_reg)) + l_rec# + l_ss
                    
                loss.backward(retain_graph=True)

            
            l_reg = torch.mean(l_reg)
            l_rec = l_rec.cpu().detach().numpy()
            l_reg = l_reg.cpu().detach().numpy()
            #l_ss = l_ss.cpu().detach().numpy()
            
            l_rec_list[epoch] = l_rec
            l_reg_list[epoch] = l_reg
            #l_ss_list[epoch] = l_ss

        if epoch % print_epochs == 0:
            t_mid = time.time()
            print('l_rec = ', l_rec)
            print('l_reg = ', l_reg)   
            #print('l_ss  = ', l_ss)
            print('Estimated time to completion: ', (t_mid-t_start)/((epoch+1)*60)*(epochs-epoch+1), " minutes")
    '''    
    for n, (u, _) in enumerate(train_loader_u):
        if n == 0:
            true_params = true_params.to(device)
            true_data = true_data.to(device)
            zmu, _, z, out_test, _ = VAE.forward(true_data)
            disentanglement_score = VAE.compute_dis_score(true_params, z)
            print(disentanglement_score)
    ''' 
    disentanglement_score=0.
    # we want to save the initialized losses also
    l_rec_list = np.insert(l_rec_list, 0, l_rec_0.cpu().detach().numpy())
    l_reg_list = np.insert(l_reg_list, 0, l_reg_0.cpu().detach().numpy())
    #l_ss_list = np.insert(l_ss_list, 0, l_ss_0.cpu().detach().numpy())
     
    #save model
    config = {'train_data_dir_u': train_data_dir_u,
              #'train_data_dir_l': train_data_dir_l,
              'test_data_dir': test_data_dir,
              'model': 'rom',
              'n_latent': n_latent,
              'activations': activations,
              'dense_blocks': dense_blocks,
              'initial_features': initial_features,
              'growth_rate': growth_rate,
              'batch_size_u': batch_size_u,
              'nt': nt,
              'tau_lookback': tau_lookback,
              'test_batch_size': test_batch_size,
              'optimizer': optimizer,
              'epochs': epochs,
              'dis_score': disentanglement_score,
              'l_reg': l_reg_list,
              'l_rec': l_rec_list,
              #'l_ss': l_ss_list,
              'model_state_dict': ROM.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(), 
              'weight_decay': wd
              }
    
    torch.save(config, save_dir + '/' + filename)
