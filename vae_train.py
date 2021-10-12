# -*- coding: utf-8 -*-
"""
Created on Fri Mar 26 17:41:10 2021

@author: Christian Jacobsen, University of Michigan
"""

from burgers_rom_vae import *
from load_data_new import load_data_new
from load_data_sub import load_data_sub
import torch
import numpy as np


def vae_load(path):
    config = torch.load(path)
    data_channels = 3
    initial_features = config['initial_features']
    growth_rate = config['growth_rate']
    n_latent = config['n_latent']
    dense_blocks = config['dense_blocks']
    act = config['activations']
    VAE = burgers_rom_vae(data_channels, initial_features, dense_blocks, growth_rate, n_latent, activations = act)
    VAE.load_state_dict(config['model_state_dict'])
    return VAE, config

def vae_train(train_data_dir_u, test_data_dir, save_dir, filename, \
                       epochs, rec_epochs, batch_size_u, test_batch_size, \
                       wd, beta0, lr_schedule, nu, tau, \
                       data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                       prior, activations, cont, cont_path):

    sub_batch_size = 100 
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
       
    train_loader_u, train_stats_u = load_data_new(train_data_dir_u, batch_size_u) # unlabeled data
    
    if cont:
        VAE, config = vae_load(cont_path)
    else:
        VAE = burgers_rom_vae(n_latent, activations)
    VAE = VAE.to(device)
    optimizer = torch.optim.Adam(VAE.parameters(), lr=lr_schedule(0), weight_decay = wd)
    if cont:
        optimizer.load_state_dict(config['optimizer_state_dict'])    
    beta = beta0


    l_rec_list = np.zeros((epochs,))
    l_reg_list = np.zeros((epochs,))
    #l_ss_list = np.zeros((epochs,))
    beta_list = np.zeros((epochs,))
    grad_list = np.zeros((epochs,))
    VAE.train()

    print_epochs = 1 

    for epoch in range(epochs):
        if epoch % print_epochs == 0:
            print('=======================================')
            print('Epoch: ', epoch) 
        optimizer.param_groups[0]['lr'] = lr_schedule(epoch) #update learning rate
        
        for n, (u, _, _) in enumerate(train_loader_u):
            sub_loader, sub_stats = load_data_sub(u, sub_batch_size)
            for m, (us, _) in enumerate(sub_loader):
                us = us.to(device, dtype=torch.float)
                if epoch==0: # compute initialized losses
                    _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(us)
                    l_rec_0 = torch.mean(l_rec)
                    l_reg_0 = torch.mean(l_reg)
                    #l_ss_0 = torch.mean(l_ss) 
                
                optimizer.zero_grad()
        
                _, _, _, _, _, l_rec, l_reg = VAE.compute_loss(us)
                
                l_rec = torch.mean(l_rec)
                #l_ss = torch.mean(l_ss)
                if epoch < rec_epochs:
                    if torch.mean(l_reg) > 1e10:
                        beta = 1
                    else:
                        beta = beta0
                    loss = l_rec# + l_ss
                else:
                    beta = VAE.update_beta(beta, l_rec, nu, tau)
                    if beta > 1:
                        beta = 1
                    
                    loss = beta*(torch.mean(l_reg)) + l_rec# + l_ss
                    
                loss.backward()
                total_norm = 0
                for p in  VAE.parameters():
                    total_norm = p.grad.detach().data.norm(2).item() **2 + total_norm
                total_norm = total_norm ** 0.5

                if total_norm <= 1e8: 
                    optimizer.step()
                else:
                    print('Grad too large!')
            
            
            l_reg = torch.mean(l_reg)
            l_rec = l_rec.cpu().detach().numpy()
            l_reg = l_reg.cpu().detach().numpy()
            #l_ss = l_ss.cpu().detach().numpy()
            
            l_rec_list[epoch] = l_rec
            l_reg_list[epoch] = l_reg
            #l_ss_list[epoch] = l_ss
            beta_list[epoch] = beta
            grad_list[epoch] = total_norm

        if epoch % print_epochs == 0:
            print('beta = ', beta)
            print('l_rec = ', l_rec)
            print('l_reg = ', l_reg)   
            #print('l_ss  = ', l_ss)
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
    beta_list = np.insert(beta_list, 0, beta0)
     
    #save model
    config = {'train_data_dir_u': train_data_dir_u,
              #'train_data_dir_l': train_data_dir_l,
              'test_data_dir': test_data_dir,
              'model': 'vae',
              'n_latent': n_latent,
              'activations': activations,
              'dense_blocks': dense_blocks,
              'initial_features': initial_features,
              'growth_rate': growth_rate,
              'batch_size_u': batch_size_u,
              'test_batch_size': test_batch_size,
              'optimizer': optimizer,
              'prior': prior,
              'beta0': beta0,
              'nu': nu,
              'tau': tau,
              'rec_epochs': rec_epochs,
              'epochs': epochs,
              'dis_score': disentanglement_score,
              'l_reg': l_reg_list,
              'l_rec': l_rec_list,
              #'l_ss': l_ss_list,
              'gradient_list': grad_list,
              'beta_final': beta_list,
              'model_state_dict': VAE.state_dict(),
              'optimizer_state_dict': optimizer.state_dict(), 
              'weight_decay': wd
              }
    
    torch.save(config, save_dir + '/' + filename)
