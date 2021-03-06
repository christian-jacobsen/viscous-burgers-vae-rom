"""
@author: Christian Jacobsen, University of Michigan

VAE configuration file

"""

import numpy as np
import torch
import torch.nn as nn


def lr_schedule_0(epoch):
    # used for VAE w/out HP network
    e0 = 1500
    e1 = 3500
    e2 = 6000
    e3 = 8500
    if epoch < e0:
        return 0.0005
    elif epoch < e1:
        return 0.0001
    elif epoch < e2:
        return 0.00005
    elif epoch < e3:
        return 0.00001
    else:
        return 0.000005
    
def lr_schedule_1(epoch):
    # used for VAE w/ HP network (more sensitive to learning rate)
    e0 = 6499
    e1 = 12500
    if epoch <= e0:
        return 0.0003
    elif epoch <= e1:
        return 0.00005
    else:
        return 0.000025


# dataset and save paths ----------------------------------------------------------------------------------------------
n_latent = 32              # latent dimension
n_ic = 1                 # number of initial conditions (for dataset)
nt = 100                 # number of time snapshots to train on
#arch = 'dilated-denseblock'  # architecture specifying 'encoder-decoder' type (not working)

train_data_dir_u = 'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)   # training data directory
train_data_dir_l = 'data/DarcyFlow/multimodal/kle2_mc512_bimodal_2.hdf5'     # testing data directory
test_data_dir = 'data/Burgers1D/burgers1d_ic_{}.hdf5'.format(n_ic)     # testing data directory

save_dir = './Burgers1D/ic_{}/n{}'.format(n_ic,n_latent) # specify a folder where all similar models belong. 
                                     #    after training, model and configuration will be saved in a subdirectory as a .pth file
continue_training = False           # specify if training is continued from a saved model
tr = 4
vae_path = './Burgers1D/ic_{}/n{}/AE_{}/AE_{}.pth'.format(n_ic,n_latent,tr,tr)       # the path to the previously saved model                
continue_path = vae_path
save_interval = None
# architecture parameters ---------------------------------------------------------------------------------------------

HP = False                 # include heirarchical prior network?

act = nn.GELU()#inplace=True)
dense_blocks = [4, 6, 4]    # vector containing dense blocks and their length
growth_rate = 4             # see dense architecture for detailed explantation. dense block growth rate
data_channels = 3           # number of input channels
initial_features = 2        # see dense architecture for explanation. Features after initial convolution


if HP:
    prior = 'N/A'           # no need for prior specification w/ HP
    full_param = False      # specifies if the prior network variances are constant (False) or parameterized by NNs (True)
else:
    prior = 'scaled_gaussian'      # specify the prior:
                            #   'std_norm' = standard normal prior (isotropic gaussian)
                            #   'scaled_gaussian' = Factorized Gaussian prior centered at origin.
omega = 0.#40*np.pi/180           # rotation angle of latent space
# training parameters --------------------------------------------------------------------------------------------------

tau_lookback = 2 #lookback window to train ROM on   

wd = 1e-7                     # weight decay (Adam optimizer)
batch_size_u = 512             # batch size (training)
batch_size_l = 512
test_batch_size = 1       # not used during training, but saved for post-processing
beta0 = 0.000000001         # \beta during reconstruction-only phase

nu = 0.005
tau = 1                     # these are parameters for the beta scheduler, more details in paper

if HP:                      # specify the learning rate schedule
    lr_schedule = lr_schedule_1
    epochs = 200#14000
    rec_epochs = 100#6500
    if full_param:
        beta_p0 = beta0
        
else:
    lr_schedule = lr_schedule_0
    epochs = 10000 # 6500
    rec_epochs = 10000# 4000







