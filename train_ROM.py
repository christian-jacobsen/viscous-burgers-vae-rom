"""
@author: Christian Jacobsen, University of Michigan

VAE training file: requires a configuration file "config_train.py" to train

"""

from config_train import *
from rom_train import *
import os
import time
import numpy as np



if __name__ == '__main__':
    
    start = time.time()
    save_dir0 = save_dir    
    if not os.path.exists(save_dir):
        os.makedirs(save_dir) 
    n_trials = 1
    for i in range(n_trials): 
        file_exists = True
        trial_num = 0
        while file_exists:
            if trial_num > 100:
                break
            save_dir_temp = save_dir + '/ROM_' + str(trial_num)
            filename = 'ROM_' + str(trial_num) + '.pth'
            path_exists = os.path.exists(save_dir_temp)
            if path_exists:
                file_exists = os.path.exists(save_dir_temp + '/' + filename)
            else:
                file_exists = False
                os.mkdir(save_dir_temp)
            trial_num += 1
        
        save_dir = save_dir_temp
        
        rom_train(train_data_dir_u, test_data_dir, save_dir, filename, \
                                 epochs,  batch_size_u, test_batch_size, nt, tau_lookback, \
                                 wd, lr_schedule, \
                                 data_channels, initial_features, dense_blocks, growth_rate, n_latent, \
                                 prior, act, vae_path)
        save_dir = save_dir0 
    end = time.time()
    
    print('------------ Training Completed --------------')
    print('Elapsed Time: ', (end-start)/60, ' mins')
    print('Save Location: ', save_dir)



