import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data_new(data_dir, batch_size, nt, shuff = True):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

    with h5py.File(data_dir, 'r') as f:
        u_data = f['u'][()]
        t_data = f['t'][()]
        phi_data = f['phi'][()] 
    print("========== LOADING DATA ==========")    
    print("solution u(t,x) data shape: {}".format(u_data.shape))
    print("time t data shape: {}".format(t_data.shape))
    print("phi data shape: {}".format(phi_data.shape))
    
    print("Loading first ",nt," time snapshots!") 

    kwargs = {'num_workers': 0,
              'pin_memory': True} if torch.cuda.is_available() else {}
    if (batch_size=='all'):
        batch_size = np.shape(u_data)[0]
        
    dataset = TensorDataset(torch.tensor(u_data[:,:,0:nt,:]), torch.tensor(t_data[:,0:nt]), torch.tensor(phi_data))
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

    # simple statistics of output data
    u_data_mean = np.mean(u_data, 0)
    u_data_var = np.sum((u_data - u_data_mean) ** 2)
    stats = {}
    stats['u_mean'] = u_data_mean
    stats['u_var'] = u_data_var
    
    print("======== LOADING FINISHED ========")

    return data_loader, stats
