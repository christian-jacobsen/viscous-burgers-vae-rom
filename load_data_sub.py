import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data_sub(u, batch_size, shuff = True):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

        
    

    kwargs = {'num_workers': 0,
              'pin_memory': True} if torch.cuda.is_available() else {}
    nx = np.shape(u)[-1] 
    u = torch.reshape(u, (-1,1,nx))

    dataset = TensorDataset(u,u)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

    # simple statistics of output data
    u_data_mean = torch.mean(u, 0)
    u_data_var = torch.sum((u - u_data_mean) ** 2)
    stats = {}
    stats['u_mean'] = u_data_mean
    stats['u_var'] = u_data_var

    return data_loader, stats
