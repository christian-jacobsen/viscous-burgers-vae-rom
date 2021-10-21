import torch
import numpy as np
import h5py
from torch.utils.data import DataLoader, TensorDataset


def load_data_rom(zmu, zlv, tau, batch_size, shuff = True):
    """Return data loader

    Args:
        data_dir: directory to hdf5 file, e.g. `dir/to/kle4225_lhs256.hdf5`
        batch_size (int): mini-batch size for loading data

    Returns:
        (data_loader (torch.utils.data.DataLoader), stats)
    """

        
    

    kwargs = {'num_workers': 0,
              'pin_memory': True} if torch.cuda.is_available() else {}
    z = zmu
    nz = np.shape(z)[-1] 
    z = torch.reshape(z, (-1,1,nz))
    ztau = torch.zeros((np.shape(z)[0]-tau, tau, nz))
    for i in range(np.shape(z)[0]-tau):
        ztau[i, :, :] = z[i:i+tau, :, :].reshape((1, tau, nz))
    z = z[tau:, 0, :] 

    zl = zlv
    nz = np.shape(zl)[-1] 
    zl = torch.reshape(zl, (-1,1,nz))
    ztaul = torch.zeros((np.shape(zl)[0]-tau, tau, nz))
    for i in range(np.shape(zl)[0]-tau):
        ztaul[i, :, :] = zl[i:i+tau, :, :].reshape((1, tau, nz)) 
    zl = zl[tau:, 0, :] 

    dataset = TensorDataset(ztau,ztaul,z,zl)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuff, **kwargs)

    # simple statistics of output data
    #u_data_mean = torch.mean(u, 0)
    #u_data_var = torch.sum((u - u_data_mean) ** 2)
    stats = {}
    stats['u_mean'] = 0#u_data_mean
    stats['u_var'] = 0#u_data_var

    return data_loader, stats
