# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 10:51:48 2020

This code solves viscous Burger's Eqn using fourth order finite difference schemes and
the RK4 multi-step method, and the solution is saved.

@author: chris
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags
from mpl_toolkits import mplot3d
import scipy.io as sio
import h5py

plt.close('all')

def F(u,dx):
    n = np.size(u)
    diags1 = [16, -1, -1, 16, -30, 16, -1, -1, 16]
    diags2 = [8, -1, 1, -8, 0, 8, -1, 1, -8]
    mat1 = diags(diags1, [-n+1, -n+2, -2, -1, 0, 1, 2, n-2, n-1], shape = (n,n)).toarray()
    mat2 = diags(diags2, [-n+1, -n+2, -2, -1, 0, 1, 2, n-2, n-1], shape = (n,n)).toarray()
    mat2 = mat2@u
    mat2 = np.multiply(mat2,u)
    f = 0.01/(12*dx*dx)*mat1@u - 1/(12*dx)*mat2 
    return f

def RK4(u0, dx, dt, F, nsteps):
    n = np.size(u0)
    uv = np.zeros((n, nsteps))
    uv[:,0] = u0.reshape((n,))
    tv = np.zeros(nsteps)
    
    for i in range(nsteps-1):
        k1 = F(uv[:,i], dx)
        k2 = F(uv[:,i]+dt*k1/2, dx)
        k3 = F(uv[:,i]+dt*k2/2, dx)
        k4 = F(uv[:,i]+dt*k3, dx)
        uv[:,i+1] = uv[:,i] + dt/6*(k1 + 2*k2 + 2*k3 + k4)
        tv[i+1] = tv[i] + dt

    return uv, tv

def IC(x0, phi):
    n = np.size(x0)
    u0 = -0.5*np.cos(2*np.pi*x0*phi)+0.5
    u0 = np.reshape(u0, (n,1))
    return u0

n_ic = 10

nx = 128
x0 = np.linspace(0,1,nx)
dx = x0[1]-x0[0]
dt = 0.2*dx
nsteps = 1200

#preallocate data storage arrays
Usave = np.zeros((n_ic, 1, nsteps, nx))
Tsave = np.zeros((n_ic, nsteps)) 
phisave = np.zeros((n_ic,))
for i in range(n_ic):
    phi = np.random.rand(1)*2 + 0.5
    u0 = IC(x0, phi)
    U, T= RK4(u0, dx, dt, F, nsteps)
    Usave[i,0,:,:] = np.transpose(U)
    Tsave[i,:] = T
    phisave[i] = phi
'''
plt.figure(1)
plt.plot(x0,U[:,0])
plt.plot(x0,U[:,-1])
plt.savefig("./burgers1d_snapshots.png")
X,TT = np.meshgrid(x0,T)
plt.figure(2)
ax = plt.axes(projection = '3d')
ax.plot_surface(X,TT,np.transpose(U), cmap = 'viridis', alpha = 1, edgecolor = 'none')
plt.xlabel('x')
plt.ylabel('t')
ax.set_title('Viscous Burgers Solution')
ax.set_zlabel('u')
plt.savefig("./burgers1d_surf.png")
plt.figure(3)
plt.imshow(U, cmap='jet')
plt.savefig("./burgers1d_field.png")
'''
f1 = h5py.File("./data/Burgers1D/burgers1d_ic_{}.hdf5".format(n_ic),"w")
f1.create_dataset("u", np.shape(Usave), data=Usave)
f1.create_dataset("t", np.shape(Tsave), data=Tsave)
f1.create_dataset("phi", np.shape(phisave), data=phisave)
f1.close

print("Saved shape of U: ", np.shape(Usave))
print("Saved shape of T: ", np.shape(Tsave))


