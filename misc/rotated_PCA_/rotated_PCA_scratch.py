# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 19:59:11 2018

@author: zmoon
"""

import matplotlib.pyplot as plt
import numpy as np
import numpy.linalg as nla
import scipy.linalg as sla
from scipy.io import savemat
#import xarray as xr

plt.style.use('ggplot')
plt.close('all')


#%% create data

#> signal will be a sum of 3 waves
wls = np.array([1000, 700, 400])  # wavelength (in x direction)
ks = 2*np.pi/wls  # wavenumber in x direction
C = 10  # speed of mean wind advecting the waves
cs = np.array([10, 50, 100])  # wave speed
ws = 2*np.pi/cs  # wave angular frequency omega
As = np.array([10, 5, 2])  # amplitudes
A_noise = 0.5

def clean_sig(t, x):
    """Compute sum of the wave signals with their properties listed above
    only properly broadcasts for float x"""
    return ( As*np.cos(ks*x - ws*t) ).sum()
clean_sig = np.vectorize(clean_sig)

N = 200
M = 50
x = np.linspace(0, 1000, M)
t = np.linspace(0, 10000, N)

X = np.zeros((N, M))  # pre-allocate data matrix
for i, t_i in enumerate(t):
    X[i,:] = clean_sig(t_i, x) + np.random.normal(loc=0, scale=A_noise, size=x.size)


#> plot some of the time steps
i_plots = [0, 3, 6]
f1, aa = plt.subplots(len(i_plots), 1, num='time series',
                      figsize=(6, 4.5), sharex=True, sharey=True)
for i, i_plot in enumerate(i_plots):
    ax = aa[i]
    l = 'i = {i:d}, t = {t:.1f}'.format(i=i_plot, t=t[i])
    ax.plot(x, X[i_plot,:], '.-', lw=1, label=l)
    ax.legend(loc='lower left')

ax.set_xlabel('x')
#ax.set_ylabel('t')
f1.tight_layout();

#> save the data as .mat for checking results in Matlab
savemat('rotated_PCA_input_data.mat', {'X': X})


#%% PCA via SVD

mu_X = X.mean(axis=0)
Xp = X - mu_X  # X'

#> SVD
U, s, V_T = sla.svd(Xp, full_matrices=False)
S = sla.diagsvd(s, *X.shape)
V = V_T.T

#> calculate PCA stuff from the SVD results
PCs = X @ V  # principal components (columns)
normalized_PCs1 = np.sqrt(N-1) * U
PDs = V      # eigenvectors of centered data covariance matrix; principal directions (cols)
eigvals = s**2/(N-1)
eigvals0 = eigvals.copy()
normalized_PCs2 = (PCs - PCs.mean(axis=0))/np.sqrt(eigvals)

#loadings = np.sqrt(eigs) * PDs  # loadings (cols)
#loadings = np.sqrt(eigs)[np.newaxis,:] * PDs
#loadings = np.diag(np.sqrt(eigs)) @ PDs  # incorrect!
loadings = PDs @ np.diag(np.sqrt(eigvals))

var_expl = eigvals/eigvals.sum()

#> some checks
#assert(np.allclose(PCs.T @ PCs, np.eye(N)))
assert(np.allclose(PDs.T @ PDs, np.eye(M)))  # principal directions are orthog
assert(np.allclose(loadings.T @ loadings, np.eye(M)*eigvals))  # loadings too, though the matrix is not !
assert(np.allclose(sla.norm(loadings, axis=0), np.sqrt(eigvals)))  # check loading vectors are scaled correctly
assert(np.isclose(np.trace(loadings.T @ loadings), eigvals.sum()))
assert(np.allclose(np.corrcoef(PCs.T), np.eye(M)))  # PCs are uncorrelated
#assert(np.isclose(np.trace(normalized_PCs2.T @ normalized_PCs2), np))
assert(np.allclose(normalized_PCs1, normalized_PCs2))

#> plot
n_retain = 6

colors = plt.cm.Dark2(np.linspace(0, 1, 8))[:8]
leg_fs = 9

f2, [a1, a2, a3] = plt.subplots(3, 1, figsize=(6, 8), num='PCs')

a3.plot(x, X[0,:], '.-', lw=1.5, c='0.2', alpha=0.8, label='true data')

for j in range(n_retain-1, 0-1, -1):  # plot PCs in reverse order
    s = '{:d}: {:.1f}% var expl.'.format(j+1, var_expl[j]*100)
    pc_j = PCs[:,j]
    alpha = 0.2 + 0.6*(n_retain-1-j)/(n_retain-1) if n_retain>4 else 0.8
    #print(pc_j.mean())
    a1.plot(t, (pc_j - pc_j.mean())/np.sqrt(eigvals[j]), 
           c=colors[j], lw=1.2, alpha=alpha, label=s)

    a2.plot(x, PDs[:,j], c=colors[j], lw=1.2, alpha=alpha)
    
    s = 'PC 1-{:d}'.format(j+1)
#    X_hat = PCs[:,:j] @ PDs[:j,:] + mu_X
    X_hat = PCs[:,:j] @ PDs[:,:j].T + mu_X
#    a3.plot(t, nla.norm(X-X_hat, axis=1), c=colors[j], lw=1.2, alpha=alpha)
    a3.plot(x, X_hat[0,:], c=colors[j], lw=1.2, alpha=1-alpha, label=s)
    

#for j in range(M):
#    X_hat = PCs[:,:j] @ PDs[:j,:] + mu_X
#    print('j', nla.norm(X-X_hat))


a1.set_xlabel('t')
a1.set_ylabel('normalized PC')
a2.set_xlabel('x')
a2.set_ylabel('normalized loading\n(principal direction)')
#a3.set_xlabel('t')
a3.set_xlabel('x')
a3.set_ylabel('')
    
ncol = 1 if n_retain <= 4 else 2
a1.legend(loc='upper right', ncol=ncol, fontsize=leg_fs, frameon=True)
a3.legend(loc='upper right', ncol=ncol, fontsize=leg_fs, frameon=True)


f2.tight_layout();