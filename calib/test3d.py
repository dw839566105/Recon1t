#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
sys.path.append('./PE_calib')
import pub
import numpy as np

import tables, h5py
import argparse
from argparse import RawTextHelpFormatter
import argparse, textwrap
import time


# In[3]:


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = np.array(((0,0,1),))


# In[134]:


PMTNo = np.size(PMTPos[:,0])
L, K = 1000, 1000
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)


# In[135]:


VertexTruth = np.vstack((xv.flatten(), np.zeros_like(xv.flatten()), yv.flatten())).T


# In[136]:


PMTPosRep = np.tile(PMTPos, (VertexTruth[:,0].shape[0],1))
vertex = np.repeat(VertexTruth, PMTNo, axis=0)


# In[137]:


order = 5
from zernike import RZern
cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=False)
cart = RZern(order)
nk = cart.nk
m = cart.mtab
n = cart.ntab
rho = np.linalg.norm(vertex, axis=1)
theta = np.arccos(cos_theta)        
X = np.zeros((rho.shape[0], nk))
for i in np.arange(nk):
    if not i % 5:
        print(f'process {i}-th event')
    X[:,i] = cart.Zk(i, rho, theta)
X = X[:,m>=0]
print(f'rank: {np.linalg.matrix_rank(X)}')
print(X.shape)

binNo = 200
bins=np.linspace(-0.9,0.4,binNo)
EventID = L*K
N = 10
# Legendre coeff
x = pub.legval(bins, np.eye(N).reshape(N, N, 1))
# 1st basis
Y = np.tile(x, L*K).T
# 2nd basis
X = np.repeat(X, bins.shape[0], axis=0)
print(X.shape, Y.shape)
basis = np.zeros((X.shape[0], X.shape[1]*Y.shape[1]))
for i_index, i in enumerate(np.arange(X.shape[1])):
    for j_index, j in enumerate(np.arange(Y.shape[1])):
        total_index = i_index*Y.shape[1] + j_index
        if not total_index % 10:
            print(total_index)
        basis[:, total_index] = X[:,i_index]*Y[:,j_index]
X = basis


# In[138]:


h = tables.open_file('test3.h5')
coef_ = h.root.coeff5[:]
print(coef_.shape)


# In[139]:


A = np.dot(coef_, np.atleast_2d(X).T)
A_test = np.sum(np.exp(A).reshape(-1,binNo), axis=1)
print(A_test.shape)


# In[169]:


AA = np.exp(A)

AA = AA.reshape(-1,binNo)

plt.subplots(2,5, dpi=300)
for i_index, i in enumerate(np.arange(0,AA.shape[1],20)):
    plt.subplot(2,5,i_index+1)
    tmp = AA[:,i]
    tmp[np.linalg.norm(VertexTruth, axis=1)>0.92] = np.nan
    tmp = tmp.reshape(L,K)
    plt.imshow(tmp, origin='lower', extent=(-1,1,-1,1))
    plt.colorbar()
plt.show()


# In[173]:


AA = np.exp(A)

AA = AA.reshape(-1,binNo)

for i_index, i in enumerate(np.arange(0,AA.shape[1],20)):
    plt.figure(dpi=150)
    tmp = AA[:,i]
    tmp[np.linalg.norm(VertexTruth, axis=1)>0.92] = np.nan
    tmp = tmp.reshape(L,K)
    plt.imshow(tmp, origin='lower', extent=(-1,1,-1,1))
    plt.title(f'time={i}ns')
    plt.colorbar()
    plt.show()


# In[174]:


AA = np.exp(A)

AA = AA.reshape(-1,binNo)

for i_index, i in enumerate(np.arange(0,AA.shape[1],20)):
    plt.figure(dpi=150)
    tmp = AA[:,i]
    tmp[np.linalg.norm(VertexTruth, axis=1)>0.92] = np.nan
    tmp = tmp.reshape(L,K)
    plt.imshow(np.log(tmp), origin='lower', extent=(-1,1,-1,1))
    plt.title(f'time={i}ns')
    plt.colorbar()
    plt.show()


# In[150]:


import matplotlib.pyplot as plt
A_test[np.linalg.norm(VertexTruth, axis=1)>0.92] = np.nan
plt.figure(dpi=100)
plt.imshow(np.log(A_test).reshape(L,K), origin='lower', extent=(-1,1,-1,1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE (log scle)')
plt.colorbar()
plt.figure(dpi=100)
plt.imshow(A_test.reshape(L,K), origin='lower', extent=(-1,1,-1,1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE')
plt.colorbar()


# In[163]:


AA.shape


# In[96]:


plt.plot(np.exp(A).reshape(-1,binNo)[L//2])
plt.plot(np.exp(A).reshape(-1,binNo)[L*K//2])
plt.plot(np.exp(A).reshape(-1,binNo)[L*K//1 - L//2])


# In[157]:


plt.figure(dpi=150)
A1 = np.exp(A).reshape(-1,binNo)[L*K//1 - L//2]
plt.plot(bins*376/2+376/2, A1/np.sum(A1), label='nearest (0,0,1)' )
A1 = np.exp(A).reshape(-1,binNo)[L*K//2]
plt.plot(bins*376/2+376/2, A1/np.sum(A1), label='center (0,0,0)')
A1 = np.exp(A).reshape(-1,binNo)[L//2]
plt.plot(bins*376/2+376/2, A1/np.sum(A1), label='farthest (0,0,-1)')
plt.xlabel('Time/ns')
plt.ylabel('Relative flux (arbitrary unit)')
plt.legend()


# In[142]:


t = bins
A_new = np.exp(A).reshape(-1,binNo)
A_time = np.zeros(L*K)
for i in np.arange(L*K):
    tmp = np.cumsum(A_new[i])/np.sum(A_new[i])
    A_time[i] = (t[np.where(np.abs(tmp - 0.1) == np.min(np.abs(tmp - 0.1)))[0][0]])
A_time[np.linalg.norm(VertexTruth, axis=1)>0.9] = np.nan


# In[145]:


plt.figure(dpi=100)
plt.imshow(A_time.reshape(L,K)*368/2 + 368/2, origin='lower', extent=(-1,1,-1,1))
plt.xlabel('relative radius of x')
plt.ylabel('relative radius of z')
plt.colorbar()


# In[175]:


np.pi*0.645**3*4/3*0.


# In[176]:


100**(1/3)*0.645


# In[178]:


4*np.pi*3**2/(np.pi*0.25**2)


# In[215]:


1.3/2.0/1e8*1e9


# In[123]:


import numpy as np
import tables, h5py
from zernike import RZern
import tables
h = tables.open_file('/home/douwei/Recon1t/calib/Zernike_50_dual.h5')

cart = RZern(50)
nk = cart.nk
m = cart.mtab
n = cart.ntab

coef_ = np.zeros(nk)
coef_[m>=0] = h.root.coeff50[:]
h.close()
import matplotlib.pyplot as plt
L, K = 1000, 1000
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
# normal scale
# im = plt.imshow(np.exp(cart.eval_grid(np.array(coef_), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
# log scale
sums = cart.eval_grid(np.array(coef_), matrix=True)
r = np.sqrt(xv**2 + yv**2)
sums[r>0.98] = np.nan
im = plt.imshow(sums, origin='lower', extent=(-1, 1, -1, 1))
plt.colorbar()


# In[129]:


plt.figure(dpi=200)
im = plt.imshow(sums.T, origin='lower', extent=(-1, 1, -1, 1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE fit by Zernike(log)')
plt.colorbar()
plt.savefig('Zernike2d_log.png')
plt.show()
plt.figure(dpi=200)
im = plt.imshow(np.exp(sums).T, origin='lower', extent=(-1, 1, -1, 1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE fit by Zernike')
plt.colorbar()
plt.savefig('Zernike2d.png')
plt.show()


# In[146]:


import numpy as np
import tables, h5py
from zernike import RZern
import tables
h = tables.open_file('/home/douwei/Recon1t/calib/Zernike_70.h5')

cart = RZern(70)
nk = cart.nk
m = cart.mtab
n = cart.ntab

coef_ = np.zeros(nk)
coef_[m>=0] = h.root.coeff70[:]
h.close()
import matplotlib.pyplot as plt
L, K = 1000, 1000
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
cart.make_cart_grid(xv, yv)
# normal scale
# im = plt.imshow(np.exp(cart.eval_grid(np.array(coef_), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
# log scale
sums = cart.eval_grid(np.array(coef_), matrix=True)
r = np.sqrt(xv**2 + yv**2)
sums[r>0.98] = np.nan
im = plt.imshow(sums, origin='lower', extent=(-1, 1, -1, 1))
plt.colorbar()


# In[147]:


plt.figure(dpi=200)
im = plt.imshow(sums.T, origin='lower', extent=(-1, 1, -1, 1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE fit by Zernike(log)')
plt.colorbar()
plt.savefig('Zernike2d_log.png')
plt.show()
plt.figure(dpi=200)
im = plt.imshow(np.exp(sums).T, origin='lower', extent=(-1, 1, -1, 1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE fit by Zernike')
plt.colorbar()
plt.savefig('Zernike2d.png')
plt.show()


# In[43]:


np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = np.array(((0,0,1),))

PMTNo = np.size(PMTPos[:,0])
L, K = 500, 500
ddx = np.linspace(-1.0, 1.0, K)
ddy = np.linspace(-1.0, 1.0, L)
xv, yv = np.meshgrid(ddx, ddy)
VertexTruth = np.vstack((xv.flatten(), np.zeros_like(xv.flatten()), yv.flatten())).T
PMTPosRep = np.tile(PMTPos, (VertexTruth[:,0].shape[0],1))
vertex = np.repeat(VertexTruth, PMTNo, axis=0)

order = 5
from zernike import RZern
cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=False)
cart = RZern(order)
nk = cart.nk
m = cart.mtab
n = cart.ntab
rho = np.linalg.norm(vertex, axis=1)
theta = np.arccos(cos_theta)        
X = np.zeros((rho.shape[0], nk))
for i in np.arange(nk):
    if not i % 5:
        print(f'process {i}-th event')
    X[:,i] = cart.Zk(i, rho, theta)
X = X[:,m>=0]
print(f'rank: {np.linalg.matrix_rank(X)}')
print(X.shape)

binNo = 100
bins=np.linspace(-0.9,0,binNo)
EventID = L*K
N = 10
# Legendre coeff
x = pub.legval(bins, np.eye(N).reshape(N, N, 1))
# 1st basis
Y = np.tile(x, L*K).T
# 2nd basis
X = np.repeat(X, bins.shape[0], axis=0)
print(X.shape, Y.shape)
basis = np.zeros((X.shape[0], X.shape[1]*Y.shape[1]))
for i_index, i in enumerate(np.arange(X.shape[1])):
    for j_index, j in enumerate(np.arange(Y.shape[1])):
        total_index = i_index*Y.shape[1] + j_index
        if not total_index % 10:
            print(total_index)
        basis[:, total_index] = X[:,i_index]*Y[:,j_index]
X = basis


# In[45]:


h = tables.open_file('test5.h5')
coef_ = h.root.coeff5[:]
print(coef_.shape)
A = np.dot(coef_, np.atleast_2d(X).T)
A_test = np.sum(np.exp(A).reshape(-1,binNo), axis=1)
print(A_test.shape)


# In[52]:


import matplotlib.pyplot as plt
A_test[np.linalg.norm(VertexTruth, axis=1)>0.6] = np.nan
plt.figure(dpi=100)
plt.imshow(A_test.reshape(L,K), origin='lower', extent=(-1,1,-1,1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE (log scle)')
plt.colorbar()
plt.figure(dpi=100)
plt.imshow(A_test.reshape(L,K), origin='lower', extent=(-1,1,-1,1))
plt.xlabel('x/m')
plt.ylabel('z/m')
plt.title('PE')
plt.colorbar()


# In[57]:


A.reshape(L,K)


# In[18]:


aaa =np.exp(A).reshape(-1,binNo)


# In[46]:


A.reshape(-1,binNo)[125250]


# In[39]:


h = tables.open_file('test5.h5')
coef_ = h.root.coeff5[:]
print(coef_.shape)
A = np.dot(coef_, np.atleast_2d(X).T)
A_test = np.sum(np.exp(A).reshape(-1,binNo), axis=1)
print(A_test.shape)


# In[58]:


coef_.shape


# In[ ]:




