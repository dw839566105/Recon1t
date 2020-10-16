#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
import os, sys
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import curve_fit


# In[16]:


def LoadDataPE(path, radius, order, axis):
    order = order-1
    data = np.zeros((order, order))
    crv  = np.zeros((order))
    filename = path + 'file_+' + radius + '.h5'
    print(filename)
    h = tables.open_file(filename,'r')
    for i in np.arange(2,order):
        coeff = 'coeff' + str(i+1)
        aic = 'AIC' + str(i+1)
        COEFF = eval('h.root.'+coeff+'[()]')
        AIC = eval('h.root.'+aic+'[()]')
        data[i,0:i+1] = COEFF
        crv[i] = AIC
    h.close()
    return data, crv

path = '../coeff_pe_1t_reflection0.00_30/'
#ra = np.hstack((np.arange(0.01, 0.40, 0.01), np.arange(0.40, 0.63, 0.002)))
ra = 0.010
order = 30
str_radius = '%.3f' % ra

data, crv = LoadDataPE(path, str_radius, order, axis)
print(data.shape)
start = 4
plt.figure(figsize=(20,10))
plt.plot(data.T)
plt.xlabel('order',fontsize=20)
plt.ylabel('AIC',fontsize=20)
plt.title('%.3f m y axis ' % ra,fontsize=20)
plt.show()


# In[ ]:


def LoadDataPE(path, radius, order, axis):
    order = order-1
    data = np.zeros((order, order))
    crv  = np.zeros((order))
    filename = path + 'file_' + radius + '-' + axis + '.h5'
    print(filename)
    h = tables.open_file(filename,'r')
    for i in np.arange(order):
        coeff = 'coeff' + str(i+1)
        aic = 'AIC' + str(i+1)
        COEFF = eval('h.root.'+coeff+'[()]')
        AIC = eval('h.root.'+aic+'[()]')
        data[i,0:i+1] = COEFF
        crv[i] = AIC
    h.close()
    return data, crv

def plot(ra, axis):
    path = '../coeff_pe_1t_2.0MeV_dns_aic_sgl/'
    #ra = np.hstack((np.arange(0.01, 0.40, 0.01), np.arange(0.40, 0.63, 0.002)))
    order = 25
    str_radius = '%.3f' % ra

    data, crv = LoadDataPE(path, str_radius, order, axis)
    start = 4
    plt.figure(figsize=(20,10))
    plt.plot(np.arange(start,24,1)+1,crv[start:],            label = 'min order = %d' % ((np.where(crv==np.min(crv)))[0][0] + 1))
    plt.xlabel('order',fontsize=20)
    plt.ylabel('AIC',fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('%.3f m %s axis ' %(ra,axis),fontsize=20)
    plt.legend(fontsize=20)
    plt.savefig('%sAIC%.3f.pdf' % (axis, ra))
    plt.show()

for i in np.arange(0.01,0.65,0.01):
    try:
        plot(i,'x')
    except:
        pass
for i in np.arange(0,0.65,0.01):    
    try:
        plot(i,'y')
    except:
        pass
for i in np.arange(0,0.65,0.01):    
    try:
        plot(i,'z')
    except:
        pass


# In[5]:


import h5py
f=h5py.File('../coeff_pe_1t_8.0MeV_shell_90/file_+0.300.h5',"r")
a = f['coeff25'].value
print(a)


# In[1]:


import tables
path = '../coeff_pe_1t_reflection0.00_30/'
radiuss = np.arange(0.01,0.65,0.01)
for index, radius in enumerate(radiuss):
    h = tables.open_file(path+'file_+%.3f.h5' % radius)
    data = []
    orders = np.arange(2,90)
    for order in orders:
        data.append(eval('h.root.AIC%d[()]'% order))
    h.close()
    plt.figure(index+1, dpi=150)
    plt.plot(orders, data - np.min(data) + 1)
    plt.semilogy()
    plt.xlabel('order')
    plt.ylabel('AIC value')
    plt.title('radius: %.3f' % radius)
    plt.savefig('fig/radius_%.3f.png' % radius)
    plt.show()


# In[39]:


import ROOT
h = ROOT.TFile('/mnt/stage/douwei/Simulation/1t_root/8.0MeV_shell/1t_+0.500_1.root')
h.SimTriggerInfo


# In[40]:


h = tables.open_file('../coeff_pe_1t_2.0MeV_dns_aic_total1/file_0.010.h5')
print(h.root.AIC)


# In[42]:


import numpy as np
import statsmodels.api as sm
from scipy import stats
print(sm.GLM.__doc__)


# In[10]:


import tables
path = '../coeff_pe_1t_reflection0.03_30/'
radiuss = np.arange(0.01,0.65,0.01)
for index, radius in enumerate(radiuss):
    h = tables.open_file(path+'file_+%.3f.h5' % radius)
    data = []
    orders = np.arange(2,30)
    for order in orders:
        data.append(eval('h.root.AIC%d[()]'% order))
    h.close()
    plt.figure(index+1, dpi=150)
    plt.plot(orders, data - np.min(data) + 1)
    plt.semilogy()
    plt.xlabel('order')
    plt.ylabel('AIC value')
    plt.title('radius: %.3f' % radius)
    #plt.savefig('fig/radius_%.3f.png' % radius)
    plt.show()


# In[ ]:




