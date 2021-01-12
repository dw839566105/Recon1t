#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import tables
from scipy.spatial.distance import pdist, squareform
import os


# In[54]:


x = np.zeros(0)
y = np.zeros(0)
z = np.zeros(0)
E = np.zeros(0)

for i,file in enumerate(np.arange(1,500)):
    h = tables.open_file('../result_1t_point_10_Recon_1t_new2_new/tri10_1e6_%d.h5' % file,'r')
    recondata = h.root.Recon
    E1 = recondata[:]['E_sph_in']
    x1 = recondata[:]['x_sph_in']
    y1 = recondata[:]['y_sph_in']
    z1 = recondata[:]['z_sph_in']
    L1 = recondata[:]['Likelihood_in']
    s1 = recondata[:]['success_in']

    E2 = recondata[:]['E_sph_out']
    x2 = recondata[:]['x_sph_out']
    y2 = recondata[:]['y_sph_out']
    z2 = recondata[:]['z_sph_out']
    L2 = recondata[:]['Likelihood_out']
    s2 = recondata[:]['success_out']

    data = np.zeros((np.size(x1),4))

    index = L1 < L2
    data[index,0] = x1[index]
    data[index,1] = y1[index]
    data[index,2] = z1[index]
    data[index,3] = E1[index]

    data[~index,0] = x2[~index]
    data[~index,1] = y2[~index]
    data[~index,2] = z2[~index]
    data[~index,3] = E2[~index]
    
    x = np.hstack((x, data[(s1 * s2)!=0,0]))
    y = np.hstack((y, data[(s1 * s2)!=0,1]))
    z = np.hstack((z, data[(s1 * s2)!=0,2]))
    E = np.hstack((E, data[(s1 * s2)!=0,3]))


# In[69]:


plt.figure(dpi=300)
plt.hist2d(x**2+y**2, z[z<0.65], bins=30)
plt.xlabel(r'$x^2+y^2/m^2$')
plt.ylabel(r'$z$/m')
plt.savefig('K40_recon.png')
plt.show()


# In[61]:


r = np.sqrt(x**2+y**2+z**2)
index = r!=0
plt.figure(dpi=300)
plt.hist2d(np.arctan2(x, y)[index], z[index]/r[index], bins=(np.linspace(-np.pi, np.pi, 50), np.linspace(-1,1,50)))
plt.xlabel(r'$\theta$/m')
plt.ylabel(r'$\phi$/m')
plt.savefig('K40_tp.png')
plt.show()


# In[68]:


plt.figure(dpi=300)
plt.hist(np.sqrt(x**2+y**2+z**2)**3, bins=100)
plt.xlabel('$R^3/m^3$')
plt.savefig('ReconRadius.png')
plt.show()


# In[63]:


r = np.sqrt(x**2+y**2+z**2)
plt.figure(dpi=300)
plt.hist(E, bins=np.linspace(0.1,2.5,30))
plt.xlabel('Energy/MeV')
plt.savefig('E.png')
plt.show()

plt.figure(dpi=300)
plt.hist(E[r<0.55], bins=np.linspace(0.1,2.5,30))
plt.xlabel('Energy/MeV')
plt.savefig('E_cut.png')
plt.show()


# In[59]:


plt.hist(z,bins=50)
plt.show()


# In[ ]:




