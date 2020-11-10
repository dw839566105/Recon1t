#!/usr/bin/env python
# coding: utf-8

# In[6]:


import uproot
import tables
import numpy as np
import matplotlib.pyplot as plt


# In[20]:


f = uproot.open('/mnt/stage/jinp/Electron-0.root')
x_truth = []
y_truth = []
z_truth = []

for a, b, c in zip(f['SimTriggerInfo']['truthList.x'].array(),f['SimTriggerInfo']['truthList.y'].array(),f['SimTriggerInfo']['truthList.z'].array()):
    x_truth.append(a)
    y_truth.append(b)
    z_truth.append(c)

x_truth = np.array(x_truth) 
y_truth = np.array(y_truth) 
z_truth = np.array(z_truth)


# In[21]:


h = tables.open_file('wav1.h5')

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

data = np.zeros((np.size(x1),3))

index = L1 < L2
data[index,0] = x1[index]
data[index,1] = y1[index]
data[index,2] = z1[index]

data[~index,0] = x2[~index]
data[~index,1] = y2[~index]
data[~index,2] = z2[~index]

xt = recondata[:]['x_truth'][0]
yt = recondata[:]['y_truth'][0]
zt = recondata[:]['z_truth'][0]

x = data[(s1 * s2)!=0,0]
y = data[(s1 * s2)!=0,1]
z = data[(s1 * s2)!=0,2]

from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('jet', 256)
newcolors = viridis(np.linspace(0, 1, 256))
pink = np.array([1, 1, 1, 1])
newcolors[:25, :] = pink
newcmp = ListedColormap(newcolors)

r = np.sqrt(x**2 + y**2 + z**2)
index = (r<0.64) & (r>0.01) & (~np.isnan(r))
H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
X, Y = np.meshgrid(xedges[1:],yedges[1:])
plt.figure(dpi=200)
plt.contourf(X,Y,np.log(np.transpose(H1)+1), cmap=newcmp)
plt.colorbar()
plt.xlabel(r'$x^2 + y^2/m^2$')
plt.ylabel('$z$/m')


# In[23]:


h = tables.open_file('wav1.h5')

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

data = np.zeros((np.size(x1),3))

index = L1 < L2
data[index,0] = x1[index]
data[index,1] = y1[index]
data[index,2] = z1[index]

data[~index,0] = x2[~index]
data[~index,1] = y2[~index]
data[~index,2] = z2[~index]

xt = x_truth
yt = y_truth
zt = z_truth

print(x_truth.shape)
print(data.shape)
x = data[(s1 * s2)!=0,0]
y = data[(s1 * s2)!=0,1]
z = data[(s1 * s2)!=0,2]


# In[47]:


r_truth = (np.sqrt(x_truth**2 + y_truth**2 + z_truth**2)/1000)[:,0]
r_recon = np.linalg.norm(data[:,0:2], axis=1)
plt.figure(figsize=(12,10))
plt.hist2d(r_truth, r_recon, bins=30, cmap=newcmp)
plt.xlabel('truth R/m')
plt.ylabel('recon R/m')

plt.colorbar(cmap=newcmp)
plt.show()


# In[27]:


r_truth.shape


# In[28]:


r_recon.shape


# In[ ]:




