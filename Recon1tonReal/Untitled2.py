#!/usr/bin/env python
# coding: utf-8

# In[1]:


import uproot
import tables
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


def readfile(filename):
    h = tables.open_file(filename)
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

    x = data[(s1 * s2)!=0, 0]
    y = data[(s1 * s2)!=0, 1]
    z = data[(s1 * s2)!=0, 2]
    E = data[(s1 * s2)!=0, 3]
    return E, x, y, z


# In[46]:


import sys, os
path = '/mnt/eternity/Jinping_1ton_Data/Recon/run00000259/'
datanames = os.listdir(path)
E = np.zeros(0)
x = np.zeros(0)
y = np.zeros(0)
z = np.zeros(0)
for filename in datanames:
    if os.path.splitext(filename)[1] == '.h5':#目录下包含.json的文件
        try:
            E0, x0, y0, z0 = readfile(os.path.join(path, filename))
            E = np.append(E, E0)
            x = np.append(x, x0)
            y = np.append(y, y0)
            z = np.append(z, z0)
        except:
            pass


# In[47]:


from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
viridis = cm.get_cmap('jet', 256)
newcolors = viridis(np.linspace(0, 1, 256))
wt = np.array([1, 1, 1, 1])
newcolors[:25, :] = wt
newcmp = ListedColormap(newcolors)

r = np.sqrt(x**2 + y**2 + z**2)
index = (r<0.64) & (r>0.01) & (~np.isnan(r))
H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
X, Y = np.meshgrid(xedges[1:],yedges[1:])
plt.figure(dpi=200)
plt.contourf(X,Y,np.log(np.transpose(H1)+1), cmap=newcmp)
#plt.contourf(X, Y, H1.T, cmap=newcmp)
plt.colorbar()
plt.xlabel(r'$x^2 + y^2/m^2$')
plt.ylabel('$z$/m')


# In[48]:


plt.figure(dpi=300)
for i in np.arange(0.60, 0.20, -0.05):
    plt.hist(E[(r<i) & (r>0.01) & (E<4)], bins=50, label = 'cut: r<%.2f m' % i)
    plt.xlabel('Energy/MeV')
    plt.ylabel('Number of Events')
plt.legend()
plt.title('Run0258')
plt.savefig('Run0258_10.pdf')


# In[50]:


plt.figure(dpi=300)
for i in np.arange(0.60, 0.20, -0.05):
    plt.hist(E[(r<i) & (r>0.15) & (E<4) & (np.abs(z/(r+1e-6))>0.2)], bins=50, label = 'cut: r<%.2f m' % i)
    plt.xlabel('Energy/MeV')
    plt.ylabel('Number of Events')
plt.legend()
plt.title('Run0259')
plt.savefig('Run0259_10.pdf')


# In[51]:


plt.figure(dpi=300)
#plt.hist(r[(E>2) & (E<4)],bins=300)
for i in np.arange(0.5, 3, 0.5):
    plt.hist(r[(E>i) & (E<4.0)], bins=50, label = 'cut: E>%.2f MeV' % i)
    plt.xlabel('Radius/m')
    plt.ylabel('Number of Events')
plt.title('Recon radius')
plt.legend()
plt.show()


# In[ ]:




