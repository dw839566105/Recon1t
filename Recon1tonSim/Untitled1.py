#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tables
import numpy as np
import matplotlib.pyplot as plt


# In[14]:


h = tables.open_file('result_1t_ground_axis_Recon_1t_new1/1t_+0.600_z.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 < L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()


# In[19]:


h = tables.open_file('pt_+0.60_z.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 < L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()
print(np.sum(zr>0.5))


# In[20]:


h = tables.open_file('p_+0.60_z.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 < L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()
print(np.sum(zr>0.5))
print(zr.shape)


# In[21]:


3635/7388


# In[22]:


3772/7388


# In[58]:


def read(file):
    h = tables.open_file(file)
    x1 = h.root.Recon[:]['x_sph_in']
    y1 = h.root.Recon[:]['y_sph_in']
    z1 = h.root.Recon[:]['z_sph_in']
    L1 = h.root.Recon[:]['Likelihood_in']
    s1 = h.root.Recon[:]['success_in']
    x2 = h.root.Recon[:]['x_sph_out']
    y2 = h.root.Recon[:]['y_sph_out']
    z2 = h.root.Recon[:]['z_sph_out']
    L2 = h.root.Recon[:]['Likelihood_out']
    s2 = h.root.Recon[:]['success_out']
    index = (L1 < L2)
    xr = x2.copy()
    xr[index] = x1[index]
    xr = xr[(s1!=0) | (s2!=0)]
    yr = y2.copy()
    yr[index] = y1[index]
    yr = yr[(s1!=0) | (s2!=0)]
    zr = z2.copy()
    zr[index] = z1[index]
    zr = zr[(s1!=0) | (s2!=0)]
    return xr, yr, zr
x = np.empty(0)
y = np.empty(0)
z = np.empty(0)
xt = np.empty(0)
yt = np.empty(0)
zt = np.empty(0)    
for i in np.arange(0, 0.650, 0.01):        
    file = 'result_1t_ground_axis_Recon_1t_new1/1t_%+.3f_z.h5' % i
    x1, y1, z1 = read(file)
    x = np.hstack((x, x1))
    y = np.hstack((y, y1))
    z = np.hstack((z, z1))
    xt = np.hstack((xt, np.zeros_like(x1)))
    yt = np.hstack((yt, np.zeros_like(y1)))
    zt = np.hstack((zt, i * np.ones_like(z1)))
r = np.sqrt(x**2+y**2+z**2)
rt = np.sqrt(xt**2+yt**2+zt**2)
    
plt.hist2d(rt,r, bins=(30), cmap='Purples')
plt.show()


# In[69]:


h = tables.open_file('result_1t_ground_axis_Recon_1t_new/1t_+0.010_z.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 <= L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()
print(np.sum(np.abs(zr)>0.5))
print(zr.shape)


# In[62]:


h = tables.open_file('pt_+0.60_zz.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 < L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()
print(np.sum(zr>0.5))
print(zr.shape)


# In[67]:


print(np.where(L1>L2))


# In[70]:


h = tables.open_file('result_1t_ground_axis_Recon_1t_new1/1t_+0.010_z.h5')
x1 = h.root.Recon[:]['x_sph_in']
y1 = h.root.Recon[:]['y_sph_in']
z1 = h.root.Recon[:]['z_sph_in']
L1 = h.root.Recon[:]['Likelihood_in']
s1 = h.root.Recon[:]['success_in']
x2 = h.root.Recon[:]['x_sph_out']
y2 = h.root.Recon[:]['y_sph_out']
z2 = h.root.Recon[:]['z_sph_out']
L2 = h.root.Recon[:]['Likelihood_out']
s2 = h.root.Recon[:]['success_out']
index = (L1 <= L2)
xr = x2.copy()
xr[index] = x1[index]
xr = xr[(s1!=0) | (s2!=0)]
yr = y2.copy()
yr[index] = y1[index]
yr = yr[(s1!=0) | (s2!=0)]
zr = z2.copy()
zr[index] = z1[index]
zr = zr[(s1!=0) | (s2!=0)]
plt.hist(zr,bins=100)
plt.show()
print(np.sum(np.abs(zr)>0.5))
print(zr.shape)
print(np.where(L1>L2))


# In[ ]:




