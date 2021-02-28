#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tables
import matplotlib.pyplot as plt


# In[29]:


radius = np.arange(0,0.65,0.01)
    
plt.figure(dpi=300)
AIC = np.zeros_like(radius)
for index, i in enumerate(radius):
    h = tables.open_file('./file_+%.3f.h5' % i)
    tmp = np.zeros(30)
    for k in np.arange(5,30):
        tmp[k] = eval('h.root.AIC%d[()]' % k)
    AIC[index] = np.where(tmp == np.min(tmp[tmp!=0]))[0][0]

plt.plot(radius, AIC, '.', label='AIC, with acrylic reflection')

radius = np.arange(0.02,0.65,0.01)
AIC = np.zeros_like(radius)
for index, i in enumerate(radius):
    h = tables.open_file('../coeff_pe_1t_point_10_track_30/file_+%.3f.h5' % i)
    tmp = np.zeros(30)
    for k in np.arange(5,30):
        tmp[k] = eval('h.root.AIC%d[()]' % k)
    AIC[index] = np.where(tmp == np.min(tmp[tmp!=0]))[0][0]
    
plt.plot(radius, AIC, '.',label='AIC, without acrylic reflection')
plt.xlabel('radius/m')
plt.ylabel('best order')
plt.legend()
plt.savefig('../AIC.png')
plt.show()


# In[35]:


radius = np.arange(0,0.65,0.01)
    
plt.figure(dpi=300)
AIC = np.zeros_like(radius)
for index, i in enumerate(radius):
    h = tables.open_file('./file_+%.3f.h5' % i)
    tmp = np.zeros(30)
    for k in np.arange(5,30):
        tmp[k] = eval('h.root.AIC%d[()]' % k)
    AIC[index] = np.where(tmp == np.min(tmp[tmp!=0]))[0][0]

plt.plot(radius/0.65, AIC, '.', label='AIC, with acrylic reflection')

plt.xlabel(r'Relative $R$')
plt.ylabel('best order')

plt.savefig('../AIC1.png')
plt.show()


# In[26]:


radius = np.arange(0.02,0.65,0.01)
AIC = np.zeros_like(radius)
for index, i in enumerate(radius):
    h = tables.open_file('../coeff_pe_1t_point_10_track_30/file_+%.3f.h5' % i)
    tmp = np.zeros(30)
    for k in np.arange(5,30):
        tmp[k] = eval('h.root.AIC%d[()]' % k)
    AIC[index] = np.where(tmp == np.min(tmp[tmp!=0]))[0][0]


# In[34]:


plt.figure(dpi=300)
radius = np.arange(0.02,0.65,0.01)
AIC = np.zeros_like(radius)
for index, i in enumerate(radius):
    h = tables.open_file('../coeff_pe_1t_point_10_track_30/file_+%.3f.h5' % i)
    tmp = np.zeros(30)
    for k in np.arange(5,30):
        tmp[k] = eval('h.root.AIC%d[()]' % k)
    AIC[index] = np.where(tmp == np.min(tmp[tmp!=0]))[0][0]
plt.plot(radius/0.65, AIC, '.', label='AIC, with acrylic reflection')

plt.xlabel(r'Relative $R$')
plt.ylabel('best order')

plt.savefig('../AIC2.png')
plt.show()


# In[ ]:




