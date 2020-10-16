#!/usr/bin/env python
# coding: utf-8

# In[117]:


import tables
import numpy as np
import matplotlib.pyplot as plt


# In[165]:


Q = x = y = z = np.empty(0)
for i in np.arange(300):
    if(i==0):
        h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/8.0MeV_shell/1t_+0.500Q.h5')
        Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
        x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
        y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
        z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
        h.close()
    elif(i>0):
        try:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/8.0MeV_shell/1t_+0.500_%dQ.h5' % i)
            Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
            x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
            y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
            z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
            h.close()
        except:
            break
r = np.sqrt(x**2+y**2+z**2)
cos_theta = z/(r+1e-6)
phi = np.arctan2(y, x)

index1 = np.digitize(cos_theta, np.linspace(-1,1,31)) 
index2 = np.digitize(phi, np.linspace(-np.pi, np.pi, 31))
data = []
xt = []
yt = []
zt = []
for i in np.arange(1,31):
    for j in np.arange(1,31):
        Q1 = np.reshape(Q,(-1,30))
        pe = np.mean(Q1[(index1==i) & (index2==j)], axis = 0)
        data.append(pe)
        xt.append(np.mean(x[(index1==i) & (index2==j)]))
        yt.append(np.mean(y[(index1==i) & (index2==j)]))
        zt.append(np.mean(z[(index1==i) & (index2==j)]))
        #plt.figure(num=i,figsize=(10,10))
        #plt.scatter(x[(index1==i) & (index2==j)], y[(index1==i) & (index2==j)],s=2)

data = np.nan_to_num(np.array(data))
xt = np.array(xt)
yt = np.array(yt)
zt = np.array(zt)


# In[167]:


r = np.sqrt(x**2+y**2+z**2)
plt.hist(r,bins=50)
plt.show()
plt.hist(z/r,bins=100)
plt.show()
plt.hist(np.arctan2(y, x),bins=100)
plt.show()
print(np.sum(np.isnan(phi)))


# In[51]:


a = np.hstack((np.arange(30), np.arange(30)))
np.reshape(a,(-1,30))


# In[13]:





# In[15]:


plt.plot(np.mean(np.reshape(Q,(-1,30))[index==1],axis=0))


# In[62]:


H, edges = np.histogram(z/r,bins=30)
index1 = np.digitize(z/r,np.linspace(-1,1,31))
index2 = np.digitize(np.arctan(y/x)+(y>0)*np.pi,np.linspace(-np.pi/2,np.pi/2*3,31))
for i in np.arange(1,31):
    for j in np.arange(1,31):
        Q1 = np.reshape(Q,(-1,30))
        plt.plot(np.mean(Q1[(index1==i) & (index2==j)], axis = 0))
        print(np.sum((index1==i) & (index2==j)))
plt.hist(y[(index1==10) & (index2==8)],bins=30)


# In[59]:


np.mean(Q1[(index1==i) & (index2==j)], axis = 0).shape


# In[3]:


import h5py
import numpy as np
import tables
k = np.empty((0,30))
x = y = z = np.empty(0)
for i in np.arange(0,0.65,0.01):
    h = tables.open_file('Tpl_%.3f.h5' % i)
    k = np.vstack((k,np.array(h.root.template[:])))
    x = np.hstack((x,np.array(h.root.x[:])))
    y = np.hstack((y,np.array(h.root.y[:])))
    z = np.hstack((z,np.array(h.root.z[:])))
    h.close()
index = ~np.isnan(np.sum(k,axis=1))
k = k[index]
x = x[index]
y = y[index]
z = z[index]
with h5py.File('./template.h5','w') as out:
        out.create_dataset('template', data = k)
        out.create_dataset('x', data = x)
        out.create_dataset('y', data = y)
        out.create_dataset('z', data = z)
print(x.shape)


# In[144]:


import h5py
k = np.empty((0,30))
x = y = z = np.empty(0)
for i in np.arange(0,0.65,0.01):
    h = tables.open_file('Tpl_%.3f.h5' % i)
    k = np.vstack((k,np.nan_to_num(np.array(h.root.template[:]))))
    x = np.hstack((x,np.array(h.root.x[:])))
    y = np.hstack((y,np.array(h.root.y[:])))
    z = np.hstack((z,np.array(h.root.z[:])))
    h.close()
index = (np.sum(k,axis=1) == 0)
#k = k[index]
#x = x[index]
#y = y[index]
#z = z[index]

print(x,y,z)


# In[155]:


a = np.sum(k, axis=1)
plt.bar(np.where(a==0)[0],np.ones_like(np.where(a==0)[0]))


# In[10]:


h = tables.open_file('./template.h5')
h.root


# In[12]:


import h5py
import numpy as np
import tables
k = np.empty((0,30))
x = y = z = np.empty(0)
for i in np.arange(0,0.65,0.01):
    print(k.shape)
    h = tables.open_file('Tpl_%.3f.h5' % i)
    k = np.vstack((k,np.array(h.root.template[:])))
    x = np.hstack((x,np.array(h.root.x[:])))
    y = np.hstack((y,np.array(h.root.y[:])))
    z = np.hstack((z,np.array(h.root.z[:])))
    h.close()
index = ~np.isnan(np.sum(k,axis=1))
k = k[index]
x = x[index]
y = y[index]
z = z[index]
print(k.shape)


# In[43]:


h = tables.open_file('./template.h5')
r = np.sqrt(h.root.x[:]**2 + h.root.y[:]**2 + h.root.z[:]**2)
plt.plot(h.root.template[:][(r>599)&(r<601)][0])
plt.plot(h.root.template[:][(r>599)&(r<601)][29])
plt.plot(h.root.template[:][(r>249)&(r<251)][0])


# In[36]:


import matplotlib.pyplot as plt
plt.hist(r,bins=100)
plt.show()


# In[86]:


r = np.sqrt(h.root.x[:]**2 + h.root.y[:]**2 + h.root.z[:]**2)
a1 = h.root.template[:][(r>599)&(r<601)&(np.abs(h.root.x[:])<30)]
a2 = h.root.template[:][(r>249)&(r<251)&(np.abs(h.root.x[:])<10)]
print(a1[0]/np.sum(a1[0]))
print(a1[0]/np.sum(a2[0]))
plt.figure()
plt.bar(x = np.arange(30), height = a1[0]/np.sum(a1[0]), width=0.1)
plt.figure()
plt.bar(x = np.arange(30), height = a2[0]/np.sum(a2[0]), width=0.1)


# In[71]:


np.sum(a1[0]/np.sum(a1))
np.sum(a2[0]/np.sum(a2))


# In[ ]:




