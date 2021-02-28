#!/usr/bin/env python
# coding: utf-8

# In[1]:


import ROOT
import sys
import tables
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


def Read(file):
    ROOT.PyConfig.IgnoreCommandLineOptions = True

    # Loop for ROOT files. 
    tTruth = ROOT.TChain("SimTriggerInfo")
    tTruth.Add(file)
    # Loop for event
    cnt = 0
    total = []
    for event in tTruth:
        if(len(event.PEList)==0):
            pass
        else:
            for truthinfo in event.truthList:
                E = truthinfo.EkMerged
                x = truthinfo.x
                y =  truthinfo.y
                z =  truthinfo.z
                for px in truthinfo.PrimaryParticleList:
                    pxx = px.px
                    pyy = px.py
                    pzz = px.pz
            Q = []
            for PE in event.PEList:
                Q.append(PE.PMTId)
                EventID = event.TriggerNo
                ChannelID = PE.PMTId
                PETime =  PE.HitPosInWindow
                photonTime = PE.photonTime
                PulseTime = PE.PulseTime
                dETime = PE.dETime
            PEs = np.zeros(30)
            PEs[0:np.max(Q)+1] = np.bincount(np.array(Q))
            for i in np.arange(30):
                total.append(PEs[i])
    return np.array(total)


# In[3]:


def ReadPMT():
    f = open(r"./PMT_1t.txt")
    line = f.readline()
    data_list = [] 
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)
    return PMT_pos
PMT_pos = ReadPMT()
PMT_pos[:,2]/np.sqrt(np.sum(PMT_pos**2, axis=1))


# In[4]:


q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.500_x.root')
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.500_z.root')
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)


# In[5]:


plt.figure(dpi=300)
plt.plot(PMT_pos[:,0], q1, 'r.')
plt.xlabel(r'Sampled $\cos \theta$')
plt.ylabel('prediction')
plt.title('vertex on x axis, r = 0.5 m')


# In[6]:


plt.figure(dpi=300)
plt.plot(PMT_pos[:,2], q2, 'r.')
plt.xlabel(r'Sampled $\cos \theta$')
plt.ylabel('prediction')
plt.title('vertex on z axis, z = 0.5 m')


# In[7]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

radius = 0.5
zs, expect = poly(radius/0.65)
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_x.root' % radius)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_y.root' % radius)
q3 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % radius)
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
q3 = np.mean(np.reshape(q3, (-1,30), order='C'), axis=0)

plt.figure(dpi=300)
plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

plt.plot(zs, expect/5,label='fitfit')
plt.title('radius=%.2f' % radius)
order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/5, label='fit')
plt.legend()
plt.show()


# In[8]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

direction = np.array((0.245,0.756,0.245))
direction1 = np.array((0.245,0.756,0.245))
radius = 0.30
zs, expect = poly(radius*np.linalg.norm(direction)/0.65)
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/point_bias1/1t_%+.3f_rand.root' % radius)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
plt.figure(dpi=300)

ct = np.sum(PMT_pos*direction, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction)
plt.plot(ct, q1, 'r.',label='0')
ct1 = np.sum(PMT_pos*direction1, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction1)
plt.plot(ct1, q2, 'b.',label='bias')

plt.plot(zs, expect/20000*4284,label='fitfit')

order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.2f0.h5' % (radius*np.linalg.norm(direction)))
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4284, label='fit')
plt.xlabel(r'$\cos\theta$')
plt.ylabel(r'hits')
plt.title('radius=%.2fR' % radius)
plt.legend()
plt.show()

print(radius*np.linalg.norm(direction))
print(np.sqrt(0.171**2+0.529**2+0.171**2))
print('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)


# In[9]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect


direction1 = np.array((0.245,0.756,0.245))
radius = 0.50
zs, expect = poly(radius/0.65)
plt.figure(dpi=300)
data = []
for index in np.arange(0.1,0.4,0.01):
    direction = np.array((0.245,0.756,index))

    q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_sweep2/1t_%+.3f_rand.root' % index)
    q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
    data.append(q1)
    ct = np.sum(PMT_pos*direction, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction)
    plt.plot(ct, q1, 'b.', alpha=0.2)
plt.plot(zs, expect/20000*4284,label='fitfit')

order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.2f0.h5' % (radius))
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4284, label='fit')
plt.xlabel(r'$\cos\theta$')
plt.ylabel(r'hits')
plt.title('radius=%.2fR' % radius)
plt.legend()
plt.show()

print(radius*np.linalg.norm(direction))
print(np.sqrt(0.171**2+0.529**2+0.171**2))
print('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)


# In[10]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect


direction1 = np.array((0.245,0.756,0.245))
radius = 0.20
zs, expect = poly(radius/0.65)
plt.figure(dpi=300)
data = []
for index in np.arange(0.1,0.4,0.01):
    direction = np.array((0.245,0.756,index))

    q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_sweep3/1t_%+.3f_rand.root' % index)
    q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
    data.append(q1)
    ct = np.sum(PMT_pos*direction, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction)
    plt.plot(ct, q1, 'b.', alpha=0.2)
plt.plot(zs, expect/20000*4284,label='fitfit')

order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.2f0.h5' % (radius))
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4284, label='fit')
plt.xlabel(r'$\cos\theta$')
plt.ylabel(r'hits')
plt.title('radius=%.2fR' % radius)
plt.legend()
plt.show()

print(radius*np.linalg.norm(direction))
print(np.sqrt(0.171**2+0.529**2+0.171**2))
print('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)


# In[11]:


np.max(ct)
np.sum(0.245**2+0.756**2+0.245*0.3)/np.sqrt(np.sum(0.245**2+0.756**2+0.245**2))/np.sqrt(np.sum(0.245**2+0.756**2+0.3**2))
np.sum(0.245**2+0.756**2+0.245*0.4)/np.sqrt(np.sum(0.245**2+0.756**2+0.245**2))/np.sqrt(np.sum(0.245**2+0.756**2+0.4**2))


# In[78]:


data


# In[74]:


A0 = np.array((152.939,471.925,152.939))
np.linalg.norm(A)
A1 = np.array((149.484, 461.263, 149.484))
np.linalg.norm(A1)
A2 = np.array((144.211, 444.995, 144.211))
np.linalg.norm(A1)


# In[45]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

direction = np.array((0.245,0.756,0.245))
direction1 = np.array((0.245,0.756,0.252))
radius = 0.30
zs, expect = poly(radius*np.linalg.norm(direction)/0.65)
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/point_bias_nfl1/1t_%+.3f_rand.root' % radius)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
plt.figure(dpi=300)

ct = np.sum(PMT_pos*direction, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction)
plt.plot(ct, q1, 'r.',label='0')
ct1 = np.sum(PMT_pos*direction1, axis=1)/np.linalg.norm(PMT_pos,axis=1)/np.linalg.norm(direction1)
plt.plot(ct1, q2, 'b.',label='bias')

plt.plot(zs, expect/20000*4284,label='fitfit')

order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.2f0.h5' % (radius*np.linalg.norm(direction)))
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4284, label='fit')
plt.xlabel(r'$\cos\theta$')
plt.ylabel(r'hits')
plt.title('radius=%.2fR' % radius)
plt.legend()
plt.show()

print(radius*np.linalg.norm(direction))
print(np.sqrt(0.171**2+0.529**2+0.171**2))
print('/mnt/stage/douwei/Simulation/1t_root/point_random_10/1t_%+.3f_rand.root' % radius)


# In[33]:


np.linalg.norm(direction)


# In[8]:


plt.figure(dpi=300)
plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

plt.plot(zs, expect/5,label='fitfit')
plt.title('radius=%.2f' % radius)
order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/5, label='fit')
plt.legend()
plt.show()


# In[147]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

radius = 0.500
zs, expect = poly(radius/0.65)
'''
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_x.root' % radius)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_y.root' % radius)
q3 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % radius)
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
q3 = np.mean(np.reshape(q3, (-1,30), order='C'), axis=0)
'''
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_%+.3f_x.root' % radius)
q3 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_%+.3f_z.root' % radius)
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
#q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
q3 = np.mean(np.reshape(q3, (-1,30), order='C'), axis=0)

plt.figure(dpi=300)
#plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
#plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
#plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

#plt.plot(zs, expect/5,label='fitfit')
plt.title('radius=%.2f m' % radius)
order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4285, label='distribution')
plt.xlabel(r'$\cos \theta$')
plt.ylabel(r'Expect')
plt.legend()
plt.semilogy()
plt.savefig('fit%.2f.png' % radius)
plt.show()

plt.figure()
z = PMT_pos[:,0]/0.832
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z, (np.exp(k)/20000*4285-q1)/q1,'.')
h.close()


# In[62]:


fig, ax = plt.subplots(2,2, figsize=(16,10), sharex=True)
h = tables.open_file('./coeff_pe_1t_point_30/file_%+.3f.h5' % radius)
z = PMT_pos[:,0]/0.832
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
ax[0][0].plot(z, (np.exp(k)-q1)/q1,'^')
ax[0][0].set_xlabel(r'cos$\theta$', fontsize=15)
ax[0][0].set_ylabel('(predict - mean)/mean', fontsize=15)
ax[0][0].set_title('vertex on x axis', fontsize=15)
'''
z = PMT_pos[:,1]/0.832
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
ax[0][1].plot(z, (np.exp(k)/20000*4285-q2)/q2,'^')
ax[0][1].set_xlabel(r'cos$\theta$', fontsize=15)
ax[0][1].set_ylabel('(predict - mean)/mean', fontsize=15)
ax[0][1].set_title('vertex on y axis', fontsize=15)
'''
z = PMT_pos[:,2]/0.832
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
ax[1][0].plot(z, (np.exp(k)-q3)/q3,'^')
ax[1][0].set_xlabel(r'cos$\theta$', fontsize=15)
ax[1][0].set_ylabel('(predict - mean)/mean', fontsize=15)
ax[1][0].set_title('vertex on z axis', fontsize=15)
h.close()
fig.suptitle('Prediction vs mean at 0.60 m', fontsize=30)
#fig.suptitle('(predict - mean)/mean')
#print(q3)
#print(PMT_pos[np.array((2,3,11,25))])
#print(PMT_pos[np.array((2,3))])


# In[ ]:





# In[52]:


print(PMT_pos[np.array((2,3,11,25))])


# In[51]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

for radius in np.arange(0.05,0.65,0.05):
    zs, expect = poly(radius/0.65)
    q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_x.root' % radius)
    q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_y.root' % radius)
    q3 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % radius)
    q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
    q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
    q3 = np.mean(np.reshape(q3, (-1,30), order='C'), axis=0)

    plt.figure(dpi=300)
    #plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
    #plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
    #plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

    plt.plot(zs, expect/20000*4285,label='fitfit')
    plt.title('radius=%.2f m' % radius)
    order = 20
    h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
    z = np.linspace(-1,1,100)
    k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
    plt.plot(z,np.exp(k)/20000*4285, label='fit')
    plt.xlabel(r'$\cos \theta$')
    plt.ylabel('Expectation')
    plt.legend()
    plt.semilogy()
    plt.show()


# In[168]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
def poly(radius):
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape
    
    z = radius
    zs = np.linspace(-1,1,1000)
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    #cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

    c = np.diag((np.ones(cut_pe)))
    x = LG.legval(zs, c).T

    expect = np.exp(np.dot(x,np.atleast_2d(k).T))
    return zs, expect

for radius in np.arange(0.50,0.65,0.01):
    zs, expect = poly(radius/0.65)
    q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_x.root' % radius)
    q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_y.root' % radius)
    q3 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % radius)
    q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
    q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
    q3 = np.mean(np.reshape(q3, (-1,30), order='C'), axis=0)

    plt.figure(dpi=300)
    plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
    plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
    plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

    #plt.plot(zs, expect/5,label='fitfit')
    plt.title('radius=%.2fm' % radius)
    order = 20
    h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
    z = np.linspace(-1,1,100)
    k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
    plt.plot(z,np.exp(k)/5, label='fit')
    plt.xlabel(r'cos$\theta$')
    plt.ylabel('Received photon number')
    plt.legend()
    plt.semilogy()
    plt.show()


# In[157]:


radius = 0.60
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % 0.60)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % 0.30)
q11 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q22 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)

plt.plot(PMT_pos[:,2], q11/np.sum(q11),'.',label='0.6')
plt.plot(PMT_pos[:,2], q22/np.sum(q22),'.',label='0.2')
plt.plot(PMT_pos[:,2], q1[0:30]/np.sum(q1[0:30]),'.', label='real')
plt.legend()


# In[159]:


radius = 0.60
q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % 0.60)
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z.root' % 0.30)
q11 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q22 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)

plt.plot(q11/np.sum(q11),'.-',label='0.6')
plt.plot(q22/np.sum(q22),'.-',label='0.2')
plt.plot(q1[0:30]/np.sum(q1[0:30]),'.-', label='real')
plt.legend()


# In[82]:


data = []
for i in np.arange(0,0.65,0.01):
    print(i)
    q2 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_z.root' %i)
    q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)
    data.append(q2)


# In[113]:





# In[91]:


h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_zQ.h5' %0.5)
Q = h.root.PETruthData[:]['Q']


# In[94]:


np.array(data).shape


# In[14]:


data = []
for i in np.arange(0,0.65,0.01):
    h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_xQ.h5' %i)
    Q = h.root.PETruthData[:]['Q']
    data.append(np.mean(np.reshape(Q, (-1,30), order='C'), axis=0))

from scipy.spatial import distance
Y = distance.pdist(np.array(data), 'cosine')
v= distance.squareform(Y)
x = np.arange(0,0.65,0.01)
plt.figure(dpi=300)
plt.contourf(x/0.65,x/0.65,v,levels=30, cmap='jet')

plt.xlabel('Relative $R$')
plt.ylabel('Relative $R$')   
plt.title(r'cosine distance on $x$ axis')
plt.colorbar()
plt.savefig('cosdist_x.png')
plt.show()


# In[15]:


data = []
for i in np.arange(0,0.65,0.01):
    h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_zQ.h5' %i)
    Q = h.root.PETruthData[:]['Q']
    data.append(np.mean(np.reshape(Q, (-1,30), order='C'), axis=0))

from scipy.spatial import distance
Y = distance.pdist(np.array(data), 'cosine')
v= distance.squareform(Y)
x = np.arange(0,0.65,0.01)
plt.figure(dpi=300)
plt.contourf(x/0.65,x/0.65,v,levels=30, cmap='jet')

plt.xlabel('Relative $R$')
plt.ylabel('Relative $R$')   
plt.title(r'cosine distance on $z$ axis')
plt.colorbar()
plt.savefig('cosdist_z.png')
plt.show()


# In[112]:


data[50]


# In[152]:


q1 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.500_x.root')
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.500_z.root')


# In[174]:


n1 = np.reshape(q1, (-1,30), order='C')
n2 = np.reshape(q2, (-1,30), order='C')
plt.figure(dpi=500)
plt.plot(PMT_pos[:,0]/0.835,n1.T,'.', alpha=0.005, color='blue', linewidth=0.1)
#plt.plot(PMT_pos[:,2]/0.835,n2.T,'.', alpha=0.005, color='green', linewidth=0.1)
plt.xlabel(r'$\cos\theta$')
plt.ylabel('Hits')
plt.show()


# In[166]:


n1.T.shape


# In[169]:


np.repeat(PMT_pos[:,2],(6731,1)).shape


# In[170]:





# In[ ]:




