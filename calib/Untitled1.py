#!/usr/bin/env python
# coding: utf-8

# In[7]:


import ROOT
import sys
import tables
import matplotlib.pyplot as plt
import numpy as np


# In[41]:


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


# In[147]:


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


# In[47]:


q1 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.500_x.root')
q2 = Read('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.500_z.root')
q1 = np.mean(np.reshape(q1, (-1,30), order='C'), axis=0)
q2 = np.mean(np.reshape(q2, (-1,30), order='C'), axis=0)


# In[49]:


plt.figure(dpi=300)
plt.plot(PMT_pos[:,0], q1, 'r.')
plt.plot(PMT_pos[:,2], q2, 'b.')


# In[119]:


from numpy.polynomial import legendre as LG
from scipy.linalg import norm
h = tables.open_file('../calib/PE_coeff_1t.h5','r')
coeff_pe_in = h.root.coeff_L_in[:]
coeff_pe_out = h.root.coeff_L_out[:]
bd_pe = h.root.bd[()] 
h.close()
cut_pe, fitcut_pe = coeff_pe_in.shape

data = []
zs = np.arange(-1,1.01,0.01)
for z in zs:
    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    data.append(k)
data = np.array(data)

#cos_theta = np.dot(vertex,PMT_pos.T) / (z*norm(PMT_pos,axis=1))

c = np.diag((np.ones(cut_pe)))
x = LG.legval(zs, c).T

expect = np.exp(np.dot(x,np.atleast_2d(data[150]).T))
plt.figure(dpi=300)
plt.plot(PMT_pos[:,0], q1, 'r.')
plt.plot(PMT_pos[:,2], q2, 'b.')
plt.plot(zs,expect/5)            


# In[103]:


plt.plot(np.sum(np.dot(x,data[0]),axis=1))


# In[140]:


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


# In[141]:


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


# In[165]:


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

radius = 0.25
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
plt.title('radius=%.2f' % radius)
order = 20
h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
z = np.linspace(-1,1,100)
k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
plt.plot(z,np.exp(k)/20000*4285, label='fit')
plt.legend()
plt.semilogy()
plt.savefig('fit%.2f.png' % radius)
plt.show()


# In[ ]:


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
    plt.plot(PMT_pos[:,0]/0.832, q1, 'r.',label='x')
    plt.plot(PMT_pos[:,1]/0.832, q2, 'k.',label='y')
    plt.plot(PMT_pos[:,2]/0.832, q3, 'b.',label='z')

    plt.plot(zs, expect/5,label='fitfit')
    plt.title('radius=%.2f' % radius)
    order = 20
    h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
    z = np.linspace(-1,1,100)
    k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
    plt.plot(z,np.exp(k)/20000*4285, label='fit')
    plt.legend()
    plt.semilogy()
    plt.show()


# In[146]:


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

    plt.plot(zs, expect/5,label='fitfit')
    plt.title('radius=%.2f' % radius)
    order = 20
    h = tables.open_file('./coeff_pe_1t_reflection0.00_30/file_%+.3f.h5' % radius)
    z = np.linspace(-1,1,100)
    k = LG.legval(z, eval('h.root.coeff%d[:]' % order))
    plt.plot(z,np.exp(k)/5, label='fit')
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


# In[ ]:




