#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tables
import h5py
import numpy as np
import matplotlib.pyplot as plt


# In[7]:


N = 30
H_r, edges_r = np.histogram(v[:,0]**3, bins = N)
H_t, edges_t = np.histogram(np.cos(v[:,1]), bins = N)
H_p, edges_p = np.histogram(v[:,2], bins = N)


# In[8]:


bins = np.zeros((N**3, 3))
mean = np.zeros((N**3, 30))
cnt = 0
for i1, i in enumerate(edges_r[0:-1]):
    for j1, j in enumerate(edges_t[0:-1]):
        for k1, k in enumerate(edges_p[0:-1]):
            bins[cnt:] = np.array((i,j,k))
            cnt = cnt + 1
            index = (v[:,0]**3>edges_r[i1]) & (v[:,0]**3<edges_r[i1+1]) &                 (np.cos(v[:,1])>edges_t[j1]) & (np.cos(v[:,1])<edges_t[j1+1]) &                 (v[:,2]>edges_p[k1]) & (v[:,2]<edges_p[k1+1])
            mean[cnt,:] = np.mean(data[index],axis=0)


# In[66]:


print(data.shape)
print(vertex.shape)
print(v.shape)


# In[9]:


pe = 0
N = 0
for i in np.arange(50):
    h = tables.open_file('hist%d.h5' % (i+1))
    
    mean = h.root.mean[:]
    vertex = h.root.vertex[:]
    
    pe = pe + mean[:,0:-1]
    N = N + mean[:,-1]
    h.close()
    
total = (pe.T/N).T
print(total.shape)

with h5py.File('total.h5','w') as out:   
    out.create_dataset('mean', data = total)
    out.create_dataset('vertex', data = vertex)


# In[88]:





# In[12]:


import tables
import numpy as np
import h5py

r = np.arange(0.01, 0.65, 0.01)
ra = []
data = []
sign = '+'
axis = 'x'
for radius in r:
    h1 = tables.open_file("./MCmean/file%s%s%.3f.h5" % (sign,axis,radius))
    try:
        a = h1.root.result[:]
        data.append(a)
        ra.append(radius)
    except:
        print(radius)
        pass
    h1.close()

real = []
recon = np.zeros((3,0))
for index, x in enumerate(ra):
    real = np.hstack((real, ra[index] * np.ones(np.size(data[index][:,0]))))
    recon = np.hstack((recon, data[index].T))
    
import matplotlib.pyplot as plt

H, xedges, yedges = np.histogram2d(real, recon[0,:], bins=30)
plt.figure(figsize=(8,6))
plt.contourf(xedges[0:-1], yedges[0:-1], np.log(np.nan_to_num(H)+1).T)
plt.xlabel('recon radius/m',fontsize=20)
plt.ylabel('real radius/m', fontsize=20)
plt.colorbar()
plt.show()
plt.figure(figsize=(8,6))
plt.hist2d(real, recon[0,:], bins=30)
plt.xlabel('recon radius/m',fontsize=20)
plt.ylabel('real radius/m', fontsize=20)
plt.colorbar()
plt.show()


# In[1]:


import tables
import numpy as np
import h5py

r = np.arange(0.01, 0.65, 0.01)
ra = []
data = []
sign = '+'
axis = 'z'
for radius in r:
    h1 = tables.open_file("./MCmean/file%s%s%.3f.h5" % (sign,axis,radius))
    try:
        a = h1.root.result[:]
        data.append(a)
        ra.append(radius)
    except:
        print(radius)
        pass
    h1.close()

real = []
recon = np.zeros((3,0))
for index, x in enumerate(ra):
    real = np.hstack((real, ra[index] * np.ones(np.size(data[index][:,0]))))
    recon = np.hstack((recon, data[index].T))
    
import matplotlib.pyplot as plt

for i in np.arange(0.2,0.65,0.05):
    index = np.where((r-i)==np.min(np.abs(r-i)))
    plt.figure(figsize=(8, 6))
    plt.hist(np.sqrt(np.sum(data[index[0][0]]**2, axis=1)),bins=np.arange(0,0.65,0.01))
    plt.title('real radius=%.2fm' % i, fontsize=20)
    plt.xlabel('recon radius/m', fontsize=20)
    plt.ylabel('counts', fontsize=20)
    plt.show()


# In[86]:


import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../calib/PMT_1t.txt")
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
#print(np.sum(np.isnan(np.sum(tp,axis=0))))
#exit()
def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, x, y, z

def main(path, radius, axis, sign='+'):
    EventIDx, ChannelIDx, xx, yx, zx = readchain(sign + radius, path, axis)
    x1 = np.array((xx[0], yx[0], zx[0]))
    size = np.size(np.unique(EventIDx))

    EventID = EventIDx
    ChannelID = ChannelIDx

    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp
    a1 = np.empty((0))
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
        a1 = np.hstack((a1,h.root.PETruthData[:]['Q']))
        h.close()
    print(np.array(a1))
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)
        test = test/(np.sum(test)/np.sum(event_pe))
        h.close()
        L_test = np.nansum(-test + np.log(test)*event_pe)
        index = np.where(L == np.max(L))[0][0]
        L1 = np.reshape(L[0:-1],(-1,64), order='F')
        plt.figure(dpi=300)
        plt.axhline(L_test,color='black',label='self likelihood')
        plt.plot(np.linspace(0.01,0.64,64),np.max(L1, axis=0),label='self likelihood')
        plt.legend()
        plt.xlabel('radius/m')
        plt.ylabel('log likelihood')
        plt.figure(dpi=300)
        plt.plot(corr[index])
        plt.plot(test)
        plt.plot(event_pe)
        plt.figure(dpi=300)
        plt.plot(PMT_pos[:,2], corr[index],'.', label='best fit')
        plt.plot(PMT_pos[:,2], test,'*',label='mean')
        plt.plot(PMT_pos[:,2], event_pe,'^',label='single')
        plt.legend()
        plt.xlabel('z position')
        plt.ylabel('pe')
        if(k_index==10):
            break

main('/mnt/stage/douwei/Simulation/1t_root/ground_axis/','0.600', 'z', '+')


# In[43]:


h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
plt.plot(np.mean(np.reshape(h.root.PETruthData[:]['Q'], (-1,30),order='C'), axis=0))


# In[94]:


def time_tpl(radius=0.600):
    time = np.empty(0)
    ch = np.empty(0)
    td = np.zeros(30)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_zQ.h5' % radius)
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_%+.3f_z_%dQ.h5' % (radius, i))
        time = np.hstack((time, h.root.GroundTruth[:]['PulseTime']))
        ch = np.hstack((ch, h.root.GroundTruth[:]['ChannelID']))

        for j in np.arange(30):
            td[j] = np.quantile(time[ch==j], 0.1)
        h.close()
    return td

a1 = time_tpl(0.6)
a2 = time_tpl(0.2)


# In[95]:


plt.plot(a1)
plt.plot(a2)
plt.show()


# In[97]:


import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../calib/PMT_1t.txt")
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
#print(np.sum(np.isnan(np.sum(tp,axis=0))))
#exit()
def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, x, y, z

def main(path, radius, axis, sign='+'):
    EventIDx, ChannelIDx, xx, yx, zx = readchain(sign + radius, path, axis)
    x1 = np.array((xx[0], yx[0], zx[0]))
    size = np.size(np.unique(EventIDx))

    EventID = EventIDx
    ChannelID = ChannelIDx

    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp
    a1 = np.empty((0))
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
        a1 = np.hstack((a1,h.root.PETruthData[:]['Q']))
        h.close()
    print(np.array(a1))
    
    a2 = np.empty((0))
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_z_%dQ.h5' % i)
        a2 = np.hstack((a2,h.root.PETruthData[:]['Q']))
        h.close()
    print(np.array(a2))
    
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)
        test = test/(np.sum(test)/np.sum(event_pe))
        test1 = np.mean(np.reshape(a2, (-1,30),order='C'), axis=0)
        test1 = test1/(np.sum(test1)/np.sum(event_pe))
        h.close()
        L_test = np.nansum(-test + np.log(test)*event_pe)
        L_test2 = np.nansum(-test1 + np.log(test1)*event_pe)
        index = np.where(L == np.max(L))[0][0]
        L1 = np.reshape(L[0:-1],(-1,64), order='F')
        plt.figure(dpi=300)
        plt.axhline(L_test,color='black',label='self likelihood')
        plt.axhline(L_test2,color='red',label='self likelihood')
        plt.plot(np.linspace(0.01,0.64,64),np.max(L1, axis=0),label='self likelihood')
        plt.legend()
        plt.xlabel('radius/m')
        plt.ylabel('log likelihood')
        plt.figure(dpi=300)
        plt.plot(corr[index])
        plt.plot(test)
        plt.plot(event_pe)
        plt.figure(dpi=300)
        plt.plot(PMT_pos[:,2], corr[index],'.', label='best fit')
        plt.plot(PMT_pos[:,2], test,'*',label='mean')
        plt.plot(PMT_pos[:,2], event_pe,'^',label='single')
        plt.legend()
        plt.xlabel('z position')
        plt.ylabel('pe')
        if(k_index==10):
            break

main('/mnt/stage/douwei/Simulation/1t_root/ground_axis/','0.600', 'z', '+')


# In[121]:


import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../calib/PMT_1t.txt")
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
#print(np.sum(np.isnan(np.sum(tp,axis=0))))
#exit()
def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    PulseTime = truthtable[:]['PulseTime']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                PulseTime = PulseTime[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, PulseTime, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, PulseTime, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, PulseTime1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PulseTime = np.hstack((PulseTime, PulseTime1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, PulseTime, x, y, z

def main(path, radius, axis, sign='+'):
    EventID, ChannelID, PulseTime, x, y, z = readchain(sign + radius, path, axis)
    x1 = np.array((x[0], y[0], z[0]))
    size = np.size(np.unique(EventID))
    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp
    a1 = np.empty(0)
    b1 = np.empty(0)
    c1 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
        a1 = np.hstack((a1, h.root.PETruthData[:]['Q']))
        b1 = np.hstack((b1, h.root.GroundTruth[:]['PulseTime']))
        c1 = np.hstack((c1, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    a2 = np.empty(0)
    b2 = np.empty(0)
    c2 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_z_%dQ.h5' % i)
        a2 = np.hstack((a2,h.root.PETruthData[:]['Q']))
        b2 = np.hstack((b2, h.root.GroundTruth[:]['PulseTime']))
        c2 = np.hstack((c2, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    t1 = np.zeros(30)
    t2 = np.zeros(30)
    for i in np.arange(30):
        t1[i] = np.quantile(b1[c1==i],0.1)
        t2[i] = np.quantile(b2[c2==i],0.1)
        
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        event_time = PulseTime[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)
        test = test/(np.sum(test)/np.sum(event_pe))
        test1 = np.mean(np.reshape(a2, (-1,30),order='C'), axis=0)
        test1 = test1/(np.sum(test1)/np.sum(event_pe))
        h.close()
        L_test = np.nansum(-test + np.log(test)*event_pe)
        L_test2 = np.nansum(-test1 + np.log(test1)*event_pe)
        index = np.where(L == np.max(L))[0][0]
        L1 = np.reshape(L[0:-1],(-1,64), order='F')
        plt.figure(dpi=300)
        plt.axhline(L_test,color='black',label='truth likelihood (0.6)')
        plt.axhline(L_test2,color='red',label='best likelihood (0.25)')
        plt.plot(np.linspace(0.01,0.64,64),np.max(L1, axis=0),label='pe likelihood')
        plt.legend()
        plt.xlabel('radius/m')
        plt.ylabel('log likelihood')
        
        tau = 0.1
        delta1 = event_time - t1[ChannelID[EventID==k]]
        print(PulseTime.shape)
        LT1 = np.sum(np.sum(delta1[delta1>0])*(1-tau) - np.sum(delta1[delta1<0])*tau)/3
        delta2 = event_time - t2[ChannelID[EventID==k]]
        LT2 = np.sum(np.sum(delta2[delta2>0])*(1-tau) - np.sum(delta2[delta2<0])*tau)/3
        plt.figure(dpi=300)
        plt.axhline(L_test - LT1,color='black',label='truth likelihood (0.6)')
        plt.axhline(L_test2 - LT2,color='red',label='best likelihood (0.25)')
        plt.plot(np.linspace(0.01,0.64,64),np.max(L1, axis=0) - LT1, label='pe likelihood + time likelihood of 0.6')
        plt.legend()
        plt.xlabel('radius/m')
        plt.ylabel('log likelihood')
        '''
        plt.figure(dpi=300)
        plt.plot(corr[index])
        plt.plot(test)
        plt.plot(event_pe)
        plt.figure(dpi=300)
        plt.plot(PMT_pos[:,2], corr[index],'.', label='best fit')
        plt.plot(PMT_pos[:,2], test,'*',label='mean')
        plt.plot(PMT_pos[:,2], event_pe,'^',label='single')
        plt.legend()
        plt.xlabel('z position')
        plt.ylabel('pe')
        '''
        if(k_index==10):
            break

main('/mnt/stage/douwei/Simulation/1t_root/ground_axis/','0.600', 'z', '+')


# In[125]:


import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../calib/PMT_1t.txt")
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
#print(np.sum(np.isnan(np.sum(tp,axis=0))))
#exit()
def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    PulseTime = truthtable[:]['PulseTime']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                PulseTime = PulseTime[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, PulseTime, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, PulseTime, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, PulseTime1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PulseTime = np.hstack((PulseTime, PulseTime1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, PulseTime, x, y, z

def main(path, radius, axis, sign='+'):
    cnt = 0
    EventID, ChannelID, PulseTime, x, y, z = readchain(sign + radius, path, axis)
    x1 = np.array((x[0], y[0], z[0]))
    size = np.size(np.unique(EventID))
    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp
    a1 = np.empty(0)
    b1 = np.empty(0)
    c1 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
        a1 = np.hstack((a1, h.root.PETruthData[:]['Q']))
        b1 = np.hstack((b1, h.root.GroundTruth[:]['PulseTime']))
        c1 = np.hstack((c1, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    a2 = np.empty(0)
    b2 = np.empty(0)
    c2 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_z_%dQ.h5' % i)
        a2 = np.hstack((a2,h.root.PETruthData[:]['Q']))
        b2 = np.hstack((b2, h.root.GroundTruth[:]['PulseTime']))
        c2 = np.hstack((c2, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    t1 = np.zeros(30)
    t2 = np.zeros(30)
    for i in np.arange(30):
        t1[i] = np.quantile(b1[c1==i],0.1)
        t2[i] = np.quantile(b2[c2==i],0.1)
        
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        event_time = PulseTime[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)
        test = test/(np.sum(test)/np.sum(event_pe))
        test1 = np.mean(np.reshape(a2, (-1,30),order='C'), axis=0)
        test1 = test1/(np.sum(test1)/np.sum(event_pe))
        h.close()
        L_test = np.nansum(-test + np.log(test)*event_pe)
        L_test2 = np.nansum(-test1 + np.log(test1)*event_pe)
        index = np.where(L == np.max(L))[0][0]
        L1 = np.reshape(L[0:-1],(-1,64), order='F')
       
        tau = 0.1
        delta1 = event_time - t1[ChannelID[EventID==k]]
        LT1 = np.sum(np.sum(delta1[delta1>0])*(1-tau) - np.sum(delta1[delta1<0])*tau)/3
        delta2 = event_time - t2[ChannelID[EventID==k]]
        LT2 = np.sum(np.sum(delta2[delta2>0])*(1-tau) - np.sum(delta2[delta2<0])*tau)/3
        if(np.max(L)>L_test):
            cnt = cnt+1
    print(k_index, cnt)

main('/mnt/stage/douwei/Simulation/1t_root/ground_axis/','0.600', 'z', '+')


# In[127]:


import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../calib/PMT_1t.txt")
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
#print(np.sum(np.isnan(np.sum(tp,axis=0))))
#exit()
def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    PulseTime = truthtable[:]['PulseTime']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                PulseTime = PulseTime[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, PulseTime, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, PulseTime, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, PulseTime1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PulseTime = np.hstack((PulseTime, PulseTime1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, PulseTime, x, y, z

def main(path, radius, axis, sign='+'):
    cnt = 0
    EventID, ChannelID, PulseTime, x, y, z = readchain(sign + radius, path, axis)
    x1 = np.array((x[0], y[0], z[0]))
    size = np.size(np.unique(EventID))
    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp
    a1 = np.empty(0)
    b1 = np.empty(0)
    c1 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
        a1 = np.hstack((a1, h.root.PETruthData[:]['Q']))
        b1 = np.hstack((b1, h.root.GroundTruth[:]['PulseTime']))
        c1 = np.hstack((c1, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    a2 = np.empty(0)
    b2 = np.empty(0)
    c2 = np.empty(0)
    for i in np.arange(3):
        if (i == 0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_zQ.h5')
        else:
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.250_z_%dQ.h5' % i)
        a2 = np.hstack((a2,h.root.PETruthData[:]['Q']))
        b2 = np.hstack((b2, h.root.GroundTruth[:]['PulseTime']))
        c2 = np.hstack((c2, h.root.GroundTruth[:]['ChannelID']))
        h.close()
    
    t1 = np.zeros(30)
    t2 = np.zeros(30)
    for i in np.arange(30):
        t1[i] = np.quantile(b1[c1==i],0.1)
        t2[i] = np.quantile(b2[c2==i],0.1)
        
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        event_time = PulseTime[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)
        test = test/(np.sum(test)/np.sum(event_pe))
        test1 = np.mean(np.reshape(a2, (-1,30),order='C'), axis=0)
        test1 = test1/(np.sum(test1)/np.sum(event_pe))
        h.close()
        L_test = np.nansum(-test + np.log(test)*event_pe)
        L_test2 = np.nansum(-test1 + np.log(test1)*event_pe)
        index = np.where(L == np.max(L))[0][0]
        L1 = np.reshape(L[0:-1],(-1,64), order='F')
       
        tau = 0.1
        delta1 = event_time - t1[ChannelID[EventID==k]]
        LT1 = np.sum(np.sum(delta1[delta1>0])*(1-tau) - np.sum(delta1[delta1<0])*tau)/3
        delta2 = event_time - t2[ChannelID[EventID==k]]
        LT2 = np.sum(np.sum(delta2[delta2>0])*(1-tau) - np.sum(delta2[delta2<0])*tau)/3
        if(np.max(L)>L_test2):
            cnt = cnt+1
        if(k_index==1000):
            break
    print(k_index, cnt)

main('/mnt/stage/douwei/Simulation/1t_root/ground_axis/','0.600', 'z', '+')


# In[ ]:




