#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tables
import matplotlib.pyplot as plt
import numpy as np
import h5py


# In[61]:


def readtime(path):
    coeff_time = []
    rd = []
    ra = np.arange(0.01,0.650,0.01)
    for i in ra:
        try:
            h = tables.open_file(path + '/file_%+.3f.h5' % i)
            coeff_time.append(h.root.coeff25[:])
            rd.append(i)
        except:
            pass
    coeff_time = np.array(coeff_time)
    '''
    for i in np.arange(5):
        plt.figure(num=i+1, dpi=300)
        plt.plot(ra[1:-2], coeff_time[1:-2,i])
        plt.xlabel('Radius/m')
        plt.ylabel('Legendre Coefficient of Time Info')
        plt.title('Time %d-th order, quantile = %.2f' % (i,0.01) )
        plt.savefig('/mnt/stage/douwei/Simulation/1t_root/fig/Time%d.pdf' % i)
        plt.show()
    '''
    return np.array(rd), coeff_time.T


# In[62]:


order = 10
fit_order = 10
rd, coeff_pe = readtime('./coeff_time_1t_reflection0.05_2MeV_0.1_1/')
print(rd.shape, coeff_pe.shape)

bd = 0.91
deg = 20
d = np.where(np.abs(rd-np.max(rd)*bd) == np.min(np.abs(rd-np.max(rd)*bd)))

for i in np.arange(order):
    index_in = (rd<=rd[d[0][0]])
    index_out = (rd>=rd[d[0][0]])

    if not i % 2:
        w_in = np.ones_like(rd[index_in])
        w_out = np.ones_like(rd[index_out])
        w_in[-1] = 1000
        w_out[0] = 1000
        if(i==0):
                w_in[-1] = 1000
        # Legendre coeff

        B_in, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))),                                                   np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])),                                                   deg = deg, w = np.hstack((w_in,w_in)), full = True)

        y1_in = np.polynomial.legendre.legval(rd[index_in]/np.max(rd), B_in)
        
        B_out, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))),                                                   np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])),                                                   deg = deg, w = np.hstack((w_out,w_out)), full = True)
        y1_out = np.polynomial.legendre.legval(rd[index_out]/np.max(rd), B_out)
        # polynormial coeff
        C_in = np.polyfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))),                           np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])),                           deg=deg)
        y2_in = np.polyval(C_in, rd[index_in]/np.max(rd))

        C_out = np.polyfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))),                   np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])),                   deg=deg)
        y2_out = np.polyval(C_out, rd[index_out]/np.max(rd))

    else:
        B_in, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))),                                                   np.hstack((coeff_pe[i,index_in], -coeff_pe[i,index_in])),                                                   deg=deg, w = np.hstack((w_in,w_in)), full=True)
        y1_in = np.polynomial.legendre.legval(rd[index_in]/np.max(rd), B_in)
        
        B_out, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))),                                                   np.hstack((coeff_pe[i,index_out], -coeff_pe[i,index_out])),                                                   deg=deg, w = np.hstack((w_out,w_out)), full=True)
        y1_out = np.polynomial.legendre.legval(rd[index_out]/np.max(rd), B_out)
        C_in = np.polyfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))),                   np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])),                   deg=deg)
        y2_in = np.polyval(C_in, rd[index_in]/np.max(rd))
        
        C_out = np.polyfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))),                   np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])),                   deg=deg)
        y2_out = np.polyval(C_out, rd[index_out]/np.max(rd))
 
    plt.figure(num = i+1, dpi = 300)
    plt.plot(rd, np.hstack((y1_in, y1_out[1:])), label = 'Legendre')
    plt.axvline(0.65*bd)
    #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
    plt.plot(rd, coeff_pe[i], label='real')
    plt.xlabel('radius/m')
    plt.ylabel('Time Legendre coefficient')
    plt.legend()
    plt.savefig('Time')
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)


# In[99]:


from numpy.polynomial import legendre as LG
def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
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
def Legendre_coeff(PMT_pos_rep, vertex, cut):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos_rep[:,0])
    # oh, it will use norm in future version
    
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos_rep,axis=1)            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos_rep**2,axis=1))
    else:
        # make r=0 as a boundry, it should be improved
        cos_theta = np.ones(size)

    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)

    print(PMT_pos_rep.shape, x.shape, cos_theta.shape)
    return x, cos_theta

plt.figure(dpi=300)
h = tables.open_file('coeff_time_1t_reflection0.05_2MeV_0.1_1/file_+0.620.h5')
z = np.linspace(-1,1,100)
k = LG.legval(z, h.root.coeff25[:])
h.close()
plt.plot(z, k, 'r-')

h1 = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+0.620Q.h5')
a1 = h1.root.GroundTruth[:]['PulseTime']
a2 = h1.root.GroundTruth[:]['ChannelID']
a3 = h1.root.GroundTruth[:]['EventID']
x = h1.root.TruthData[:]['x']
y = h1.root.TruthData[:]['y']
z = h1.root.TruthData[:]['z']
x1 = np.vstack((x, y, z)).T
counts = np.bincount(a3)
counts = counts[counts!=0]
PMT_pos = ReadPMT()
vertex = np.repeat(x1, counts, axis=0)
PMT_pos_rep = PMT_pos[a2]
tmp_x_p, cos_theta = Legendre_coeff(PMT_pos_rep, vertex, 6)
plt.scatter(cos_theta, a1, s=0.3,alpha=0.05)
plt.show()


# In[95]:


plt.hist(a1, bins=1000)
plt.show()


# In[114]:


plt.figure(dpi=300)
H, xe, ye = np.histogram2d(cos_theta, a1, bins=1000)
plt.contourf(xe[:-1], ye[:-1], np.log(H.T+1))
#plt.contourf(xe[:-1], ye[:-1], H.T)
plt.figure(dpi=300)
index = 990
plt.plot(ye[:-1], H[index][:])
h = tables.open_file('coeff_time_1t_reflection0.05_2MeV_0.1_1/file_%+.2f0.h5' % (0.65*xe[index]))
z = np.linspace(-1,1,100)
k = LG.legval(xe[index], h.root.coeff25[:])
plt.axvline(k, color='red')
h.close()
plt.show()


# In[25]:


print(coeff_pe[1][51],coeff_pe[1][57])
np.interp(rd[51:58],np.array((rd[51], rd[57])), np.array((coeff_pe[1][51],coeff_pe[1][57])))


# In[41]:


def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon(path, order):
    ra = np.arange(0.01, 0.65, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe

order = 5
fit_order = 10
rd, coeff_pe = main_photon('coeff_pe_1t_shell_200000/',order)
rd = np.array(rd)
coeff_pe = np.array(coeff_pe)
#coeff_pe[1][55] = (coeff_pe[1][54] + coeff_pe[1][57])/2 + (coeff_pe[1][54] - coeff_pe[1][57])*1/3 
#coeff_pe[1][56] = (coeff_pe[1][54] + coeff_pe[1][57])/2 + (coeff_pe[1][54] - coeff_pe[1][57])*1/3 
coeff_pe[1][51:58] = np.interp(rd[51:58],np.array((rd[51], rd[57])), np.array((coeff_pe[1][51],coeff_pe[1][57])))
_fit_coeff = np.zeros((order, fit_order))

bd = 0.88
deg = 40
d = np.where(np.abs(rd-np.max(rd)*bd) == np.min(np.abs(rd-np.max(rd)*bd)))

for i in np.arange(order):

    if not i % 2:
        # Legendre coeff

        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                   np.hstack((coeff_pe[i+1], coeff_pe[i+1])),                                                   deg = deg, full = True)

        y1 = np.polynomial.legendre.legval(rd/np.max(rd), B)
        
        C = np.polyfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                           np.hstack((coeff_pe[i+1], coeff_pe[i+1])),                           deg=deg)
        y2 = np.polyval(C, rd/np.max(rd))
    else:
        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                   np.hstack((coeff_pe[i+1], - coeff_pe[i+1])),                                                   deg = deg, full = True)

        y1 = np.polynomial.legendre.legval(rd/np.max(rd), B)
        
        C = np.polyfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                           np.hstack((coeff_pe[i+1], -coeff_pe[i+1])),                           deg=deg)
        y2 = np.polyval(C, rd/np.max(rd))
    plt.figure(num = i+1, dpi = 300)
    plt.plot(rd, coeff_pe[i+1], label='real')
    plt.plot(rd, y1, label = 'Legendre')
    plt.plot(rd, y1, label = 'poly')
    plt.legend()
    plt.show()
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)


# In[32]:


qts = np.array((0.01,0.02,0.05,0.1,0.2,0.5))
order = 5
fit_order = 10
for qt in qts:
    rd, coeff_pe = readtime('./coeff_time_1t_shell_200000_%.2f/' % qt)

    bd = 0.88
    deg = 10
    d = np.where(np.abs(rd-np.max(rd)*bd) == np.min(np.abs(rd-np.max(rd)*bd)))

    for i in np.arange(order):
        plt.figure(num=i+1,dpi=300)
        plt.plot(rd, coeff_pe[i], label=f'qt = {qt}', linestyle='dotted')
        plt.xlabel('Radius/m')
        plt.ylabel('Time Legendre coefficient')
        plt.legend()
        plt.title('Time Legendre coefficients %d-th' % i)
        plt.savefig('coeff_Time_%d.png' % i)
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)


# In[137]:


from numpy.polynomial import legendre as LG
def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
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
def Legendre_coeff(PMT_pos_rep, vertex, cut):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos_rep[:,0])
    # oh, it will use norm in future version
    
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos_rep,axis=1)            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos_rep**2,axis=1))
    else:
        # make r=0 as a boundry, it should be improved
        cos_theta = np.ones(size)

    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)

    print(PMT_pos_rep.shape, x.shape, cos_theta.shape)
    return x, cos_theta

plt.figure(dpi=300)
h = tables.open_file('coeff_time_1t_reflection0.05_2MeV_0.1/file_+0.500.h5')
z = np.linspace(-1,1,100)
k = LG.legval(z, h.root.coeff5[:])
h.close()
plt.plot(z, k, 'r-', label='fit')

h1 = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+0.500Q.h5')
a1 = h1.root.GroundTruth[:]['PulseTime']
a2 = h1.root.GroundTruth[:]['ChannelID']
a3 = h1.root.GroundTruth[:]['EventID']
x = h1.root.TruthData[:]['x']
y = h1.root.TruthData[:]['y']
z = h1.root.TruthData[:]['z']
x1 = np.vstack((x, y, z)).T
counts = np.bincount(a3)
counts = counts[counts!=0]
PMT_pos = ReadPMT()
vertex = np.repeat(x1, counts, axis=0)
PMT_pos_rep = PMT_pos[a2]
tmp_x_p, cos_theta = Legendre_coeff(PMT_pos_rep, vertex, 6)
#plt.scatter(cos_theta, a1, s=0.3,alpha=0.05)
bins = np.linspace(-1,1,100)
index = np.digitize(cos_theta, bins)
tmp = []
for ii, i in enumerate(np.unique(index)):
    tmp.append(np.quantile(a1[index==i],0.12))
plt.plot(bins[1:], np.array(tmp),'.',label='raw data')
plt.xlabel('cos theta')
plt.ylabel('expected pulse time')
plt.title('qt = 0.1')
plt.legend()
plt.show()


# In[119]:


cos_theta[index==1]


# In[ ]:




