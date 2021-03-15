#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
from scipy.spatial.distance import pdist, squareform
import os


# In[18]:


# example of read 1 file
def main(path,axis):
    
    x_recon = np.empty(0)
    y_recon = np.empty(0)
    z_recon = np.empty(0)
    x_truth = np.empty(0)
    y_truth = np.empty(0)
    z_truth = np.empty(0)
    
    for i,file in enumerate(np.arange(0,0.65,0.01)):
        h = tables.open_file('../%s/1t_%+.3f_%s.h5' % (path, file, axis),'r')
        recondata = h.root.Recon
        E1 = recondata[:]['E_sph_in']
        x1 = recondata[:]['x_sph_in']
        y1 = recondata[:]['y_sph_in']
        z1 = recondata[:]['z_sph_in']

        data = np.zeros((np.size(x1),3))

        data[:,0] = x1
        data[:,1] = y1
        data[:,2] = z1

        xt = 0
        yt = 0
        zt = 0
        if(axis=='x'):
            xt = file
        elif(axis=='y'):
            yt = file
        elif(axis=='z'):
            zt = file
        else:
            print(haha)
        x = x1
        y = y1
        z = z1

        x_recon = np.hstack((x_recon, x))
        y_recon = np.hstack((y_recon, y))
        z_recon = np.hstack((z_recon, z))
        x_truth = np.hstack((x_truth, xt*np.ones_like(x)))
        y_truth = np.hstack((y_truth, yt*np.ones_like(y)))
        z_truth = np.hstack((z_truth, zt*np.ones_like(z)))
    r_recon = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
    r_truth = np.sqrt(x_truth**2 + y_truth**2 + z_truth**2)
    plt.figure(dpi=300)
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap('jet', 256)
    newcolors = viridis(np.linspace(0, 1, 65536))
    pink = np.array([1, 1, 1, 1])
    newcolors[:25, :] = pink
    newcmp = ListedColormap(newcolors)
    if(axis=='x'):
        plt.hist2d(x_truth, x_recon, bins=(np.arange(0.005,0.655,0.01), np.arange(0.005,0.655,0.01)), cmap=newcmp)
    elif(axis=='z'):    
        plt.hist2d(z_truth, z_recon, bins=(np.arange(0.005,0.655,0.01), np.arange(0.005,0.655,0.01)), cmap=newcmp)
    plt.colorbar()
    plt.xlabel('Truth r/m')
    plt.ylabel('Recon r/m')
    plt.title('Vertex recon by SH on %s axis' % axis)
    plt.show()
    return x_recon, y_recon, z_recon, x_truth, y_truth, z_truth
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','x')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','y')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','z')

x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','x')
#x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_new2','y')
x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','z')
#main('result_1t_2.0MeV_dns_Recon_1t_10','y')
#main('result_1t_2.0MeV_dns_Recon_1t_10','z')


# In[9]:


h = tables.open_file('../result_1t_point_axis_Recon_1t_da_new/1t_+0.550_z.h5','r')
recondata = h.root.Recon
E1 = recondata[:]['E_sph_in']
x1 = recondata[:]['x_sph_in']
y1 = recondata[:]['y_sph_in']
z1 = recondata[:]['z_sph_in']
h.close()

r = np.sqrt(x1**2+y1**2+z1**2)
plt.hist(r, bins=np.arange(0,0.65,0.01))
plt.show()


# In[16]:


h = tables.open_file('../result_1t_point_axis_Recon_1t_da_new/1t_+0.520_x.h5','r')
recondata = h.root.Recon
E1 = recondata[:]['E_sph_in']
x1 = recondata[:]['x_sph_in']
y1 = recondata[:]['y_sph_in']
z1 = recondata[:]['z_sph_in']
h.close()

r = np.sqrt(x1**2+y1**2+z1**2)
plt.hist(r, bins=np.arange(0,0.65,0.01))
plt.show()


# In[17]:


np.arange(0,0.65,0.01)


# In[45]:


import scipy.special
file = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+0.560_zQ.h5')
data = file.root.PETruthData[:]['Q'].reshape(-1, 30)

# np.exp(-lambda)*(lambda**n)/n!

k = np.nansum(-data + data*np.log(data) - np.log(scipy.special.factorial(data)), axis=1)


# In[52]:


import scipy.special

A = np.zeros(0)
B = np.zeros(0)
for i in np.arange(0,0.65,0.01):
    file = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_zQ.h5' % i)
    data = file.root.PETruthData[:]['Q'].reshape(-1, 30)

    # np.exp(-lambda)*(lambda**n)/n!

    k = np.nansum(-data + data*np.log(data) - np.log(scipy.special.factorial(data)), axis=1)
    B = np.hstack((B, k))
    A = np.hstack((A, np.ones_like(k)*i))
plt.hist2d(A, B, bins=(65,30))
plt.show()


# In[53]:


import scipy.special

A = np.zeros(0)
B = np.zeros(0)
for i in np.arange(0,0.65,0.01):
    file = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_xQ.h5' % i)
    data = file.root.PETruthData[:]['Q'].reshape(-1, 30)

    # np.exp(-lambda)*(lambda**n)/n!

    k = np.nansum(-data + data*np.log(data) - np.log(scipy.special.factorial(data)), axis=1)
    B = np.hstack((B, k))
    A = np.hstack((A, np.ones_like(k)*i))
plt.hist2d(A, B, bins=(65,30))
plt.show()


# In[73]:


# example of read 1 file

def main(path,axis):
    
    x_recon = np.empty(0)
    y_recon = np.empty(0)
    z_recon = np.empty(0)
    x_truth = np.empty(0)
    y_truth = np.empty(0)
    z_truth = np.empty(0)
    
    A = np.zeros(0)
    B = np.zeros(0)

    for i,file in enumerate(np.arange(0.01,0.65,0.01)):
        h = tables.open_file('../%s/1t_%+.3f_%s.h5' % (path, file, axis),'r')
        recondata = h.root.Recon
        E1 = recondata[:]['E_sph_in']
        x1 = recondata[:]['x_sph_in']
        y1 = recondata[:]['y_sph_in']
        z1 = recondata[:]['z_sph_in']

        data = np.zeros((np.size(x1),3))

        data[:,0] = x1
        data[:,1] = y1
        data[:,2] = z1

        xt = 0
        yt = 0
        zt = 0
        if(axis=='x'):
            xt = file
        elif(axis=='y'):
            yt = file
        elif(axis=='z'):
            zt = file
        else:
            print(haha)
        x = x1
        y = y1
        z = z1

        x_recon = np.hstack((x_recon, x))
        y_recon = np.hstack((y_recon, y))
        z_recon = np.hstack((z_recon, z))
        x_truth = np.hstack((x_truth, xt*np.ones_like(x)))
        y_truth = np.hstack((y_truth, yt*np.ones_like(y)))
        z_truth = np.hstack((z_truth, zt*np.ones_like(z)))
        
        file = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_%sQ.h5' % (file, axis))
        data = file.root.PETruthData[:]['Q'].reshape(-1, 30)
        print(np.sum(np.sum(data,axis=1)<10))
        print(x.shape, xt.shape, data.shape[0])

        # np.exp(-lambda)*(lambda**n)/n!

        k = np.nansum(-data + data*np.log(data) - np.log(scipy.special.factorial(data)), axis=1)
        B = np.hstack((B, k))
        A = np.hstack((A, np.ones_like(k)*i))
    print(B.shape, x_truth.shape)
    r_recon = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
    r_truth = np.sqrt(x_truth**2 + y_truth**2 + z_truth**2)
    plt.figure(dpi=300)
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap('jet', 256)
    newcolors = viridis(np.linspace(0, 1, 65536))
    pink = np.array([1, 1, 1, 1])
    newcolors[:25, :] = pink
    newcmp = ListedColormap(newcolors)
    if(axis=='x'):
        plt.hist2d(x_truth, B, bins=(65,30), cmap=newcmp)
    elif(axis=='z'):    
        plt.hist2d(z_truth, B, bins=(65,30), cmap=newcmp)
    plt.colorbar()
    plt.xlabel('Truth r/m')
    plt.ylabel('Recon r/m')
    plt.title('Vertex recon by SH on %s axis' % axis)
    plt.show()
    return x_recon, y_recon, z_recon, x_truth, y_truth, z_truth
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','x')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','y')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','z')

x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','x')
#x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_new2','y')
x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','z')
#main('result_1t_2.0MeV_dns_Recon_1t_10','y')
#main('result_1t_2.0MeV_dns_Recon_1t_10','z')


# In[64]:


file = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/point_axis/1t_+%.3f_xQ.h5' % i)


# In[70]:


np.unique(file.root.GroundTruth[:]['EventID']).shape


# In[71]:


# example of read 1 file
def main(path,axis):
    
    x_recon = np.empty(0)
    y_recon = np.empty(0)
    z_recon = np.empty(0)
    x_truth = np.empty(0)
    y_truth = np.empty(0)
    z_truth = np.empty(0)
    
    for i,file in enumerate(np.arange(0,0.65,0.01)):
        h = tables.open_file('../%s/1t_%+.3f_%s.h5' % (path, file, axis),'r')
        recondata = h.root.Recon
        E1 = recondata[:]['E_sph_in']
        x1 = recondata[:]['x_sph_in']
        y1 = recondata[:]['y_sph_in']
        z1 = recondata[:]['z_sph_in']

        data = np.zeros((np.size(x1),3))

        data[:,0] = x1
        data[:,1] = y1
        data[:,2] = z1

        xt = 0
        yt = 0
        zt = 0
        if(axis=='x'):
            xt = file
        elif(axis=='y'):
            yt = file
        elif(axis=='z'):
            zt = file
        else:
            print(haha)
        x = x1
        y = y1
        z = z1

        x_recon = np.hstack((x_recon, x))
        y_recon = np.hstack((y_recon, y))
        z_recon = np.hstack((z_recon, z))
        x_truth = np.hstack((x_truth, xt*np.ones_like(x)))
        y_truth = np.hstack((y_truth, yt*np.ones_like(y)))
        z_truth = np.hstack((z_truth, zt*np.ones_like(z)))
    r_recon = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
    r_truth = np.sqrt(x_truth**2 + y_truth**2 + z_truth**2)
    plt.figure(dpi=300)
    from matplotlib import cm
    from matplotlib.colors import ListedColormap, LinearSegmentedColormap
    viridis = cm.get_cmap('jet', 256)
    newcolors = viridis(np.linspace(0, 1, 65536))
    pink = np.array([1, 1, 1, 1])
    newcolors[:25, :] = pink
    newcmp = ListedColormap(newcolors)
    if(axis=='x'):
        plt.hist2d(x_truth, x_recon, bins=(np.arange(0.005,0.655,0.01), np.arange(0.005,0.655,0.01)), cmap=newcmp)
    elif(axis=='z'):    
        plt.hist2d(z_truth, z_recon, bins=(np.arange(0.005,0.655,0.01), np.arange(0.005,0.655,0.01)), cmap=newcmp)
    plt.colorbar()
    plt.xlabel('Truth r/m')
    plt.ylabel('Recon r/m')
    plt.title('Vertex recon by SH on %s axis' % axis)
    plt.show()
    return x_recon, y_recon, z_recon, x_truth, y_truth, z_truth
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','x')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','y')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','z')

x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','x')
#x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_new2','y')
x_recon, y_recon, z_recon, x_truth, y_truth, z_truth = main('result_1t_point_axis_Recon_1t_da_new','z')
#main('result_1t_2.0MeV_dns_Recon_1t_10','y')
#main('result_1t_2.0MeV_dns_Recon_1t_10','z')


# In[132]:


# example of read 1 file
def main(path):
    
    x_recon = np.empty(0)
    y_recon = np.empty(0)
    z_recon = np.empty(0)
    E_recon = np.empty(0)
    x_truth = np.empty(0)
    y_truth = np.empty(0)
    z_truth = np.empty(0)
    
    for i,file in enumerate(np.arange(0.2,0.3,0.01)):
        try:
            h = tables.open_file('../%s/1t_%+.3f.h5' % (path, file),'r')
            recondata = h.root.Recon
            E = recondata[:]['E_sph_in']
            x = recondata[:]['x_sph_in']
            y = recondata[:]['y_sph_in']
            z = recondata[:]['z_sph_in']

            data = np.zeros((np.size(x1),3))

            data[:,0] = x1
            data[:,1] = y1
            data[:,2] = z1

            #xt = recondata[:]['x_truth']
            #yt = recondata[:]['y_truth']
            #zt = recondata[:]['z_truth']

            x_recon = np.hstack((x_recon, x))
            y_recon = np.hstack((y_recon, y))
            z_recon = np.hstack((z_recon, z))
            E_recon = np.hstack((E_recon, E))
            #x_truth = np.hstack((x_truth, xt))
            #y_truth = np.hstack((y_truth, yt))
            #z_truth = np.hstack((z_truth, zt))
        except:
            pass
    r_recon = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
    return x_recon, y_recon, z_recon, E_recon
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','x')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','y')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','z')

x_recon, y_recon, z_recon, E_recon = main('result_1t_point_10_Recon_1t_da_new_PE')


# In[133]:


r = np.sqrt(x_recon**2 + y_recon**2 + z_recon**2)
index = (r > 0.2) & (r < 0.3) & (E_recon<3) & (E_recon>1)
#index = (E_recon<2) & (E_recon>1)
plt.figure(dpi=300)
plt.hist2d(np.arctan2(y_recon[index], x_recon[index]), z_recon[index]/r[index], bins=80)
plt.colorbar()
plt.xlabel(r'$\phi$')
plt.ylabel(r'$\cos\theta$')
plt.savefig('tp_map.png')
plt.show()


# In[112]:


plt.hist(r,bins=30)
plt.show()


# In[116]:


plt.hist(z_recon[r>0.01],bins=30)
plt.show()


# In[ ]:




