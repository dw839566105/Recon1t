#!/usr/bin/env python
# coding: utf-8

# In[4]:


import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
from scipy.spatial.distance import pdist, squareform
import os


# In[5]:


# example of read 1 file
def main(path,axis):
    for i,file in enumerate(np.arange(-0.6001,0.60,0.05)):

        h = tables.open_file('../%s/1t_%+.3f_%s.h5' % (path, file, axis),'r')
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
        
        
        r = np.sqrt(x**2 + y**2 + z**2)
        index = (r<0.64) & (r>0.01) & (~np.isnan(r))
        H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
        X, Y = np.meshgrid(xedges[1:],yedges[1:])
        plt.figure(dpi=200)
        plt.contourf(X,Y,np.log(np.transpose(H1)+1))
        plt.colorbar()
        plt.xlabel(r'$x^2 + y^2/m^2$')
        plt.ylabel('$z$/m')
        plt.title('axis = %s, radius=%+.2fm' % (axis,file))
        plt.savefig('./fig/Scatter_1MeV%+.2f_%s.pdf' % (file,axis))
        plt.show()
        #index1 = (~index) & (~np.isnan(x2))
        #plt.hist(np.nan_to_num(np.sqrt(data[index1,0]**2 + data[index1,1]**2 + data[index1,2]**2)),bins=100)
        #plt.show()
        plt.figure(dpi=200)
        index2 = index
        #index2 = index
        plt.hist(np.sqrt(x[index2]**2+y[index2]**2+z[index2]**2), bins=100,label='recon')
        plt.axvline(np.abs(file) * np.sqrt(26)/5, color='red', label='real')
        plt.axvline(0.88 * 0.65,color='green',linewidth=1,label='bound')
        plt.xlabel('Recon radius/m')
        plt.ylabel('Num')
        plt.legend()
        recon = eval(axis)
        plt.title('axis = %s, Radius=%+.2fm, std = %.4fm' % (axis, file, np.std(recon[index2]-np.abs(file) * np.sqrt(26)/5)))
        plt.savefig('./fig/HistR_1MeV%+.2f_%s.pdf' % (file,axis))
        #plt.show()
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','x')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','y')
#main('result_1t_2.0MeV_dns_Recon_1t_shell_cubic','z')

main('result_z','z')


# In[ ]:




