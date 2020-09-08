import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py
from scipy.spatial.distance import pdist, squareform
import os, sys

def main(path1,path2,axis):
    for i,file in enumerate(np.arange(-0.6001,0.60,0.05)):

        h = tables.open_file('%s/1t_%+.3f_%s.h5' % (path1, file, axis),'r')
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
        h.close()
        x = data[(s1 * s2)!=0,0]
        y = data[(s1 * s2)!=0,1]
        z = data[(s1 * s2)!=0,2]
        
        fig = plt.figure(num=i*2+1, dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.40)
        ax = plt.subplot(2,2,1)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        r = np.sqrt(x**2 + y**2 + z**2)
        index = (r<0.64) & (r>0.01) & (~np.isnan(r))
        H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
        X, Y = np.meshgrid(xedges[1:],yedges[1:])
        plt.contourf(X,Y,np.log(np.transpose(H1)+1))
        cbar = plt.colorbar()
        cbar.set_ticks([])
        
        plt.xlabel(r'$x^2 + y^2/m^2$')
        plt.ylabel('z/m')
        ax = plt.subplot(2,2,3)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        r = np.sqrt(x**2 + y**2 + z**2)
        index = (r<0.60) & (r>0.01) & (~np.isnan(r))
        H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
        X, Y = np.meshgrid(xedges[1:],yedges[1:])
        plt.contourf(X,Y,np.log(np.transpose(H1)+1))
        cbar = plt.colorbar()
        cbar.set_ticks([])
        plt.xlabel(r'$x^2 + y^2/m^2$')
        plt.ylabel('z/m')

        fig = plt.figure(num=i*2+2, dpi=300)
        plt.subplots_adjust(wspace=0.4, hspace=0.55)
        ax = plt.subplot(2,2,1)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = (r<0.64) & (r>0.01) & (~np.isnan(r))
        plt.hist(np.sqrt(x[index]**2+y[index]**2+z[index]**2), bins=100,label='recon')
        plt.axvline(np.abs(file), color='red', label='real')
        plt.axvline(0.88 * 0.65,color='green',linewidth=1,label='bound')
        plt.xlabel('Recon radius/m')
        plt.ylabel('Num')
        recon = eval(axis)
        plt.title('std: %.3f m, bias:%.3f m ' % (np.std(recon[index]-file), np.mean(np.abs(recon[index]-file))))
        plt.legend()
        
        ax = plt.subplot(2,2,3)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = (r<0.60) & (r>0.01) & (~np.isnan(r))
        plt.hist(np.sqrt(x[index]**2+y[index]**2+z[index]**2), bins=100,label='recon')
        plt.axvline(np.abs(file), color='red', label='real')
        plt.axvline(0.88 * 0.65,color='green',linewidth=1,label='bound')
        plt.xlabel('Recon radius/m')
        plt.ylabel('Num')
        recon = eval(axis)
        plt.title('std: %.3f m, bias:%.3f m ' % (np.std(recon[index]-file), np.mean(np.abs(recon[index]-file))))
        plt.legend()
        ###############################
        ###############################
        ###############################
        h = tables.open_file('%s/1t_%+.3f_%s.h5' % (path2, file, axis),'r')
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
        h.close()
        x = data[(s1 * s2)!=0,0]
        y = data[(s1 * s2)!=0,1]
        z = data[(s1 * s2)!=0,2]
        
        fig = plt.figure(num=i*2+1)
        ax = plt.subplot(2,2,2)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        r = np.sqrt(x**2 + y**2 + z**2)
        index = (r<0.64) & (r>0.01) & (~np.isnan(r))
        H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
        X, Y = np.meshgrid(xedges[1:],yedges[1:])
        plt.contourf(X,Y,np.log(np.transpose(H1)+1))
        cbar = plt.colorbar()
        cbar.set_ticks([])
        plt.xlabel(r'$x^2 + y^2/m^2$')
        plt.ylabel('z/m')
        
        ax = plt.subplot(2,2,4)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        r = np.sqrt(x**2 + y**2 + z**2)
        index = (r<0.60) & (r>0.01) & (~np.isnan(r))
        H1, xedges, yedges = np.histogram2d(x[index]**2 + y[index]**2, z[index], bins=50)
        X, Y = np.meshgrid(xedges[1:],yedges[1:])
        plt.contourf(X,Y,np.log(np.transpose(H1)+1))
        cbar = plt.colorbar()
        cbar.set_ticks([])
        plt.xlabel(r'$x^2 + y^2/m^2$')
        plt.ylabel('z/m')
        fig.suptitle(f'axis=%s, radius=%.2f m ' % (axis, file))
        plt.savefig('./fig/Scatter_2MeV%+.2f_%s.pdf' % (file,axis)) 
        plt.close()
        
        
        fig = plt.figure(num=i*2+2)
        ax = plt.subplot(2,2,2)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = (r<0.64) & (r>0.01) & (~np.isnan(r))
        plt.hist(np.sqrt(x[index]**2+y[index]**2+z[index]**2), bins=100,label='recon')
        plt.axvline(np.abs(file), color='red', label='real')
        plt.axvline(0.88 * 0.65,color='green',linewidth=1,label='bound')
        plt.xlabel('Recon radius/m')
        plt.ylabel('Num')
        plt.legend()
        recon = eval(axis)
        plt.title('std: %.3f m, bias:%.3f m ' % (np.std(recon[index]-file), np.mean(np.abs(recon[index]-file))))
        
        ax = plt.subplot(2,2,4)
        ax.tick_params(axis='both', which='minor', labelsize=8)
        index = (r<0.60) & (r>0.01) & (~np.isnan(r))
        plt.hist(np.sqrt(x[index]**2+y[index]**2+z[index]**2), bins=100,label='recon')
        plt.axvline(np.sqrt(26)/5*np.abs(file), color='red', label='real')
        plt.axvline(0.88 * 0.65,color='green',linewidth=1,label='bound')
        plt.xlabel('Recon radius/m')
        plt.ylabel('Num')
        plt.legend()
        recon = eval(axis)
        plt.title('std: %.3f m, bias:%.3f m ' % (np.std(recon[index]-file), np.mean(np.abs(recon[index]-file))))        
        fig.suptitle(f'axis=%s, radius=%.2f m ' % (axis, file))
        plt.savefig('./fig/HistR_2MeV%+.2f_%s.pdf' % (file,axis))
        plt.close()
path1 = sys.argv[1]
path2 = sys.argv[2]
axis = sys.argv[3]

main(path1,path2, axis)