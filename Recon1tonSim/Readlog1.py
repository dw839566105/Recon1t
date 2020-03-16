from scipy import interpolate
import numpy as np
import matplotlib.pyplot as plt
import os
import tables

def coeff():
    radius = np.arange(-0.65,0.66,0.01)
    new_cell = np.zeros([len(radius), 5])
    filename = '/home/douwei/coeff.h5'
    h = tables.open_file(filename,'r')
    recondata = h.root.coeff
    x = np.arange(-0.60,0.61,0.01)
    coeff_x = np.array(recondata[:])
    #x = np.array(x_axis[:])
    h.close()
    for j in np.arange(len(coeff_x[:,0])):
        x_left1 = np.min(x)-0.01
        x_left2 = np.min(x)-0.02
        x_left3 = -1
            
        x_right1 = np.max(x)+0.01
        x_right2 = np.max(x)+0.02
        x_right3 = 1
        xx = np.hstack((x_left3, x_left2, x_left1, x, x_right1, x_right2, x_right3))
        yy = np.hstack((0, 0, 0, coeff_x[j,:], 0, 0, 0))
        print(coeff_x.shape, len(coeff_x[:,0]))
        print(xx.shape,yy.shape)
        f = interpolate.interp1d(xx, yy, kind='cubic')            
        new_cell[:,j] = f(radius)
    return radius, new_cell

if __name__ == '__main__':
    radius, new_cell = coeff()
    print(new_cell)
    '''
    for i in range(len(new_cell[0,:,0])):
        plt.figure(i)
        plt.plot(radius, new_cell[:,i,:])       
        plt.xlabel('radius/m')
        plt.ylabel('coeff value')
        plt.legend(['0.8','1.0','1.2','1.5','1.8','2.0'])
        plt.title((str(i) + '-th value'))
        plt.savefig(('./fig'+str(i)+'.jpg'))
    '''
