'''
This is the heart of total PE calib
The PE calib program use Poisson regression, which is equal to decomposition of log expect
since the problem is spherical symmetric, we use Legendre polynomial

    1. each PMT position 'x' and fixed vertex(or vertexes with the same radius r) 'v'
    2. transform 'v' to (0,0,z) as an axis
    3. do the same transform to x, we got the zenith angle 'theta' and azimuth angle 'phi', 'phi' can be ignored
    4. calculate the different Legendre order of 'theta', as a big matrix 'X' (PMT No. * order)
    5. expected 'y' by total pe, got GLM(generalize linear model) 'X beta = g(y)', 'beta' is a vector if coefficient, 'g' is link function, we use log
    6. each r has a set of 'beta' for later analysis
    
The final optimize using scipy.optimize instead of sklear.**regression
'''

import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from scipy.optimize import rosen_der
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
from scipy.linalg import norm
from numdifftools import Jacobian, Hessian

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def LoadBase():
    '''
    # to vanish the PMT difference, just a easy script
    # output: relative different bias
    '''
    path = './base.h5'
    h1 = tables.open_file(path)
    base = h1.root.correct[:]
    h1.close()
    return base
# after using the same PMT this part can be omitted
# base = np.log(LoadBase()) # dont forget log

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

def Calib(theta, *args):
    '''
    # core of this program
    # input: theta: parameter to optimize
    #      *args: include 
          total_pe: [event No * PMT size] * 1 vector ( used to be a 2-d matrix)
          PMT_pos: PMT No * 3
          cut: cut off of Legendre polynomial
          LegendreCoeff: Legendre value of the transformed PMT position (Note, it is repeated to match the total_pe)
    # output: L : likelihood value
    '''
    total_pe, PMT_pos, cut, LegendreCoeff = args
    y = total_pe
    # corr = np.dot(LegendreCoeff, theta) + np.tile(base, (1, np.int(np.size(LegendreCoeff)/np.size(base)/np.size(theta))))[0,:]
    corr = np.dot(LegendreCoeff, theta)
    # Poisson regression as a log likelihood
    # https://en.wikipedia.org/wiki/Poisson_regression
    L0 = - np.sum(np.sum(np.transpose(y)*np.transpose(corr) \
        - np.transpose(np.exp(corr))))
    # how to add the penalty? see
    # http://jmlr.csail.mit.edu/papers/volume17/15-021/15-021.pdf
    # the following 2 number is just a good attempt
    # https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html#sklearn.linear_model.ElasticNet
    rho = 1
    alpha = 0
    L = L0/(2*np.size(y)) + alpha * rho * norm(theta,1) + 1/2* alpha * (1-rho) * norm(theta,2) # elastic net
    return L0

def Legendre_coeff(PMT_pos, vertex, cut):
    '''
    # calulate the Legendre value of transformed X
    # input: PMT_pos: PMT No * 3
          vertex: 'v' 
          cut: cut off of Legendre polynomial
    # output: x: as 'X' at the beginnig    
    
    '''
    size = np.size(PMT_pos[:,0])
    # oh, it will use norm in future version
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos,axis=1)\
            /np.sqrt(np.sum(vertex**2)*np.sum(PMT_pos**2,axis=1))
    else:
        # make r=0 as a boundry, it should be improved
        cos_theta = np.ones(size)
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    return x

def fun_der(x, *args):
    total_pe, PMT_pos, cut, LegendreCoeff = args
    #return rosen_der(Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff)))
    #return Jacobian(Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff)))(x).ravel()
    return Jacobian(Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff)))

def fun_hess(x, *args):
    total_pe, PMT_pos, cut, LegendreCoeff = args
    return Hessian(Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff)))

def MyHessian(x, *args):
    # hession matrix calulation written by dw, for later uncertainty analysis
    # it not be examed
    # what if it is useful one day
    total_pe, PMT_pos, cut, LegendreCoeff= args
    H = np.zeros((len(x),len(x)))
    h = 1e-6
    k = 1e-6
    for i in np.arange(len(x)):
        for j in np.arange(len(x)):
            if (i != j):
                delta1 = np.zeros(len(x))
                delta1[i] = h
                delta1[j] = k
                delta2 = np.zeros(len(x))
                delta2[i] = -h
                delta2[j] = k

                L1 = - Calib(x + delta1, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L2 = - Calib(x - delta1, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L3 = - Calib(x + delta2, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L4 = - Calib(x - delta2, *(total_pe, PMT_pos, cut, LegendreCoeff))
                H[i,j] = (L1+L2-L3-L4)/(4*h*k)
            else:
                delta = np.zeros(len(x))
                delta[i] = h
                L1 = - Calib(x + delta, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L2 = - Calib(x - delta, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L3 = - Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff))
                H[i,j] = (L1+L2-2*L3)/h**2                
    return H

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
    for i in np.arange(0, 1):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%s.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%d.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, x, y, z
    
def main_Calib(radius, path, fout, cut_max):
    '''
    # main program
    # input: radius: %+.3f, 'str' (in makefile, str is default)
    #        path: file storage path, 'str'
    #        fout: file output name as .h5, 'str' (.h5 not included')
    #        cut_max: cut off of Legendre
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    print('begin reading file', flush=True)
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5'
    with h5py.File(fout,'w') as out:
        # read files by table
        # we want to use 6 point on different axis to calib
        # +x, -x, +y, -y, +z, -z or has a little lean (z axis is the worst! use (0,2,10) instead) 
        # In ceter, no positive or negative, the read program should be changed
        # In simulation, the (0,0,0) saved as '*-0.000.h5' is negative
        if(eval(radius) != 0.00):
            # positive direction
            EventIDx, ChannelIDx, xx, yx, zx = readchain('+' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('+' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('+' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            x1 = np.array((xx[0], yx[0], zx[0]))
            y1 = np.array((xy[0], yy[0], zy[0]))
            z1 = np.array((xz[0], yz[0], zz[0]))
            sizex_p = np.size(np.unique(EventIDx))
            sizey_p = np.size(np.unique(EventIDy))
            sizez_p = np.size(np.unique(EventIDz))

            EventID_p = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID_p = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_p = np.hstack((xx, xy, xz))
            y_p = np.hstack((yx, yy, yz))
            z_p = np.hstack((zx, zy, zz))

            # negative direction
            EventIDx, ChannelIDx, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('-' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            x2 = np.array((xx[0], yx[0], xz[0]))
            y2 = np.array((xy[0], yy[0], zy[0]))
            z2 = np.array((xz[0], yz[0], zz[0]))

            EventID_n = np.hstack((EventIDx, EventIDy, EventIDz)) + np.max(EventID_p)
            ChannelID_n = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_n = np.hstack((xx, xy, xz))
            y_n = np.hstack((yx, yy, yz))
            z_n = np.hstack((zx, zy, zz))

            # written total_pe into a column
            sizex_n = np.size(np.unique(EventIDx))
            sizey_n = np.size(np.unique(EventIDy))
            sizez_n = np.size(np.unique(EventIDz))

            # gather
            EventID = np.hstack((EventID_p, EventID_n))
            ChannelID = np.hstack((ChannelID_p, ChannelID_n))
            x = np.hstack((x_p, x_n))
            y = np.hstack((y_p, y_n))
            z = np.hstack((z_p, z_n))         
            size = np.size(np.unique(EventID))
        else:
            EventIDx, ChannelIDx, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('-' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            x1 = np.array((xx[0], yx[0], xz[0]))
            y1 = np.array((xy[0], yy[0], zy[0]))
            z1 = np.array((xz[0], yz[0], zz[0]))
            sizex_p = np.size(np.unique(EventIDx))
            sizey_p = np.size(np.unique(EventIDy))
            sizez_p = np.size(np.unique(EventIDz))

            EventID_p = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID_p = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_p = np.hstack((xx, xy, xz))
            y_p = np.hstack((yx, yy, yz))
            z_p = np.hstack((xz, yz, zz))
            
            EventID = EventID_p
            ChannelID = ChannelID_p
            x = x_p
            y = y_p
            z = z_p 
            size = np.size(np.unique(EventID))

        total_pe = np.zeros(np.size(PMT_pos[:,0])*size)
        vertex = np.zeros((3,np.size(PMT_pos[:,0])*size))
        print('total event: %d' % np.size(np.unique(EventID)), flush=True)
        
        for k_index, k in enumerate(np.unique(EventID)):
            if not k_index % 1e4:
                print('preprocessing %d-th event' % k_index, flush=True)
            hit = ChannelID[EventID == k]
            tabulate = np.bincount(hit)
            event_pe = np.zeros(np.size(PMT_pos[:,0]))
            # tabulate begin with 0
            event_pe[0:np.size(tabulate)] = tabulate
            total_pe[(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = event_pe
            # although it will be repeated, mainly for later EM:
            # vertex[0,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = x[k_index]
            # vertex[1,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = y[k_index]
            # vertex[2,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = z[k_index]
            total_pe[(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = event_pe

        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        
        if(eval(radius) != 0.00):
            tmp_x_p = Legendre_coeff(PMT_pos, x1/1e3, cut_max)
            tmp_x_p = np.tile(tmp_x_p, (sizex_p,1))
            tmp_y_p = Legendre_coeff(PMT_pos, y1/1e3, cut_max)
            tmp_y_p = np.tile(tmp_y_p, (sizey_p,1))
            tmp_z_p = Legendre_coeff(PMT_pos, z1/1e3, cut_max)
            tmp_z_p = np.tile(tmp_z_p, (sizez_p,1))
            tmp_x_n = Legendre_coeff(PMT_pos, x2/1e3, cut_max)
            tmp_x_n = np.tile(tmp_x_n, (sizex_n,1))
            tmp_y_n = Legendre_coeff(PMT_pos, y2/1e3, cut_max)
            tmp_y_n = np.tile(tmp_y_n, (sizey_n,1))
            tmp_z_n = Legendre_coeff(PMT_pos, z2/1e3, cut_max)
            tmp_z_n = np.tile(tmp_z_n, (sizez_n,1))              

            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p, tmp_x_n, tmp_y_n, tmp_z_n))
            #print('files shape:', tmp_x_p.shape, tmp_y_p.shape, tmp_z_p.shape, tmp_x_n.shape, tmp_y_n.shape, tmp_z_n.shape)
        else:
            tmp_x_p = Legendre_coeff(PMT_pos, x1/1e3, cut_max)
            tmp_x_p = np.tile(tmp_x_p, (sizex_p,1))
            tmp_y_p = Legendre_coeff(PMT_pos, y1/1e3, cut_max)
            tmp_y_p = np.tile(tmp_y_p, (sizey_p,1))
            tmp_z_p = Legendre_coeff(PMT_pos, z1/1e3, cut_max)
            tmp_z_p = np.tile(tmp_z_p, (sizez_p,1))
            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p))
            #print('files shape:', tmp_x_p.shape, tmp_y_p.shape, tmp_z_p.shape)  
        # this part for later EM maybe
        '''
        LegendreCoeff = np.zeros((0,cut_max))           
        for k in np.arange(size):
            LegendreCoeff = np.vstack((LegendreCoeff,Legendre_coeff(PMT_pos,np.array((x[k], y[k], z[k]))/1e3, cut_max)))
       ''' 
        #print('begin get coeff')
        #print('total pe shape:', total_pe.shape)
        #print('Legendre coeff shape:',LegendreCoeff.shape)
        
        for cut in np.arange(1,cut_max,1): # just take special values
            if(cut==1):
                theta0 = np.zeros(cut) # initial value
                theta0[0] = 0.8 + np.log(2) # intercept is much more important
            else:
                theta0 = np.hstack((result.x, 0))
            #result = minimize(Calib, theta0, method='BFGS', jac=fun_der, hess=fun_hess, args = (total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut])) 
            result = minimize(Calib, theta0, method='SLSQP', args = (total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut])) 
            #result = minimize(Calib, theta0, method='BFGS', jac=fun_der, args = (total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut])) 
            record = np.array(result.x, dtype=float)
            #print(result.success)
            #H = Hessian(Calib)(result.x, *(total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut]))
            #print(H)
            #H1 = MyHessian(result.x, *(total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut]))
            #print(H1)
            #print(np.linalg.pinv(H))
            #print(result.hess_inv)
            # hessian not tested
            # H = hessian(result.x, *(total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut]))
            # H_I = np.linalg.pinv(np.matrix(H))
            vs_x = np.reshape(total_pe[0:sizex_p*30],(-1,30), order='C')
            mean_x = np.mean(vs_x, axis=0)
            vs_y = np.reshape(total_pe[sizex_p*30:sizex_p*30+sizey_p*30],(-1,30), order='C')
            mean_y = np.mean(vs_y, axis=0)
            vs_z = np.reshape(total_pe[sizex_p*30+sizey_p*30:sizex_p*30+sizey_p*30+sizez_p*30],(-1,30), order='C')
            mean_z = np.mean(vs_z, axis=0)
            
            vs_xx = np.reshape(total_pe[sizex_p*30+sizey_p*30+sizez_p*30:sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30],(-1,30), order='C')
            mean_xx = np.mean(vs_xx, axis=0)
            vs_yy = np.reshape(total_pe[sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30:sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+sizey_n*30],(-1,30), order='C')
            mean_yy = np.mean(vs_yy, axis=0)
            vs_zz = np.reshape(total_pe[sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+sizey_n*30:sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+sizey_n*30+sizez_n*30],(-1,30), order='C')
            mean_zz = np.mean(vs_zz, axis=0)


            args = (total_pe, PMT_pos, cut)
            # following should be optimize...
            predict_x = [];
            predict_x.append(np.exp(np.dot(LegendreCoeff[0:30,0:cut], result.x)))
            predict_x = np.transpose(predict_x)
            predict_y = [];
            predict_y.append(np.exp(np.dot(LegendreCoeff[sizex_p*30:sizex_p*30+30,0:cut], result.x)))
            predict_y = np.transpose(predict_y)
            predict_z = [];
            predict_z.append(np.exp(np.dot(LegendreCoeff[sizex_p*30+sizey_p*30:sizex_p*30+sizey_p*30+30,0:cut], result.x)))
            predict_z = np.transpose(predict_z)
            predict_xx = [];
            predict_xx.append(np.exp(np.dot(LegendreCoeff[sizex_p*30+sizey_p*30+sizez_p*30:sizex_p*30+sizey_p*30+sizez_p*30+30,0:cut], result.x)))
            predict_xx = np.transpose(predict_xx)
            predict_yy = [];
            predict_yy.append(np.exp(np.dot(LegendreCoeff[sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30:sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+30,0:cut], result.x)))
            predict_yy = np.transpose(predict_yy)
            predict_zz = [];
            predict_zz.append(np.exp(np.dot(LegendreCoeff[sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+sizey_n*30:sizex_p*30+sizey_p*30+sizez_p*30+sizex_n*30+sizey_n*30+30,0:cut], result.x)))
            predict_zz = np.transpose(predict_zz)
            '''
            print('%d-th coeff:\n' % cut, record,'\n')
            print('x')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_x[:,0],'\n')
            print('Mean hit:\n', mean_x,'\n')      
            print('y')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_y[:,0],'\n')
            print('Mean hit:\n', mean_y,'\n')      
            print('z')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_z[:,0],'\n')
            print('Mean hit:\n', mean_z,'\n')      
            print('-x')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_xx[:,0],'\n')
            print('Mean hit:\n', mean_xx,'\n')      
            print('-y')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_yy[:,0],'\n')
            print('Mean hit:\n', mean_yy,'\n')      
            print('-z')
            print('='*80)
            print('%d-th predict:\n' % cut, predict_zz[:,0],'\n')
            print('Mean hit:\n', mean_zz,'\n')      
            print('Saving file...')
            '''
            ##########
            y = total_pe
            corr = np.dot(LegendreCoeff[:,0:cut], result.x)
            L0 = np.transpose(y)*np.transpose(corr) \
                - np.transpose(np.exp(corr))
            
            corr_x_exp = np.tile(mean_x, (1,sizex_p))
            corr_y_exp = np.tile(mean_y, (1,sizey_p))
            corr_z_exp = np.tile(mean_z, (1,sizez_p))
            corr_xx_exp = np.tile(mean_xx, (1,sizex_n))
            corr_yy_exp = np.tile(mean_yy, (1,sizey_n))
            corr_zz_exp = np.tile(mean_zz, (1,sizez_n))
            corr_new = []
            corr_new.append(np.hstack((corr_x_exp, corr_y_exp, corr_z_exp, corr_xx_exp, corr_yy_exp, corr_zz_exp)))
            corr_new = np.log(np.array(corr_new).T)
            
            #print(corr_new, corr)
            
            L1 = np.transpose(y)*np.transpose(corr_new) \
                - np.transpose(np.exp(corr_new))
            
            #print(np.sum(L1[0][0][0:sizex_p*30:30]))
            #print(np.sum(L1[0][0][1:sizex_p*30:30]))
            aaa = np.reshape(L0[0:sizex_p*30],(30,-1),order='F')
            print('='*30,' %d ' % cut, '='*30, flush=True)
            bbb = np.reshape(L1[0, 0, 0:sizex_p*30],(30,-1),order='F')
            #print(np.sum(aaa, axis=1))
            print(result.success)
            aic = 2*(cut+30) - 2*np.sum(aaa)
            sup = -2 * np.sum(bbb)
            print(aic, sup, flush=True)

            aic1 = 2*(cut+180)-2*np.sum(L0)
            sup1 = -2 * np.sum(bbb)
            #print((np.sum(aaa, axis=1)).shape)
            print(aic1)
            '''
            print(L0,L1)
            print(-2 *(np.abs(L0-L1)))

            print(np.nansum(y*np.log(y/np.exp(corr.T))-(y-np.exp(corr.T))))
            print(np.nansum(y*np.log(y/np.exp(corr_new.T))-(y-np.exp(corr_new.T))))
            '''
            #print('here',cut,'\t',2*cut+2*L0)
            # chi2square             
            # chi2sq = 2*np.sum(- total_pe + predict + np.nan_to_num(total_pe*np.log(total_pe/predict)), axis=1)/(np.max(EventID)-30)
            out.create_dataset('coeff' + str(cut), data = record)
            out.create_dataset('mean' + str(cut), data = mean_x)
            out.create_dataset('predict' + str(cut), data = predict_x)
            out.create_dataset('AIC' + str(cut), data = aic)
            #out.create_dataset('rate' + str(cut), data = np.size(total_pe)/30)
            #out.create_dataset('hess' + str(cut), data = H)
            out.create_dataset('success' + str(cut), data = result.success)
            #out.create_dataset('chi' + str(cut), data = chi2sq)

if len(sys.argv)!=5:
    print("Wront arguments!")
    print("Usage: python main_calib.py 'radius' 'path' outputFileName[.h5] Max_order")
    sys.exit(1)
    
PMT_pos = ReadPMT()
# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3], eval(sys.argv[4]))
