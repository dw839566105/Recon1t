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
import time
from scipy.optimize import minimize
from scipy.optimize import rosen_der
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
from scipy.linalg import norm
from numdifftools import Jacobian, Hessian
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
import statsmodels.api as sm


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
    #corr = np.dot(LegendreCoeff, theta) + np.tile(base, (1, np.int(np.size(LegendreCoeff)/np.size(base)/np.size(theta))))[0,:]
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
        cos_theta = np.sum(vertex*PMT_pos_rep,axis=1)\
            /np.sqrt(np.sum(vertex**2, axis=1)*np.sum(PMT_pos_rep**2,axis=1))
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
    Q = h1.root.PETruthData[:]['Q']

    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    if(np.float(np.size(np.unique(EventID)))!= Q.shape[0]/30):
        exit()
    
    return (EventID, Q, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 50):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%sQ.h5' % (path, radius)
            EventID, Q, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%dQ.h5' % (path, radius, i)
                EventID1, Q1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                Q = np.hstack((Q, Q1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, Q, x, y, z
    
def main_Calib(radius, path, fout, cut_max, PMT_pos):
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
        # positive direction
        EventID, Q, xx, yx, zx = readchain(radius, path,'+')
        x1 = np.vstack((xx, yx, zx)).T

        print('total event: %d' % np.size(np.unique(EventID)), flush=True)

        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        tmp = time.time()
        EventNo = np.size(np.unique(EventID))
        PMTNo = np.size(PMT_pos[:,0])
        PMT_pos_rep = np.tile(PMT_pos, (EventNo,1))
        vertex = np.repeat(x1, PMTNo, axis=0)/1e3

        tmp_x_p, cos_theta = Legendre_coeff(PMT_pos_rep, vertex, cut_max)
        print(f'use {time.time() - tmp} s')
        LegendreCoeff = tmp_x_p

        # this part for later EM maybe
        '''
        LegendreCoeff = np.zeros((0,cut_max))           
        for k in np.arange(size):
            LegendreCoeff = np.vstack((LegendreCoeff,Legendre_coeff(PMT_pos,np.array((x[k], y[k], z[k]))/1e3, cut_max)))
       ''' 
        #print('begin get coeff')
        #print('total pe shape:', total_pe.shape)
        #print('Legendre coeff shape:',LegendreCoeff.shape)
        
        for cut in np.arange(2,cut_max,1): # just take special values
            X = LegendreCoeff[:,0:cut]
            y = Q
            print(X.shape, y.shape)
            model = sm.GLM(y, X, family=sm.families.Poisson())
            result = model.fit()
            print(result.summary())
            L = result.aic
            coeff = result.params
            std = result.bse
            print(result.aic)
            print(result.bic)
            out.create_dataset('coeff' + str(cut), data = coeff)
            out.create_dataset('std' + str(cut), data = std)
            out.create_dataset('AIC' + str(cut), data = L)

if len(sys.argv)!=5:
    print("Wront arguments!")
    print("Usage: python main_calib.py 'radius' 'path' outputFileName[.h5] Max_order")
    sys.exit(1)
    
PMT_pos = ReadPMT()
# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3], eval(sys.argv[4]), PMT_pos)
