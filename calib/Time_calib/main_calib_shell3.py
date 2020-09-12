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
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
from sklearn.ensemble import GradientBoostingRegressor
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
    EventID, ChannelID, flight_time, PMT_pos, cut, LegendreCoeff, qt, ts = args
    y = flight_time
    T_i = np.dot(LegendreCoeff, theta)
    # quantile regression
    # quantile = 0.01
    L0 = Likelihood_quantile(y, T_i, qt, ts, EventID)
    L = L0 + np.sum(np.abs(theta))
    return L0

def Calib1(theta, *args):
    t0, EventID, ChannelID, flight_time, PMT_pos, cut, LegendreCoeff = args
    y = flight_time
    T_i = np.dot(LegendreCoeff, np.hstack((theta[0],t0)))
    # quantile regression
    # quantile = 0.01
    L0 = Likelihood_quantile(y, T_i, theta[1], 26, EventID)
    L = L0 + np.sum(np.abs(theta))
    print(theta, L0)
    return L0

def Likelihood_quantile(y, T_i, qt, ts, EventID):
    #less = T_i[y<T_i] - y[y<T_i]
    #more = y[y>=T_i] - T_i[y>=T_i]
    #R = (1-tau)*np.sum(less) + tau*np.sum(more)
    R = (1-qt)*(T_i-y)*(y<T_i) + (qt)*(y-T_i)*(y>=T_i)
    H,edges =np.histogram(EventID, weights = R, bins = np.hstack((np.unique(EventID), np.max(EventID)+1)))
    Q = np.bincount(EventID)
    #L0 = 0
    L = Q[1:]*np.log(qt*(1-qt)/ts) - H/ts
    L0 = np.nansum(L)
    return - L0

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
    ChannelID = truthtable[:]['ChannelID']
    PETime = truthtable[:]['PETime']
    photonTime = truthtable[:]['photonTime']
    PulseTime = truthtable[:]['PulseTime']
    dETime = truthtable[:]['dETime']
    
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
                ChannelID = ChannelID[~(EventID == ID)]
                PETime = PETime[~(EventID == ID)]
                photonTime = photonTime[~(EventID == ID)]
                PulseTime = PulseTime[~(EventID == ID)]
                dETime = dETime[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z)
    
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
            filename = '%s1t_%s.h5' % (path, radius)
            EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%d.h5' % (path, radius, i)
                EventID1, ChannelID1, PETime1, photonTime1, PulseTime1, dETime1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PETime = np.hstack((PETime,PETime1))
                photonTime = np.hstack((photonTime, photonTime1))
                PulseTime = np.hstack((PulseTime, PulseTime1))
                dETime = np.hstack((dETime, dETime1))

                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z
    
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
        EventID, ChannelID, PETime, photonTime, PulseTime, dETime, xx, yx, zx = readchain(radius, path,'+')
        x1 = np.vstack((xx, yx, zx)).T
        size = np.size(np.unique(EventID))
        total_pe = np.zeros(np.size(PMT_pos[:,0])*size)
        print('total event: %d' % np.size(np.unique(EventID)), flush=True)
        
        input_time = PETime
        # input_time = PulseTime - photonTime
        
        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        tmp = time.time()
        EventNo = np.size(np.unique(EventID))
        PMTNo = np.size(PMT_pos[:,0])
        counts = np.bincount(EventID)
        counts = counts[counts!=0]
        PMT_pos_rep = PMT_pos[ChannelID]
        vertex = np.repeat(x1, counts, axis=0)
        print(PMT_pos_rep.shape, vertex.shape)
        tmp_x_p, cos_theta = Legendre_coeff(PMT_pos_rep, vertex, cut_max)
        print(f'use {time.time() - tmp} s')
        LegendreCoeff = tmp_x_p
        '''
        for cut in np.arange(5,cut_max,1): # just take special values
            X = LegendreCoeff[:,0:cut]
            y = PETime
            alpha = 0.01
            reg = GradientBoostingRegressor(loss='quantile', alpha=alpha,
                            n_estimators=250, max_depth=3,
                            learning_rate=.1, min_samples_leaf=9,
                            min_samples_split=9)
            #reg = TweedieRegressor(power=1, alpha=alpha, link='log', max_iter=1000, tol=1e-6, fit_intercept=False)
            reg.fit(X, y)
            print('%d-th intercept:\n' % cut, reg.intercept_,'\n')
            print('%d-th coeff:\n' % cut, reg.coef_,'\n')
            
            coeff = np.hstack((reg.intercept_,reg.coef_))
            out.create_dataset('coeff' + str(cut), data = coeff)
        '''
        #bins = np.bincount(EventID)
        #A = np.diag((np.ones(bins.size-1)))
        #matrix = np.repeat(A, bins[1:], axis=0)
        #print('???')
        for cut in np.arange(5,cut_max,1):
            theta0 = np.zeros(cut) # initial value
            theta0[0] = np.mean(PETime) - 26
            args = (EventID, ChannelID, PETime, PMT_pos_rep, cut, LegendreCoeff[:,0:cut])
            qts = np.array((0.01,0.02,0.05,0.1,0.15,0.2,0.5))
            tss = np.arange(10,50,5)
            L = np.zeros(len(tss))
            for index1, qt in enumerate(qts):
                result = minimize(Calib, theta0, method='SLSQP',\
                    args = (EventID, ChannelID, PulseTime, PMT_pos_rep, cut, LegendreCoeff[:,0:cut], qt, 25*qt))
                for index2, td in enumerate(tss):
                    ts = td*qt
                    _L = Calib(theta0,\
                            *(EventID, ChannelID, PETime, PMT_pos_rep, cut, LegendreCoeff[:,0:cut], qt, ts))
                    print(f'{qt},{ts}: {result.x}, {_L}')
                    L[index2] = _L
                print(result.x)            
                out.create_dataset('coeff_%d_%.2f' %(cut, qt) , data = result.x)
                out.create_dataset('L_%d_%.2f' % (cut, qt), data = L)

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
