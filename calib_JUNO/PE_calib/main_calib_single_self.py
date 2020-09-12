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
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def ReadPMT(file):
    f = open(file)
    line = f.readline()
    data_list = [] 
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)
    return PMT_pos

def LoadBase():
    '''
    # to vanish the PMT difference, just a easy script
    # output: relative different bias
    '''
    h = tables.open_file('/mnt/stage/harmonic/unify/cal/down/z0.h5')
    EID = h.root.PETruth[:]['EventID']
    CID = h.root.PETruth[:]['ChannelID']
    PEList_down = np.zeros((np.size(np.unique(EID)),\
        (np.max(CID) - np.min(CID))+1))
    for i in np.unique(EID):
        A = np.bincount(CID[EID==i])
        PEList_down[i, np.min(CID[EID==i]):np.max(CID[EID==i])+1] = A
    h.close()
    
    h = tables.open_file('/mnt/stage/harmonic/unify/cal/up/z0.h5')
    EID = h.root.PETruth[:]['EventID']
    CID = h.root.PETruth[:]['ChannelID']
    PEList_up = np.zeros((np.size(np.unique(EID)), \
        (np.max(CID) - np.min(CID))+1))
    for i in np.arange(10):
        A = np.bincount(CID[EID==i])
        PEList_up[i, np.min(CID[EID==i]):np.max(CID[EID==i])+1] = A
    h.close()
    baseA = np.mean(PEList_up[:,PMT_A[:,0].astype('int')], axis=0)
    baseB = np.mean(PEList_up[:,PMT_B[:,0].astype('int')], axis=0)
    return baseA, baseB



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

    corr = np.dot(LegendreCoeff, theta) + np.log(base)

    #print(corr)
    #print(np.sum(np.isnan(base)), np.sum(np.isinf(base)))
    #corr = np.dot(LegendreCoeff, theta)
    # Poisson regression as a log likelihood
    # https://en.wikipedia.org/wiki/Poisson_regression
    A = np.sum(np.sum(np.transpose(y)*np.transpose(corr)))
    B = np.sum(np.transpose(np.exp(corr)))

    L0 = - np.sum(np.transpose(y)*np.transpose(corr) \
        - np.transpose(np.exp(corr)))
    #print(np.transpose(np.exp(corr)))
    #print(theta, L0)
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
    #return rosen_deralib(x, *(total_pe, PMT_pos, cut, LegendreCoeff)))
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
        # positive direction

        #h = tables.open_file('/mnt/stage/harmonic/unify/cal/up/z0.h5')
        h = tables.open_file(path + 'z' + radius + '.h5')
        EID = h.root.PETruth[:]['EventID']
        CID = h.root.PETruth[:]['ChannelID']
        PEList = np.zeros((np.size(np.unique(EID)), \
            (np.max(CID) - np.min(CID))+1))
        for i in np.arange(10):
            A = np.bincount(CID[EID==i])
            PEList[i, np.min(CID[EID==i]):np.max(CID[EID==i])+1] = A
        h.close()
        
        PEList1 = PEList[:,PMT_A[:,0].astype('int')]
        PEList2 = PEList[:,PMT_B[:,0].astype('int')]
        total_pe = np.hstack((PEList1, PEList2))
        R = 19.388
        tmp = np.vstack(( \
            R*np.cos(PMT_B[:,1])*np.cos(PMT_B[:,2]), \
            R*np.cos(PMT_B[:,1])*np.sin(PMT_B[:,2]), \
            R*np.sin(PMT_B[:,1])))
        PMT_pos = np.vstack((PMT_A[:,1:4], tmp.T))
        print('total event: %d' % np.size(np.unique(EID)), flush=True)
        
        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        size = np.size(np.unique(EID))
        tmp_x_p = Legendre_coeff(PMT_pos, np.array([0,0,1]), cut_max)
        #tmp_x_p = np.tile(tmp_x_p, (size,1))
        LegendreCoeff = tmp_x_p
        #total_pe = np.reshape(total_pe,(-1,1),order='C')
        # this part for later EM maybe
        
        for cut in np.arange(5,cut_max,1): # just take special values
            print(f'processing {cut}-th')
            theta0 = np.zeros(cut) # initial value
            theta0[0] = 1 # intercept is much more important 
            result = minimize(Calib, theta0, method='SLSQP', args = (total_pe.T, PMT_pos, cut, LegendreCoeff[:,0:cut])) 
            record = np.array(result.x, dtype=float)
            vs_x = np.reshape(total_pe,(-1,np.size(PMT_pos[:,0])), order='C')
            mean_x = np.mean(vs_x, axis=0)


            args = (total_pe, PMT_pos, cut)
            # following should be optimize...
            predict_x = np.exp(np.dot(LegendreCoeff[:,0:cut], result.x)) * base
            print('-'*80)
            print(mean_x.shape)
            print(mean_x)
            print('-'*80)
            print(predict_x.shape)
            print(predict_x)
            out.create_dataset('coeff' + str(cut), data = record)
            out.create_dataset('mean' + str(cut), data = mean_x)
            out.create_dataset('predict' + str(cut), data = predict_x)

if len(sys.argv)!=5:
    print("Wront arguments!")
    print("Usage: python main_calib_single_TW.py 'radius' 'path' outputFileName[.h5] Max_order")
    sys.exit(1)  
PMT_A = ReadPMT(r"/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/PMTPos_Acrylic_with_chimney.csv")
PMT_B = ReadPMT(r"/cvmfs/juno.ihep.ac.cn/sl6_amd64_gcc830/Pre-Release/J20v1r0-Pre2/offline/Simulation/DetSimV2/DetSimOptions/data/3inch_pos.csv")
base = LoadBase()
base = np.hstack((base[0],base[1])) + 1e-3
# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3], eval(sys.argv[4]))
