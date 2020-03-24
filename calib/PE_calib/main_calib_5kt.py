import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt

def Calib(theta, *args):
    total_pe, PMT_pos, cut = args
    y = total_pe
    # fixed axis
    x = Legendre_coeff(PMT_pos)
    # Poisson regression
    L = - np.sum(np.sum(np.transpose(y)*np.transpose(np.dot(x, theta)) \
        - np.transpose(np.exp(np.dot(x, theta)))))
    return L

def Legendre_coeff(PMT_pos):
    vertex = np.array([0,0,0,10])
    cos_theta = np.sum(vertex[1:4]*PMT_pos,axis=1)\
        /np.sqrt(np.sum(vertex[1:4]**2)*np.sum(PMT_pos**2,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] = -1
    size = np.size(PMT_pos[:,0])
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    return x  

def rosen_hess(x, *args):
    x = np.asarray(x)
    H = np.diag(-400*x[:-1],1) - np.diag(400*x[:-1],-1)
    diagonal = np.zeros_like(x)
    diagonal[0] = 1200*x[0]**2-400*x[1]+2
    diagonal[-1] = 200
    diagonal[1:-1] = 202 + 1200*x[1:-1]**2 - 400*x[2:]
    H = H + np.diag(diagonal)
    return H

def rosen_der(x, *args):
    xm = x[1:-1]
    xm_m1 = x[:-2]
    xm_p1 = x[2:]
    der = np.zeros_like(x)
    der[1:-1] = 200*(xm-xm_m1**2) - 400*(xm_p1 - xm**2)*xm - 2*(1-xm)
    der[0] = -400*x[0]*(x[1]-x[0]**2) - 2*(1-x[0])
    der[-1] = 200*(x[-1]-x[-2]**2)
    return der

def hessian(x, *args):
    total_pe, PMT_pos, cut = args
    H = np.zeros((len(x),len(x)))
    h = 1e-3
    k = 1e-3
    for i in np.arange(len(x)):
        for j in np.arange(len(x)):
            if (i != j):
                delta1 = np.zeros(len(x))
                delta1[i] = h
                delta1[j] = k
                delta2 = np.zeros(len(x))
                delta2[i] = -h
                delta2[j] = k


                L1 = - Calib(x + delta1, *(total_pe, PMT_pos, cut))
                L2 = - Calib(x - delta1, *(total_pe, PMT_pos, cut))
                L3 = - Calib(x + delta2, *(total_pe, PMT_pos, cut))
                L4 = - Calib(x - delta2, *(total_pe, PMT_pos, cut))
                H[i,j] = (L1+L2-L3-L4)/(4*h*k)
            else:
                delta = np.zeros(len(x))
                delta[i] = h
                L1 = - Calib(x + delta, *(total_pe, PMT_pos, cut))
                L2 = - Calib(x - delta, *(total_pe, PMT_pos, cut))
                L3 = - Calib(x, *(total_pe, PMT_pos, cut))
                H[i,j] = (L1+L2-2*L3)/h**2                
    return H


def main_Calib(radius, fout):
    # read file series
    
    try:
        for j in np.arange(1,10,1):
            filename = '/mnt/stage/douwei/Simulation/5kt_root/1MeV/5kt_+' + radius + '_' + str(j)+ '.h5'           
            h1 = tables.open_file(filename,'r')
            print(filename)
            truthtable = h1.root.GroundTruth

            EventID_tmp = truthtable[:]['EventID']
            ChannelID_tmp = truthtable[:]['ChannelID'] - 1
            EventID = np.hstack((EventID, EventID_tmp))
            ChannelID = np.hstack((ChannelID, ChannelID_tmp))

            h1.close()
    except:
        j = j - 1
    
    total_pe = np.zeros((np.size(PMT_pos[:,0]),max(EventID)))
    for k in np.arange(1, max(EventID)):
        event_pe = np.zeros(np.size(PMT_pos[:,0]))
        hit = ChannelID[EventID == k]
        tabulate = np.bincount(hit)
        event_pe[0:np.size(tabulate)] = tabulate
        total_pe[:,k-1] = event_pe
    theta0 = np.zeros(cut) # initial value
    result = minimize(Calib,theta0, method='SLSQP',jac=rosen_der, args = (total_pe, PMT_pos, cut))  
    record = np.array(result.x, dtype=float)
    '''
    H = hessian(result.x, *(total_pe, PMT_pos, cut))
    H_I = np.linalg.pinv(np.matrix(H))
    
    x = Legendre_coeff(PMT_pos)
    expect = np.mean(total_pe, axis=1)
    args = (total_pe, PMT_pos, cut)
    predict = [];
    predict.append(np.exp(np.dot(x, result.x)))
    #predict.append(expect)
    predict = np.transpose(predict)
    print(2*np.sum(- total_pe + predict + np.nan_to_num(total_pe*np.log(total_pe/predict)), axis=1)/(np.max(EventID)-30))
    
    #print(np.dot(x, result.x) - expect)
    exit()
    # print(np.size(total_pe,1))
    '''
    with h5py.File(fout,'w') as out:
        out.create_dataset('coeff', data = record)

## read data from calib files
def ReadPMT(geo):
    f = open(r'../PMT' + geo + '.txt')
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)

cut = 5 # Legend order
PMT_pos = ReadPMT(sys.argv[4])
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3])
