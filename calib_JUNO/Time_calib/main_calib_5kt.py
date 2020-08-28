import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import os
#os.chdir()
#print(os.getcwd())

def Calib(theta, *args):
    ChannelID, flight_time, PMT_pos, cut = args
    y = flight_time
    # fixed axis
    x = Legendre_coeff(PMT_pos)
    Legend_coeff = x[ChannelID,:]
    # quantile regression
    T_i = np.dot(Legend_coeff, theta)
    L = Likelihood_quantile(y, T_i, 0.01, 0.3)
    # L = np.log(np.sum((np.transpose(np.dot(Legend_coeff, theta))-y)**2))
    # print(L)
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    less = T_i[y<T_i] - y[y<T_i]
    more = y[y>=T_i] - T_i[y>=T_i]
    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    #log_Likelihood = exp
    return R

def Legendre_coeff(PMT_pos):
    vertex = np.array([0,0,10])
    cos_theta = np.sum(vertex*PMT_pos,axis=1) \
        /np.sqrt(np.sum(vertex**2)*np.sum(PMT_pos**2,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1
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
    ChannelID, PETime, PMT_pos, cut = args
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
    filename = '/mnt/stage/douwei/Simulation/5kt_root/1MeV_h5/5kt_' + radius + '.h5'

    # read files by table
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID'] - 1
    PETime = truthtable[:]['PETime']
    photonTime = truthtable[:]['photonTime']
    PulseTime = truthtable[:]['PulseTime']
    dETime = truthtable[:]['dETime']
    h1.close()
    
    # read file series
    
    try:
        for j in np.arange(1,10,1):
            filename = '/mnt/stage/douwei/Simulation/5kt_root/2MeV_h5/5kt_' + radius + '_' + str(j)+ '.h5'
            print(filename)  
            h1 = tables.open_file(filename,'r')
            truthtable = h1.root.GroundTruth

            EventID_tmp = truthtable[:]['EventID']
            ChannelID_tmp = truthtable[:]['ChannelID'] - 1
            PETime_tmp = truthtable[:]['PETime']
            photonTime_tmp = truthtable[:]['photonTime']
            PulseTime_tmp = truthtable[:]['PulseTime']
            dETime_tmp = truthtable[:]['dETime']
            
            EventID = np.hstack((EventID, EventID_tmp))
            ChannelID = np.hstack((ChannelID, ChannelID_tmp))
            PETime = np.hstack((PETime, PETime_tmp))
            photonTime = np.hstack((photonTime, photonTime_tmp))
            PulseTime = np.hstack((PulseTime, PulseTime_tmp))
            dETime = np.hstack((dETime, dETime_tmp))
            
            h1.close()
    except:
        j = j - 1
    
    total_pe = np.zeros((np.size(PMT_pos[:,0]),max(EventID)))
    
    flight_time = PulseTime - dETime
    
    ChannelID = ChannelID[~(flight_time==0)]
    flight_time = flight_time[~(flight_time==0)]
    theta0 = np.zeros(cut) # initial value
    theta0[0] = np.mean(flight_time) - 26
    result = minimize(Calib, theta0, method='SLSQP', args = (ChannelID, flight_time, PMT_pos, cut))  
    record = np.array(result.x, dtype=float)
    print(result.x)
    
    with h5py.File(fout,'w') as out:
        out.create_dataset('coeff', data = record)

def ReadPMT(geo):
    f = open(r'PMT_' + geo + '.txt')
    line = f.readline()
    data_list = []
    while line:
        num = list(map(float,line.split()))
        data_list.append(num)
        line = f.readline()
    f.close()
    PMT_pos = np.array(data_list)
    PMT_pos = PMT_pos[:,1:4]
    return PMT_pos

cut = 5 # Legend order
PMT_pos = ReadPMT(sys.argv[3])
main_Calib(sys.argv[1],sys.argv[2])
