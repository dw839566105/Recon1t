# recon range: [-1,1], need * detector radius

import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from scipy.linalg import norm
import warnings
warnings.filterwarnings('ignore')

# physical constant (if need)
Light_yield = 4285*0.88 # light yield
Att_LS = 18 # attenuation length of LS
Att_Wtr = 300 # attenuation length of water
tau_r = 1.6 # fast time constant
TTS = 5.5/2.355
QE = 0.20
PMT_radius = 0.254
c = 2.99792e8
n = 1.48

# boundaries
shell_in = 0.85 # Acrylic
shell_out = 0.8
shell = 0.65

def findfile(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')    
    coeff = 'coeff' + str(order)    
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def load_spline(path='../calib/coeff_pe_1t_2.0MeV_dns_Lasso_els5/', upperlimit=0.65, cptbd=0.4, lowerlimit=0, order=15):
    ra1 = np.arange(upperlimit + 1e-5, cptbd, -0.002)
    ra2 = np.arange(cptbd + 1e-5, lowerlimit, -0.01)
    ra = np.hstack((ra1,ra2))
    
    coeff = np.zeros((order, np.size(ra)))
    for r_index, radius in enumerate(ra):
        str_radius = '%.3f' % radius
        k = findfile(path, str_radius, order)
        coeff[:,r_index] = k
    return ra, coeff

def fun_cubic(radius, coeff):
    func_list = []
    for i in np.arange(np.size(coeff[:,0])):
        # cubic interp
        xx = radius
        yy = coeff[i]
        if (np.min(xx)>0.01):
            yy = np.hstack((coeff[i,xx == np.min(xx)], coeff[i]))
            xx = np.hstack((0,radius))
        f = interpolate.interp1d(xx, yy, kind='cubic')
        func_list.append(f)
    return func_list

def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    return v

def Likelihood(vertex, *args):
    '''
    vertex[1]: r
    vertex[2]: theta
    vertex[3]: phi
    '''
    fired_PMT, time_array, pe_array = args
    L1 = Likelihood_PE(vertex, *(pe_array))
    L2 = Likelihood_Time(vertex, *(fired_PMT, time_array))
    return L1 + L2
                         
def Likelihood_PE(vertex, *args):
    event_pe = args
    y = event_pe    
    z = abs(vertex[1])
    
    if z > 1:
        z = np.sign(z)-1e-6
            
    if z<-1e-6:
        vertex[2] = vertex[2] + np.pi
        vertex[3] = vertex[3] + np.pi
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    elif(np.abs(z) < 1e-6):
        # assume (0,0,1)
        cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    
    size = np.size(PMT_pos[:,0])
    cut = np.size(PE_coeff[:,0])
    
    
    x = np.zeros((size, cut))
    # legendre theta of PMTs
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    if z < 1e-3:
        x = np.ones((30,cut))
        
    # legendre coeff by polynomials
    k = np.zeros(cut)
    for i in np.arange(cut):
        # cubic interp
        if(np.abs(z)>1):
            z = 1-1e-6
        k[i] = PE_func_list[i](z)

    k[0] = vertex[0]
    expect = np.exp(np.dot(x,k))
    a1 = expect**y
    a2 = np.exp(-expect)
    a1[a1<1e-20] = 1e-20
    a2[a2>1e50] = 1e50
    L = - np.sum(np.sum(np.log(a1*a2)))
    if(np.isnan(L)):
        print(z, expect)
        exit()
    return L

def Likelihood_Time(vertex, *args):
    fired, time = args
    y = time
    # fixed axis
    z = abs(vertex[1])
    if z > 1:
        z = np.sign(z)-1e-6
        
    if z<-1e-6:
        vertex[2] = vertex[2] + np.pi
        vertex[3] = vertex[3] + np.pi
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    elif(np.abs(z) < 1e-6):
        # assume (0,0,1)
        cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1

    cos_total = cos_theta[fired]
    
    size = np.size(cos_total)
    cut = np.size(Time_coeff[:,0])
    x = np.zeros((size, cut))
    # legendre theta of PMTs
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_total,c)
    if z < 1e-3:
        x = np.ones((30,cut))
    # legendre coeff by polynomials
    k = np.zeros((1,cut))
    for i in np.arange(cut):
        # cubic interp
        if(np.abs(z)>1):
            z = 1-1e-6
        k[0,i] = PE_func_list[i](z)
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    #L = Likelihood_quantile(y, T_i[:,0], 0.1, 0.3)
    L = - np.nansum(TimeProfile(y, T_i[:,0]))
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    less = T_i[y<T_i] - y[y<T_i]
    more = y[y>=T_i] - T_i[y>=T_i]

    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    #log_Likelihood = exp
    return R

def TimeProfile(y,T_i):
    time_correct = y - T_i
    time_correct[time_correct<=-4] = -4
    p_time = TimeUncertainty(time_correct, 26)
    return p_time

def TimeUncertainty(tc, tau_d):
    TTS = 2.2
    tau_r = 1.6
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    return p_time

def ReadPMT():
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

def recon(fid, fout, *args):
    PMT_pos, event_count = args
    # global event_count,shell,PE,time_array,PMT_pos, fired_PMT
    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(fid) # filename
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)    # EventNo
        # inner recon
        E_sph = tables.Float16Col(pos=1)        # Energy
        x_sph = tables.Float16Col(pos=2)        # x position
        y_sph = tables.Float16Col(pos=3)        # y position
        z_sph = tables.Float16Col(pos=4)        # z position
        t0 = tables.Float16Col(pos=5)       # time offset
        success = tables.Int64Col(pos=6)    # recon failure   
        Likelihood = tables.Float16Col(pos=7)
        
        x_truth = tables.Float16Col(pos=8)        # x position
        y_truth = tables.Float16Col(pos=9)        # y position
        z_truth = tables.Float16Col(pos=10)        # z position
        E_truth = tables.Float16Col(pos=11)        # z position
                        
        # unfinished
        tau_d = tables.Float16Col(pos=12)    # decay time constant

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    f = uproot.open(fid)
    a = f['SimTriggerInfo']
    for chl, Pkl, xt, yt, zt, Et in zip(a.array("PEList.PMTId"),
                    a.array("PEList.HitPosInWindow"),
                    a.array("truthList.x"),
                    a.array("truthList.y"),
                    a.array("truthList.z"),
                    a.array("truthList.EkMerged")):
        pe_array = np.zeros(np.size(PMT_pos[:,1])) # Photons on each PMT (PMT size * 1 vector)
        fired_PMT = np.zeros(0)     # Hit PMT (PMT Seq can be repeated)
        time_array = np.zeros(0, dtype=int)    # Time info (Hit number)
        for ch, pk in zip(chl, Pkl):
            try:
                pe_array[ch] = pe_array[ch]+1
                time_array = np.hstack((time_array, pk))
                fired_PMT = np.hstack((fired_PMT, ch*np.ones(np.size(pk))))
            except:
                pass

        fired_PMT = fired_PMT.astype(int)
        # initial result
        result_vertex = np.empty((0,5)) # reconstructed vertex
        
        # Constraints
        E_min = -10
        E_max = 10
        tau_min = 0.01
        tau_max = 100
        t0_min = -300
        t0_max = 600
        
        # inner recon
        # initial value
        x0_in = np.zeros((1,5))
        x0_in[0][0] = 0.8 + np.log(np.sum(pe_array)/60)
        x0_in[0][4] = np.mean(time_array) - 26

        x0_in[0][1] = np.sum(pe_array*PMT_pos[:,0])/np.sum(pe_array)/shell
        x0_in[0][2] = np.sum(pe_array*PMT_pos[:,1])/np.sum(pe_array)/shell
        x0_in[0][3] = np.sum(pe_array*PMT_pos[:,2])/np.sum(pe_array)/shell

        a = c2r(x0_in[0][1:4])
        a[0] = a[0]/shell
        # not added yet
        x0 = np.hstack((x0_in[0][0], a, x0_in[0][4]))
        result_ini = minimize(Likelihood, x0, method='SLSQP',bounds=((-10, 10), (1e-3, 1-1e-3), (None, None), (None, None), (None, None)), args = (fired_PMT, time_array, pe_array))
        
        in2 = r2c(result_ini.x[1:4])*shell
        
        recondata['EventID'] = event_count
        recondata['x_sph'] = in2[0]
        recondata['y_sph'] = in2[1]
        recondata['z_sph'] = in2[2]
        recondata['E_sph'] = result_ini.x[0]
        recondata['t0'] = result_ini.x[4]       
        recondata['success'] = result_ini.success
        recondata['Likelihood'] = result_ini.fun                
        recondata.append()
        
        rr = np.arange(0.01, 0.65, 0.01)
        L = np.zeros_like(rr)
        for bi, bb in enumerate(rr):
            vertex = result_ini.x.copy()
            vertex[1] = bb/0.65
            L[bi] = Likelihood(vertex, *(fired_PMT, time_array, pe_array))
        imax = np.where(L == np.max(L))
        #plt.figure()
        #plt.plot(rr/0.65, L)
        diff = np.diff(L)
        index = (diff[1:]*diff[:-1]*(diff[1:]>diff[:-1]))<0
        seq = np.array(np.where(index==1)) + 1
        seq = seq[0]
        
        if((imax[0] == 0) or (imax[0] == np.size(rr)-1)):
            pass
        else:
            seq = np.hstack((seq, imax[0] - 1, imax[0] + 1))
            #plt.axvline(rr[imax[0]]/0.65)
        
        if(np.size(seq)>1):
            result_new = []
            xx0 = []
            for si,ss in enumerate(seq):
                x0 = result_ini.x.copy()
                x0[1] = rr[ss]/0.65
                #plt.axvline(rr[ss]/0.65, color = 'green')
                xx0.append(x0[1])
                result_new.append(minimize(Likelihood, x0, method='SLSQP',\
                   bounds=((-10, 10), (1e-3, 1-1e-3), (None, None), (None, None), (None, None)), \
                   args = (fired_PMT, time_array, pe_array)))

        #plt.axvline(result_ini.x[1], color = 'red')
        
        recondata.append()
        print('initial')
        print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.3f, Likelihood: %+.2f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_ini.fun))

        for si,ss in enumerate(seq):
            print('iter')
            out2 = r2c(result_new[si].x[1:4])*shell
            print('%d: [%+.2f, %+.2f, %+.2f] begin:%+.3f radius: %+.3f, Likelihood: %+.2f' % (event_count, out2[0], out2[1], out2[2], xx0[si]*shell, norm(out2), result_new[si].fun))
            
            recondata['EventID'] = event_count
            recondata['x_sph'] = out2[0]
            recondata['y_sph'] = out2[1]
            recondata['z_sph'] = out2[2]
            recondata['E_sph'] = result_new[si].x[0]
            recondata['t0'] = result_new[si].x[4]       
            recondata['success'] = result_new[si].success
            recondata['Likelihood'] = result_new[si].fun                
            recondata.append()
        
        print('-'*80)
        event_count = event_count + 1
        #plt.savefig('test%d.png' % event_count)
        
    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

if len(sys.argv)!=4:
    print("Wront arguments!")
    print("Usage: python Recon.py MCFileName[.root] outputFileName[.h5] order")
    sys.exit(1)

# Read PMT position
PMT_pos = ReadPMT()
event_count = 0
# Reconstruction
fid = sys.argv[1] # input file .h5
fout = sys.argv[2] # output file .h5

PE_radius, PE_coeff = load_spline(order = eval(sys.argv[3]))
Time_radius, Time_coeff = load_spline(path='../calib/coeff_time_1t_2.0MeV_dns_Lasso/', lowerlimit = 0.01, order = 5)
PE_func_list = fun_cubic(PE_radius/0.65, PE_coeff)
Time_func_list = fun_cubic(Time_radius/0.65, Time_coeff)
args = PMT_pos, event_count
recon(fid, fout, *args)
