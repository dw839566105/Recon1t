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
np.set_printoptions(precision=3, suppress=True)

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
shell = 0.65
def readtpl():
    h = tables.open_file("../MC/template.h5")
    tp = h.root.template[:]
    bins = np.vstack((h.root.x[:], h.root.y[:], h.root.z[:])).T
    h.close()
    return tp, bins

def load_coeff():
    h = tables.open_file('../calib/PE_coeff_1t.h5','r')
    coeff_pe_in = h.root.coeff_L_in[:]
    coeff_pe_out = h.root.coeff_L_out[:]
    bd_pe = h.root.bd[()] 
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape

    h = tables.open_file('../calib/Time_coeff2_1t_0.1.h5','r')
    coeff_time = h.root.coeff_L[:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe_in, coeff_pe_out, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time, bd_pe

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
    coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe = args
    L1 = Likelihood_PE(vertex, *(coeff_pe, PMT_pos, pe_array, cut_pe))
    L2 = Likelihood_Time(vertex, *(coeff_time, PMT_pos, fired_PMT, time_array, cut_time))
    return L1 + L2

def Likelihood_PE(vertex, *args):
    coeff, PMT_pos, event_pe, cut = args
    y = event_pe
    
    z = abs(vertex[1])
    if z > 1-1e-3:
        z = np.sign(z)-1e-3
            
    if z<1e-3:
        # assume (0,0,1)
        # cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
        vertex[1] = 1e-3
        z = 1e-3
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    
    size = np.size(PMT_pos[:,0])
    
    c = np.diag((np.ones(cut)))
    x = LG.legval(cos_theta, c).T
    
    k = np.zeros(cut)

    if(z>=bd_pe):
        k = LG.legval(z, coeff_pe_out.T)
    else:
        k = LG.legval(z, coeff_pe_in.T)
    
    k[0] = vertex[0]
    expect = np.exp(np.dot(x,k))
    a1 = expect**y
    a2 = np.exp(-expect)
    a1[(a1<1e-20) & (np.isnan(a1))] = 1e-20
    a1[(a1>1e50) & (np.isinf(a1))] = 1e50
    a2[(a2<1e-20) & (np.isnan(a2))] = 1e-20
    a2[(a2>1e50) & (np.isinf(a2))] = 1e50
    
    L = - np.sum(np.sum(np.log(a1*a2)))
    if(np.isinf(L) or L>1e20):
        L = 1e20

    if(np.isnan(L)):
        print(L, a1, a2)
        print(z, expect)
        print(event_pe)
        print(vertex)
        exit()
    return L

def Likelihood_Time(vertex, *args):
    coeff, PMT_pos, fired, time, cut = args
    y = time
    # fixed axis
    z = abs(vertex[1])
    if z > 1:
        z = np.sign(z)-1e-6

    if z<1e-3:
        # assume (0,0,1)
        # cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
        vertex[1] = 1e-3
        z = 1e-3
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1

    cos_total = cos_theta[fired]
    
    size = np.size(cos_total)
    
    c = np.diag((np.ones(cut)))
    x = LG.legval(cos_total, c).T
    
    ### test
    '''
    cos_theta = np.linspace(-1,1,1000)
    c = np.diag((np.ones(cut)))
    x = LG.legval(cos_theta, c).T
    k = np.zeros((1,cut))
    k[0] = LG.legval(z, coeff_time.T)
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    plt.plot(cos_theta, T_i)
    plt.title('Radius=%.2f' % z)
    plt.savefig('test.png')
    exit()
    '''
    # legendre coeff by polynomials
    k = np.zeros((1,cut))
    k[0] = LG.legval(z, coeff_time.T)
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    L = np.nansum(Likelihood_quantile(y, T_i[:,0], 0.1, 2.6))
    #L = - np.nansum(TimeProfile(y, T_i[:,0]))
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    #less = T_i[y<T_i] - y[y<T_i]
    #more = y[y>=T_i] - T_i[y>=T_i]
    
    #R = (1-tau)*np.sum(less) + tau*np.sum(more)
    L1 = (T_i-y)*(y<T_i)*(1-tau) + (y-T_i)*(y>=T_i)*tau
    L = L1/ts

    #print(np.sum(L1),L1.shape)
    #log_Likelihood = exp
    return L

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
    PMT_pos, event_count, tp, bins = args
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
        E_sph_in = tables.Float16Col(pos=1)        # Energy
        x_sph_in = tables.Float16Col(pos=2)        # x position
        y_sph_in = tables.Float16Col(pos=3)        # y position
        z_sph_in = tables.Float16Col(pos=4)        # z position
        t0_in = tables.Float16Col(pos=5)       # time offset
        success_in = tables.Int64Col(pos=6)    # recon failure   
        Likelihood_in = tables.Float16Col(pos=7)
        
        # outer recon
        E_sph_out = tables.Float16Col(pos=8)        # Energy
        x_sph_out = tables.Float16Col(pos=9)        # x position
        y_sph_out = tables.Float16Col(pos=10)        # y position
        z_sph_out = tables.Float16Col(pos=11)        # z position
        t0_out = tables.Float16Col(pos=12)       # time offset
        success_out = tables.Int64Col(pos=13)    # recon failure 
        Likelihood_out = tables.Float16Col(pos=14)

        # truth info
        x_truth = tables.Float16Col(pos=15)        # x position
        y_truth = tables.Float16Col(pos=16)        # y position
        z_truth = tables.Float16Col(pos=17)        # z position
        E_truth = tables.Float16Col(pos=18)        # z position
        # unfinished
        tau_d = tables.Float16Col(pos=19)    # decay time constant
    class TruthData(tables.IsDescription):
        pes = tables.Int16Col(pos=20)

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    TruthTable = h5file.create_table(group, "Truth", TruthData, "Truth")
    truthdata = TruthTable.row
    # Loop for event

    f = uproot.open(fid)
    a = f['SimTriggerInfo']
    for chl, Pkl, xt, yt, zt, Et in zip(a.array("PEList.PMTId"),
                    #a.array("PEList.HitPosInWindow"),
                    a.array("PEList.PulseTime"),
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
        if np.sum(pe_array)!=0:
            for ii in np.arange(np.size(pe_array)):
                truthdata['pes'] = pe_array[ii]
                truthdata.append()
            
            # Use MC to find the initial value
            data = tp
            rep = np.tile(pe_array,(np.size(bins[:,0]),1))
            real_sum = np.sum(data, axis=1)
            corr = (data.T/(real_sum/np.sum(pe_array))).T
            L = np.nansum(-corr + np.log(corr)*pe_array, axis=1)
            index = np.where(L == np.max(L))[0][0]
            print(f'max template is {np.max(L)}')
            fired_PMT = fired_PMT.astype(int)
            # initial result
            result_vertex = np.empty((0,5)) # reconstructed vertex

            # Constraints
            E_min = -10
            E_max = 10
            # inner recon
            # initial value
            x0_in = np.zeros((1,5))
            x0_in[0][0] = 0.8 + np.log(np.sum(pe_array)/60)
            x0_in[0][4] = np.quantile(time_array,0.05)

            x0_in[0][1:4] = bins[index]/1000/shell
            a = c2r(x0_in[0][1:4])
            x0_in = np.hstack((x0_in[0][0], a, x0_in[0][4]))
            result_in = minimize(Likelihood, x0_in, method='SLSQP',bounds=((E_min, E_max), (0, 1), (None, None), (None, None), (None, None)), args = (coeff_time, coeff_pe_in, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe))

            # new added avoid boundry:
            in2 = r2c(result_in.x[1:4])*shell
            recondata['x_sph_in'] = in2[0]
            recondata['y_sph_in'] = in2[1]
            recondata['z_sph_in'] = in2[2]
            recondata['E_sph_in'] = result_in.x[0]
            recondata['success_in'] = result_in.success
            recondata['Likelihood_in'] = result_in.fun

            # outer recon
            # initial value
            x0_out = x0_in.copy()
            x0_out[1]=0.92
            result_out = minimize(Likelihood, x0_out, method='SLSQP',bounds=((E_min, E_max), (0,1), (None, None), (None, None),(None, None)), args = (coeff_time, coeff_pe_out, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe))

            out2 = r2c(result_out.x[1:4]) * shell
            recondata['x_sph_out'] = out2[0]
            recondata['y_sph_out'] = out2[1]
            recondata['z_sph_out'] = out2[2]
            recondata['E_sph_out'] = result_out.x[0]
            recondata['success_out'] = result_out.success
            recondata['Likelihood_out'] = result_out.fun

            #### truth
            nml = test/(np.sum(test)/np.sum(pe_array))
            L0 = np.nansum(-nml + np.log(nml)*pe_array)
            print(f'truth likelihood is {L0}')
            x0_truth = x0_in.copy()
            x0_truth[1:4]=c2r(np.array((0,0,0.6/0.65)))
            result_truth = minimize(Likelihood, x0_truth, method='SLSQP',bounds=((E_min, E_max), (0,1), (None, None), (None, None),(None, None)), args = (coeff_time, coeff_pe_out, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe))

            truth2 = r2c(result_truth.x[1:4]) * shell

            print('-'*60)
            print(x0_in)
            print(x0_out)

            print('inner')
            print(result_in.fun)
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.2f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
            #print(event_count, result_in.x[1:4] * shell, np.sqrt(np.sum(result_in.x[1:4]**2)),result_in.fun)
            print('outer')
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.2f' % (event_count, out2[0], out2[1], out2[2], norm(out2), result_out.fun))
            print('truth')
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.2f' % (event_count, truth2[0], truth2[1], truth2[2], norm(truth2), result_out.fun))
            #print(event_count, result_out.x[1:4] * shell, np.sqrt(np.sum(result_out.x[1:4]**2)), result_out.fun)
            print('-'*60)
            
            if(result_truth.fun > result_in.fun):
                print(result_truth.fun)
                print(pe_array)
                plt.figure()
                plt.plot(PMT_pos[:,2], nml, '^')
                plt.plot(PMT_pos[:,2], pe_array, '*')
                plt.savefig('view_normal2.png')
                print(fired_PMT.size)
                plt.figure()
                plt.plot(PMT_pos[fired_PMT][:,2], time_array,'*')
                plt.savefig('view_time2.png')
                exit()   
            else:
                print(pe_array)
                plt.figure()
                plt.plot(PMT_pos[:,2], nml, '^')
                plt.plot(PMT_pos[:,2], pe_array, '*')
                plt.savefig('view_normal1.png')
                plt.plot(PMT_pos[fired_PMT][:,2], time_array,'*')
                plt.savefig('view_time1.png')
                #exit()
        else:
            recondata['x_sph_in'] = 0
            recondata['y_sph_in'] = 0
            recondata['z_sph_in'] = 0
            recondata['E_sph_in'] = 0
            recondata['success_in'] = 0
            recondata['Likelihood_in'] = 0
            
            recondata['x_sph_out'] = 0
            recondata['y_sph_out'] = 0
            recondata['z_sph_out'] = 0
            recondata['E_sph_out'] = 0
            recondata['success_out'] = 0
            recondata['Likelihood_out'] = 0
            print('empty event!')
            print('-'*60)
        recondata.append()
        
        event_count = event_count + 1

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
coeff_pe_in, coeff_pe_out, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time, bd_pe\
    = load_coeff()
tp, bins = readtpl()
args = PMT_pos, event_count, tp, bins

a1 = np.empty(0)
b1 = np.empty(0)
c1 = np.empty(0)
for i in np.arange(3):
    if (i == 0):
        h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_zQ.h5')
    else:
        h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/ground_axis/1t_+0.600_z_%dQ.h5' % i)
    a1 = np.hstack((a1, h.root.PETruthData[:]['Q']))
    b1 = np.hstack((b1, h.root.GroundTruth[:]['PulseTime']))
    c1 = np.hstack((c1, h.root.GroundTruth[:]['ChannelID']))
    h.close()
test = np.mean(np.reshape(a1, (-1,30),order='C'), axis=0)

recon(fid, fout, *args)
