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
from scipy.stats import norm as normpdf
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)
Gain = np.loadtxt('/mnt/stage/PMTGainCalib_Run0257toRun0271.txt',\
        skiprows=0, usecols=np.hstack((np.arange(0,8), np.arange(9,14))))

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
    h = tables.open_file('../calib/PE_coeff_1t_29_80.h5','r')
    coeff_pe = h.root.coeff_L[:]
    h.close()
    cut_pe, fitcut_pe = coeff_pe.shape

    h = tables.open_file('../calib/Time_coeff2_1t_0.1.h5','r')
    coeff_time = h.root.coeff_L[:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time

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
    coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, charge_array = args
    L1, E = Likelihood_PE(vertex, *(coeff_pe, PMT_pos, pe_array, cut_pe, N, pdf_pe))
    L2 = Likelihood_Time(vertex, *(coeff_time, PMT_pos, fired_PMT, time_array, cut_time, charge_array))
    return L1 + L2 

def Likelihood_PE(vertex, *args):
    coeff, PMT_pos, event_pe, cut, N, pdf_tpl = args
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
    k = LG.legval(z, coeff_pe.T)
    
    expect = np.exp(np.dot(x,k))
    nml = np.sum(expect)/np.sum(y)
    expect = expect/nml
    k[0] = k[0] - np.log(nml)
    vertex[0] = k[0]

    a1 = np.atleast_2d(expect).T ** N * pdf_tpl
    a1 = np.sum(a1,axis=1)
    a2 = np.exp(-expect)
    
    L = - np.sum(np.sum(np.log(a1*a2)))
    if(np.isinf(L) or L>1e20):
        L = 1e20
    #print(vertex[0], np.log(nml))
    return L, vertex[0]

def Likelihood_PE1(vertex, *args):
    coeff, PMT_pos, event_pe, cut, N, pdf_tpl = args
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
    k = LG.legval(z, coeff_pe.T)
    
    expect = np.exp(np.dot(x,k))
    nml = np.sum(expect)/np.sum(y)
    expect = expect/nml
    k[0] = k[0] - np.log(nml)
    vertex[0] = k[0]

    '''
    a1 = expect**y
    a2 = np.exp(-expect)
    a1[(a1<1e-20) & (np.isnan(a1))] = 1e-20
    a1[(a1>1e50) & (np.isinf(a1))] = 1e50
    a2[(a2<1e-20) & (np.isnan(a2))] = 1e-20
    a2[(a2>1e50) & (np.isinf(a2))] = 1e50
    '''
    a1 = np.atleast_2d(expect).T ** N * pdf_tpl
    a1 = np.sum(a1,axis=1)
    a2 = np.exp(-expect)
    
    L = - np.sum(np.sum(np.log(a1*a2)))
    if(np.isinf(L) or L>1e20):
        L = 1e20
    print(vertex[0], np.log(nml))
    return L, vertex[0]

def Likelihood_Time(vertex, *args):
    coeff, PMT_pos, fired, time, cut, weight = args
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
        cos_theta = np.dot(v, PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v, PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    # accurancy and nan value
    cos_theta = np.nan_to_num(cos_theta)
    cos_theta[cos_theta>1] = 1
    cos_theta[cos_theta<-1] =-1

    cos_total = cos_theta[fired]
    
    size = np.size(cos_total)
    
    c = np.diag((np.ones(cut)))
    x = LG.legval(cos_total, c).T

    # legendre coeff by polynomials
    k = np.zeros((1,cut))
    k[0] = LG.legval(z, coeff_time.T)
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    L = np.nansum(Likelihood_quantile(y, T_i[:,0], 0.1, 2.6, weight))
    #L = - np.nansum(TimeProfile(y, T_i[:,0]))
    return L

def Likelihood_quantile(y, T_i, tau, ts, weight):
    #less = T_i[y<T_i] - y[y<T_i]
    #more = y[y>=T_i] - T_i[y>=T_i]    
    #R = (1-tau)*np.sum(less) + tau*np.sum(more)
    
    L = (T_i-y)*(y<T_i)*(1-tau) + (y-T_i)*(y>=T_i)*tau
    nml = tau*(1-tau)/ts**weight
    L_norm = np.exp(-np.atleast_2d(L).T*weight) * nml / ts

    return L_norm

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
    f = tables.open_file(fid)
    charge = f.root.AnswerWF[:]['Charge']
    Hittime = f.root.AnswerWF[:]['HitPosInWindow']
    Events = f.root.AnswerWF[:]['TriggerNo']
    CID = f.root.AnswerWF[:]['ChannelID']
    EID = np.unique(EventNo)
    #f = uproot.open(fid)
    #a = f['SimTriggerInfo']
    #for chl, Pkl, xt, yt, zt, Et in zip(a.array("PEList.PMTId"),
    #                a.array("PEList.HitPosInWindow"),
    #                a.array("truthList.x"),
    #                a.array("truthList.y"),
    #                a.array("truthList.z"),
    #                a.array("truthList.EkMerged")):
    for Event in EID:

        charge_array = charge[Events==Event]/164
        fired_PMT = CID[Events==Event]        
        pe_array, cid = np.histogram(fired_PMT, bins=np.arange(31)-0.5, weights=charge_array)

        time_array = Hittime[Events == Event]

        N = np.atleast_2d(np.round(pe_array)).T \
            - np.atleast_2d(np.arange(-10,10)) # range: -10:10
        sigma_array = sigma/Gain*np.sqrt(N)
        pdf_pe = normpdf.pdf(np.atleast_2d(pe_array).T, 
            N, \
            np.atleast_2d(sigma_array)+1e-6 \
            )
        pdf_pe[N<0] = 0
        N[N<0] = 0
        pdf_pe = pdf_pe/np.atleast_2d(np.sum(pdf_pe, axis=1)).T
        
        if np.sum(pe_array_tmp)!=0:
            #for ii in np.arange(np.size(pe_array)):
            #    truthdata['pes'] = pe_array[ii]
            #    truthdata.append()
            
            # Use MC to find the initial value
            data = tp
            rep = np.tile(pe_array_tmp,(np.size(bins[:,0]),1))
            real_sum = np.sum(data, axis=1)
            corr = (data.T/(real_sum/np.sum(pe_array))).T
            L = np.nansum(-corr + np.log(corr)*pe_array, axis=1)
            index = np.where(L == np.max(L))[0][0]

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
            result_in = minimize(Likelihood, x0_in, method='SLSQP',bounds=((E_min, E_max), (0, 1), (None, None), (None, None), (None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, charge_array))
            L, E = Likelihood_PE1(result_in.x, *(coeff_pe, PMT_pos, pe_array, cut_pe, N, pdf_tpl))
            result_in.x[0] = E
            # new added avoid boundry:
            in2 = r2c(result_in.x[1:4])*shell
            recondata['x_sph_in'] = in2[0]
            recondata['y_sph_in'] = in2[1]
            recondata['z_sph_in'] = in2[2]
            # recondata['E_sph_in'] = E
            recondata['success_in'] = result_in.success
            recondata['Likelihood_in'] = result_in.fun

            # outer recon
            # initial value
            vertex = x0_in.copy()
            vertex[1] = 0.92
            y = np.linspace(0, 2*np.pi, 30)
            x = np.linspace(0, np.pi, 30)
            xx, yy = np.meshgrid(x, y, sparse=False)
            mesh = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
            data = np.zeros_like(mesh[:,0])
            for i in np.arange(np.size(data)):
                vertex[2:4] = mesh[i]
                data[i] = Likelihood(vertex, *(coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, charge_array))
            index = np.where(data==np.min(data)) 
            
            x0_out = x0_in.copy()
            #x0_out[1:4] = r2c(np.array((0.92, mesh[index[0][0],0], mesh[index[0][0],1])))
            x0_out[1:4] = np.array((0.92, mesh[index[0][0],0], mesh[index[0][0],1]))
            result_out = minimize(Likelihood, x0_out, method='SLSQP',bounds=((E_min, E_max), (0,1), (None, None), (None, None),(None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, charge_array))
            L, E = Likelihood_PE(result_out.x, *(coeff_pe, PMT_pos, pe_array, cut_pe, N, pdf_tpl))
            result_out.x[0] = E
            out2 = r2c(result_out.x[1:4]) * shell
            recondata['x_sph_out'] = out2[0]
            recondata['y_sph_out'] = out2[1]
            recondata['z_sph_out'] = out2[2]
            # recondata['E_sph_out'] = E
            recondata['success_out'] = result_out.success
            recondata['Likelihood_out'] = result_out.fun
            
            base_in = LG.legval(result_in.x[1], coeff_pe.T)
            base_out = LG.legval(result_out.x[1], coeff_pe.T)
            #print(base_in[0], result_out.x[1],base_out[0])
            print(f'inner: {np.exp(result_in.x[0] - base_in[0] + np.log(2))}')
            print(f'outer: {np.exp(result_out.x[0] - base_out[0] + np.log(2))}')
            template_E = 2/2 * 4285/4285
            #print(np.log(template_E))
            recondata['E_sph_in'] = np.exp(result_in.x[0] - base_in[0] + np.log(template_E) + np.log(2))
            recondata['E_sph_out'] = np.exp(result_out.x[0] - base_out[0] + np.log(template_E) + np.log(2))
            
            print('-'*60)
            print(x0_in)
            print(x0_out)
            print('inner')
            print(f'Template likelihood: {-np.max(L)}')
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
            #print(event_count, result_in.x[1:4] * shell, np.sqrt(np.sum(result_in.x[1:4]**2)),result_in.fun)
            print('outer')
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, out2[0], out2[1], out2[2], norm(out2), result_out.fun))
            '''
            print('truth')
            print('%d: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, truth2[0], truth2[1], truth2[2], norm(truth2), result_truth.fun))
            #print(event_count, result_out.x[1:4] * shell, np.sqrt(np.sum(result_out.x[1:4]**2)), result_out.fun)
            print('-'*60)
            '''

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
coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time\
    = load_coeff()
tp, bins = readtpl()
args = PMT_pos, event_count, tp, bins

recon(fid, fout, *args)
