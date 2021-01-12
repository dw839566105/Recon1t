# recon range: [-1,1], need * detector radius
import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
#import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
#from numpy.polynomial import legendre as LG
from numba import jit
from scipy import special
from scipy.linalg import norm
from scipy.stats import norm as normpdf
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')
sys.stdout.flush()
np.set_printoptions(precision=3, suppress=True)

Gain = np.loadtxt('/mnt/stage/PMTGainCalib_Run0257toRun0271.txt',\
        skiprows=0, usecols=np.hstack((np.arange(0,8), np.arange(9,14))))

# boundaries
shell = 0.65
bd = 0.572

@jit(nopython=True)
def legval(x, c):
    """
    stole from the numerical part of numpy.polynomial.legendre

    """
    if len(c) == 1:
        return c[0]
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        nd = len(c)
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            nd = nd - 1
            c0 = c[-i] - (c1*(nd - 1))/nd
            c1 = tmp + (c1*x*(2*nd - 1))/nd
    return c0 + c1*x

def readtpl():
    # Read MC grid recon result
    h = tables.open_file("../MC/template.h5")
    tp = h.root.template[:]
    bins = np.vstack((h.root.x[:], h.root.y[:], h.root.z[:])).T
    h.close()
    return tp, bins

def load_coeff():
    h = tables.open_file('../calib/PE_coeff_1t_new.h5','r')
    coeff_pe_in = h.root.coeff_in[:]
    coeff_pe_out = h.root.coeff_out[:]
    bd_pe = h.root.bd[()]
    h.close()
    cut_pe, fitcut_pe = coeff_pe_in.shape

    h = tables.open_file('../calib/Time_coeff2_1t_0.1.h5','r')
    coeff_time = h.root.coeff_L[:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe_in, coeff_pe_out, coeff_time, cut_pe, cut_time, bd_pe


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
    #v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    v[2] = np.arctan2(c[1],c[0])
    return v

def Likelihood(vertex, *args):
    '''
    vertex[1]: r
    vertex[2]: theta
    vertex[3]: phi
    '''
    fired_PMT, time_array, pe_array = args
    z, x = Calc_basis(vertex)
    L1, E = Likelihood_PE(z, x, pe_array)
    L2 = Likelihood_Time(z, x, vertex[4], fired_PMT, time_array)
    return L1 + L2

def Calc_basis(vertex): 
    # boundary
    v = r2c(vertex[1:4])
    z = norm(v)
    if z > 1-1e-3:
        z = 1-1e-3
    # calculate cos theta
    cos_theta = np.dot(v, PMT_pos.T) / (norm(v)*norm(PMT_pos,axis=1))
    ### Notice: Here may not continuous! ###
    cos_theta[np.isnan(cos_theta)] = 1 # for v in detector center    
    
    # Generate Legendre basis
    # x = legval(cos_theta, np.diag((np.ones(cut)))).T 
    x = legval(cos_theta, np.eye(cut).reshape((cut,cut,1))).T
    return z, x
    
def Likelihood_PE(z, x, pe_array):
    # Recover coefficient
    if(z<bd):
        k = legval(z, coeff_pe_in.T)
    else:
        k = legval(z, coeff_pe_out.T)
    # Recover expect
    expect = np.exp(np.dot(x,k))
    # Energy fit 
    nml = np.sum(expect)/np.sum(pe_array)
    expect = expect/nml
    k[0] = k[0] - np.log(nml) # 0-th

    # Poisson likelihood
    # p(q|lambda) = sum_n p(q|n)p(n|lambda)
    #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
    # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1
    a0 = expect ** pe_array
    a2 = np.exp(-expect)

    # -ln Likelihood
    L = - np.sum(np.sum(np.log(a0*a2)))
    # avoid inf (very impossible vertex) 
    if(np.isinf(L) or L>1e20):
        L = 1e20
    return L, k[0]

def Likelihood_Time(z, x, T0, fired_PMT, time_array):
    x = x[fired_PMT][:,:cut_time]
    
    # Recover coefficient
    k = np.atleast_2d(legval(z, coeff_time.T)).T
    k[0,0] = T0
    
    # Recover expect
    T_i = np.dot(x, k)
    
    # Likelihood
    L = - np.nansum(Likelihood_quantile(time_array, T_i[:,0], 0.1, 2.6))
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    # less = T_i[y<T_i] - y[y<T_i]
    # more = y[y>=T_i] - T_i[y>=T_i]    
    # R = (1-tau)*np.sum(less) + tau*np.sum(more)
    
    # since lucy ddm is not sparse, use PE as weight
    L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
    #nml = tau*(1-tau)/ts
    #L_norm = np.exp(-np.atleast_2d(L).T) * nml / ts
    #L = np.sum(np.log(L_norm), axis=1)
    L0 =  - L/ts
    return L0

def recon(fid, fout):

    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    event_count = 0
    # Create the output file and the group
    print(fid) # filename
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)    # EventNo
        # inner recon
        E_sph_in = tables.Float16Col(pos=1)        # Energy
        x_sph_in = tables.Float16Col(pos=2)        # x position
        y_sph_in = tables.Float16Col(pos=3)        # y position
        z_sph_in = tables.Float16Col(pos=4)        # z position
        t0_in = tables.Float16Col(pos=5)          # time offset
        success_in = tables.Int64Col(pos=6)        # recon status   
        Likelihood_in = tables.Float16Col(pos=7)
        
        # outer recon
        E_sph_out = tables.Float16Col(pos=8)         # Energy
        x_sph_out = tables.Float16Col(pos=9)         # x position
        y_sph_out = tables.Float16Col(pos=10)        # y position
        z_sph_out = tables.Float16Col(pos=11)        # z position
        t0_out = tables.Float16Col(pos=12)          # time offset
        success_out = tables.Int64Col(pos=13)        # recon status 
        Likelihood_out = tables.Float16Col(pos=14)
    
    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    # define outer grid
    vertex = np.ones(5)
    vertex[1] = 0.92
    k = legval(vertex[1], coeff_pe_out.T)
    vertex[0] = k[0]
    NN = 50
    y = np.linspace(0, 2*np.pi, NN)
    x = np.linspace(0, np.pi, NN)
    xx, yy = np.meshgrid(x, y, sparse=False)
    mesh = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
    tp_out = np.zeros((np.size(mesh[:,0]), 30))
    for i in np.arange(tp_out.shape[0]):
        vertex[2:4] = mesh[i]
        z, x = Calc_basis(vertex)
        # Recover expect
        expect = np.exp(np.dot(x,k))
        tp_out[i] = expect
    # Gain = 164
    # sigma = 40
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

        fired_PMT = fired_PMT.astype('int')
        # For hit info
        # pe_array, cid = np.histogram(chl, bins=np.arange(31)) 
        # For very rough estimate
        # pe_array = np.round(pe_array)

        # calculate pdf template
        '''
        ## DO NOT USE IN LUCY DDM 
        N0 = np.atleast_2d(np.round(PE/Gain)).T \
            - np.atleast_2d(np.arange(-3,3)) # range: -10:10
        sigma_array = sigma/Gain*np.sqrt(N0)
        pdf_weight = normpdf.pdf(np.atleast_2d(PE/Gain).T,\
            N0, \
            np.atleast_2d(sigma_array)+1e-6 \
            )
        pdf_weight[N0<0] = 0
        N0[N0<0] = 0
        '''

        if np.sum(pe_array)!=0:
            # Use MC to find the initial value
            rep = np.tile(pe_array,(np.size(bins[:,0]),1))
            real_sum = np.sum(tp, axis=1)
            corr = (tp.T/(real_sum/np.sum(pe_array))).T
            L = -np.nansum(-corr + np.log(corr)*pe_array, axis=1)
            index = np.where(L == np.min(L))[0][0]

            # Constraints (log scale)
            E_min = -10
            E_max = 10
            
            # inner recon
            # initial value
            # Energy recon will be removed later
            x0_in = np.zeros((1,5))
            x0_in[0][0] = 0.8 + np.log(np.sum(pe_array)/60)
            x0_in[0][4] = np.quantile(time_array,0.1)
            # x0_in[0][1:4] = c2r(bins[index]/1000/shell)
            x0_in[0][1:4] = c2r(np.sum(PMT_pos.T * pe_array, axis=1)/np.sum(pe_array) * 1.5)
            result_in = minimize(Likelihood, x0_in, method='SLSQP',bounds=((E_min, E_max), (0, 1), (None, None), (None, None), (None, None)), args = (fired_PMT, time_array, pe_array))
            z, x = Calc_basis(result_in.x)
            L, E_in = Likelihood_PE(z, x, pe_array)

            # xyz coordinate
            in2 = r2c(result_in.x[1:4])*shell
            recondata['x_sph_in'] = in2[0]
            recondata['y_sph_in'] = in2[1]
            recondata['z_sph_in'] = in2[2]
            recondata['success_in'] = result_in.success
            recondata['Likelihood_in'] = result_in.fun
            recondata['t0_in'] = result_in.x[4]

            # choose best vertex to outer recon
            rep = np.tile(pe_array,(np.size(mesh[:,0]),1))
            real_sum = np.sum(tp_out, axis=1)
            corr = (tp_out.T/(real_sum/np.sum(pe_array))).T
            L = -np.nansum(-corr + np.log(corr)*pe_array, axis=1)
            index = np.where(L == np.min(L))[0][0]
            
            x0_out = x0_in[0]
            # x0_out[1:4] = np.array((0.92, mesh[index,0], mesh[index,1]))
            x0_out[1] = 0.92
            result_out = minimize(Likelihood, x0_out, method='SLSQP',bounds=((E_min, E_max), (0,1), (None, None), (None, None),(None, None)), args = (fired_PMT, time_array, pe_array))
            z, x = (Calc_basis(result_out.x))
            L, E_out = Likelihood_PE(z, x, pe_array)
            
            out2 = r2c(result_out.x[1:4]) * shell
            recondata['x_sph_out'] = out2[0]
            recondata['y_sph_out'] = out2[1]
            recondata['z_sph_out'] = out2[2]
            recondata['success_out'] = result_out.success
            recondata['Likelihood_out'] = result_out.fun
            recondata['t0_out'] = result_out.x[4]
            
            # 0-th order (Energy intercept)
            if result_in.x[1]<bd:
                base_in = legval(result_in.x[1], coeff_pe_in.T)
            else:
                base_in = legval(result_in.x[1], coeff_pe_out.T)
                
            if result_out.x[1]<bd:
                base_out = legval(result_out.x[1], coeff_pe_in.T)
            else:
                base_out = legval(result_out.x[1], coeff_pe_out.T)

            print('-'*60)
            print(f'inner: {np.exp(E_in - base_in[0] + np.log(2))}')
            print(f'outer: {np.exp(E_out - base_out[0] + np.log(2))}')
            template_E = 2/2 * 4285/4285 # template is 2.0 MeV, light yield 4285/MeV
            recondata['E_sph_in'] = np.exp(E_in - base_in[0] + np.log(template_E) + np.log(2))
            recondata['E_sph_out'] = np.exp(E_out - base_out[0] + np.log(template_E) + np.log(2))

            print('inner')
            print(f'Template likelihood: {-np.max(L)}')
            print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
            print('outer')
            print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, out2[0], out2[1], out2[2], norm(out2), result_out.fun))
            
        else:
            recondata['x_sph_in'] = 0
            recondata['y_sph_in'] = 0
            recondata['z_sph_in'] = 0
            recondata['E_sph_in'] = 0
            recondata['success_in'] = 0
            recondata['Likelihood_in'] = 0
            recondata['t0_in'] = 0
            
            recondata['x_sph_out'] = 0
            recondata['y_sph_out'] = 0
            recondata['z_sph_out'] = 0
            recondata['E_sph_out'] = 0
            recondata['success_out'] = 0
            recondata['Likelihood_out'] = 0
            recondata['t0_out'] = 0
            print('empty event!')
            print('-'*60)
        recondata.append()
        event_count = event_count + 1
        sys.stdout.flush()

    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

if len(sys.argv)!=3:
    print("Wront arguments!")
    print("Usage: python Recon.py MCFileName[.root] outputFileName[.h5]")
    sys.exit(1)

# Read PMT position
PMT_pos = np.loadtxt(r"./PMT_1t.txt")

# Reconstruction
fid = sys.argv[1] # input file .h5
fout = sys.argv[2] # output file .h5

coeff_pe_in, coeff_pe_out, coeff_time, cut_pe, cut_time, bd = load_coeff()
cut = np.max((cut_pe, cut_time))

tp, bins = readtpl()
recon(fid, fout)
