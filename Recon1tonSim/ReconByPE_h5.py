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
import warnings

warnings.filterwarnings('ignore')

h = tables.open_file('../calib/PE_coeff.h5','r')
coeff = h.root.coeff[:]
h.close()
cut, fitcut = coeff.shape

# physical constant
Light_yield = 4285*0.88 # light yield
Att_LS = 18 # attenuation length of LS
Att_Wtr = 300 # attenuation length of water
tau_r = 1.6 # fast time constant
TTS = 5.5/2.355
QE = 0.20
PMT_radius = 0.254
c = 2.99792e8
n = 1.48
shell = 0.6 # Acrylic

def Likelihood_Sph(vertex, *args):
    coeff, PMT_pos, event_pe, cut = args
    y = event_pe
    # fixed axis
    z = np.sqrt(np.sum(vertex[1:4]**2))
    cos_theta = np.sum(vertex[1:4]*PMT_pos,axis=1)\
        /np.sqrt(np.sum(vertex[1:4]**2)*np.sum(PMT_pos**2,axis=1))
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

    k = np.zeros(cut)
    for i in np.arange(cut):
        # cubic interp
        if(np.abs(z)>0.65):
            z = 0.65*np.sign(z)
        k[i] = np.sum(np.polynomial.legendre.legval(z,coeff[i,:]))
    
    #k[0] = k[0] + np.log(vertex[0])
    k[0] = vertex[0]
    expect = np.exp(np.dot(x,k))
    L = - np.sum(np.sum(np.log((expect**y)*np.exp(-expect))))
    return L

def con_sph(args):
    E_min,\
    E_max,\
    tau_min,\
    tau_max,\
    t0_min,\
    t0_max\
    = args
    cons = ({'type': 'ineq', 'fun': lambda x: (x[0] - E_min)*(E_max - x[0])},\
    {'type': 'ineq', 'fun': lambda x: shell**2 - (x[1]**2 + x[2]**2 + x[3]**2)})
    return cons

def ReadPMT():
    f = open(r"./PMT1t.txt")
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
        x = tables.Float16Col(pos=1)        # x position
        y = tables.Float16Col(pos=2)        # y position
        z = tables.Float16Col(pos=3)        # z position
        t0 = tables.Float16Col(pos=4)       # time offset
        E = tables.Float16Col(pos=5)        # energy
        tau_d = tables.Float16Col(pos=6)    # decay time constant
        success = tables.Int64Col(pos=7)    # recon failure
        x_sph = tables.Float16Col(pos=8)        # x position
        y_sph = tables.Float16Col(pos=9)        # y position
        z_sph = tables.Float16Col(pos=10)        # z position
        E_sph = tables.Float16Col(pos=11)        # energy
        success_sph = tables.Int64Col(pos=12)    # recon failure
        
        x_truth = tables.Float16Col(pos=13)        # x position
        y_truth = tables.Float16Col(pos=14)        # y position
        z_truth = tables.Float16Col(pos=15)        # z position
        E_truth = tables.Float16Col(pos=16)        # z position
    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    h = tables.open_file(fid,'r')
    rawdata = h.root.GroundTruth
    EventID = rawdata[:]['EventID']
    ChannelID = rawdata[:]['ChannelID']
    Time = rawdata[:]['PETime']

    for i in np.arange(np.max(EventID)):
        event_count = event_count + 1
        index = (EventID==event_count)
        pe_array = np.zeros(np.size(PMT_pos[:,1])) # Photons on each PMT (PMT size * 1 vector)
        fired_PMT = ChannelID[index]
        for j in np.arange(np.size(fired_PMT)):
            pe_array[fired_PMT[j]] = pe_array[fired_PMT[j]]+1
                
        time_array = Time[index]        
        fired_PMT = fired_PMT.astype(int)
        # initial result
        result_vertex = np.empty((0,6)) # reconstructed vertex
        # initial value x[0] = [1,6]
        
        x0 = np.zeros((1,4))
        x0[0][0] = pe_array.sum()/60
        x0[0][1] = np.sum(pe_array*PMT_pos[:,0])/np.sum(pe_array)
        x0[0][2] = np.sum(pe_array*PMT_pos[:,1])/np.sum(pe_array)
        x0[0][3] = np.sum(pe_array*PMT_pos[:,2])/np.sum(pe_array)
        # Constraints
        E_min = 0.01
        E_max = 10
        tau_min = 0.01
        tau_max = 100
        t0_min = -300
        t0_max = 300
       
        # initial value
        x0 = np.zeros((1,4))
        x0[0][0] = pe_array.sum()/60
        x0[0][1] = np.sum(pe_array*PMT_pos[:,0])/np.sum(pe_array)
        x0[0][2] = np.sum(pe_array*PMT_pos[:,1])/np.sum(pe_array)
        x0[0][3] = np.sum(pe_array*PMT_pos[:,2])/np.sum(pe_array)

        if(np.sqrt(np.sum(x0**2))>0.65):
            x0 = x0/np.sqrt(np.sum(x0**2))*0.65
        con_args = E_min, E_max, tau_min, tau_max, t0_min, t0_max
        cons_sph = con_sph(con_args)
        record = np.zeros((1,4))
        
        result1 = minimize(Likelihood_Sph, x0, method='SLSQP',constraints=cons_sph, args = (coeff, PMT_pos, pe_array, cut))
        x1 = x0.copy();
        x1[1::] = x1[1::]/np.sqrt(np.sum(x1[1::]**2))*0.50
        result2 = minimize(Likelihood_Sph, x1, method='SLSQP',constraints=cons_sph, args = (coeff, PMT_pos, pe_array, cut))

        if result1.fun < result2.fun:
            result = result1
        else:
            result = result2
              
        recondata['x_sph'] = result.x[1]
        recondata['y_sph'] = result.x[2]
        recondata['z_sph'] = result.x[3]
        recondata['E_sph'] = result.x[0]
        recondata['success_sph'] = result.success

        vertex = result.x[1:4]

        event_count = event_count + 1
        recondata.append()

    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

if len(sys.argv)!=3:
    print("Wront arguments!")
    print("Usage: python Recon.py MCFileName[.h5] outputFileName[.h5]")
    sys.exit(1)


# Read PMT position
PMT_pos = ReadPMT()
event_count = 0
# Reconstruction
fid = sys.argv[1] # input file .h5
fout = sys.argv[2] # output file .h5
args = PMT_pos, event_count
recon(fid, fout, *args)
