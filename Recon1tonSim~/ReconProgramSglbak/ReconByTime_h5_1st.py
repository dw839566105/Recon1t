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

def Likelihood_Time(vertex, *args):
    PMT_pos, fired, time = args
    y = time - vertex[0]
    dist = np.sqrt(np.sum((PMT_pos[fired] - vertex[1:4])**2, axis=1))
    flight_time = dist/(c/n)*1e9
    L = - np.nansum(TimeProfile(y, flight_time))
    return L

def TimeProfile(y,T_i):
    time_correct = y - T_i
    time_correct[time_correct<=-8] = -8
    p_time = TimeUncertainty(time_correct, 26)
    return p_time

def TimeUncertainty(tc, tau_d):
    TTS = 2.2
    tau_r = 1.6
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time  = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    return p_time

def con_sph(args):
    E_min,\
    E_max,\
    tau_min,\
    tau_max,\
    t0_min,\
    t0_max\
    = args
    cons = ({'type': 'ineq', 'fun': lambda x: shell**2 - (x[1]**2 + x[2]**2 + x[3]**2)})
    return cons

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
    h.close()
    
    for i in np.arange(np.max(EventID)):
        event_count = event_count + 1
        index = (EventID==event_count)
        pe_array = np.zeros(np.size(PMT_pos[:,1])) # Photons on each PMT (PMT size * 1 vector)

        fired_PMT = ChannelID[index]
        for j in np.arange(np.size(fired_PMT)):
            pe_array[fired_PMT[j]] = pe_array[fired_PMT[j]]+1
        
        fired_PMT = fired_PMT.astype(int)        
        time_array = Time[index]
        
        # filter
        index_1 = (time_array>np.mean(time_array)-100) & (time_array < np.mean(time_array)+100)
        time_array = time_array[index_1]
        fired_PMT = fired_PMT[index_1]
        
        PMT_No = np.unique(fired_PMT)

        time_final = np.zeros(np.size(PMT_No))
        fired_final = np.zeros(np.size(PMT_No))
        
        for j,k in enumerate(PMT_No):
            time_final[j] = np.min(time_array[fired_PMT==k])
            fired_final[j] = k
            
        time_array = time_final
        fired_PMT = fired_final
        fired_PMT = fired_PMT.astype(int) 
        
        x0 = np.zeros((1,4))
        x0[0][0] = np.mean(time_array) - 26
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

        con_args = E_min, E_max, tau_min, tau_max, t0_min, t0_max
        
        cons_sph = con_sph(con_args)
        record = np.zeros((1,4))
        
        result = minimize(Likelihood_Time, x0, method='SLSQP',constraints=cons_sph, args = (PMT_pos, fired_PMT, time_array))
              
        recondata['x_sph'] = result.x[1]
        recondata['y_sph'] = result.x[2]
        recondata['z_sph'] = result.x[3]
        recondata['success_sph'] = result.success

        vertex = result.x[1:4]
        print(result.x, np.sqrt(np.sum(vertex**2)))
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
