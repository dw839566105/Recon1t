import numpy as np
import scipy, h5py
import scipy.stats as stats
from scipy.linalg import norm
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from Readlog import coeff3d
# coeff3d
EE_tmp, radius, coeff = coeff3d()
EE = np.zeros(len(EE_tmp))
for i in np.arange(len(EE)):
    EE[i] = eval(EE_tmp[i])
EE[-1] = 10
cut = np.size(coeff[0,:,0])

func_list = []
for i in np.arange(cut):
    # cubic interp
    xx = radius
    yy = coeff[:,i,-1]
    f = interpolate.interp1d(xx, yy, kind='cubic')
    func_list.append(f)

EE_tmp = np.hstack((0.1,EE))
EE_value = coeff[np.int((np.size(radius)-1)/2),0,:]
EE_value = np.hstack((EE_value[0],EE_value))

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
shell = 0.65 # Acrylic

def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.sin(c[1]) #z
    rho = c[0] * np.cos(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def Likelihood_Sph(vertex, *args):
    '''
    vertex[1]: r
    vertex[2]: theta
    vertex[3]: phi
    '''

    coeff, PMT_pos, event_pe, cut = args
    y = event_pe
    
    z = abs(vertex[1])
    if z > shell:
        return np.inf

    if z<0.001:
        # assume (0,0,1)
        cos_theta = PMT_pos[:,2] / norm(PMT_pos,axis=1)
    else:
        v = r2c(vertex[1:4])
        cos_theta = np.dot(v,PMT_pos.T) / (z*norm(PMT_pos,axis=1))
    
    size = np.size(PMT_pos[:,0])
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)

    k = np.zeros((np.size(coeff[0,:,0])))
    #print(np.size(coeff[0,:]))
    for i in np.arange(cut):
        # cubic interp
        if(np.abs(z)>0.65):
            z = 0.65*np.sign(z)
        k[i] = func_list[i](z)

    k[0] = vertex[0]
    expect = np.exp(np.dot(x,k))
    L = - np.sum( y*np.log(expect) - expect )
    return L

def Likelihood_Tau(t0, *args):
    Time = args
    taud = t0[0]
    Time = Time - t0[1]
    # fixed axis
    Time[Time<=-8] = -8
    p_time = TimeUncertainty(Time, taud)
    L = - np.sum(p_time)
    return L

def TimeUncertainty(tc, tau_d):
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time  = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    return p_time

def Likelihood_ML(fit, *args):
    Energy,\
    x,\
    y,\
    z,\
    t,\
    tau_d\
    = fit
    PMT_pos, pe_array, time_array, fired_PMT = args
    distance, Omega = SolidAngle(x,y,z)
    lmbd = Att(x,y,z)
    # expect photons
    expect = Energy*\
        Light_yield*\
        np.exp(-distance*lmbd/Att_LS - distance*(1-lmbd)/Att_Wtr)*\
        Omega*\
        QE
    # log Poisson # p_pe = - np.log(stats.poisson.pmf(PE, expect))
    log_p_pe = - expect + pe_array*np.log(expect) 
    # this part is nonsense {- np.log(special.factorial(pe_array))}
    Likelihood_pe = - np.nansum(log_p_pe)
    # log Time profile pdf
    # log_p_time = TimeProfile(time_array, distance[fired_PMT], tau_d, t)
    # Likelihood_time = - np.nansum(log_p_time)
    # total likelihood
    Likelihood_total = Likelihood_pe
    #Likelihood_total = Likelihood_pe + Likelihood_time
    return Likelihood_total

def SolidAngle(x, y, z):
    distance = np.sqrt(np.sum((PMT_pos - np.array((x,y,z)))**2, axis=1))
    radius_O1 = PMT_radius # PMT bottom surface
    PMT_vector = - PMT_pos/np.transpose(np.tile(np.sqrt(np.sum(PMT_pos**2,1)),[3,1]))
    O1 = np.tile(np.array([x,y,z]),[len(PMT_pos[:,0]),1])
    O2 = PMT_pos
    flight_vector = O2 - O1
    d2 = np.sqrt(np.sum(flight_vector**2,1))
    theta1 = np.sum(PMT_vector*flight_vector,1)/np.sqrt(np.sum(PMT_vector**2,1)*np.sum(flight_vector**2,1))
    Omega = (1-d2/np.sqrt(d2**2+radius_O1*np.abs(theta1)))/2
    
    return distance, Omega

def Att(x, y, z):
    '''
    this function returns ratio in different material 
    lmbd is in the LS and 1-lmbda is the water
    '''
    # LS distance
    d1 = np.tile(np.array([x,y,z]),[len(PMT_pos[:,1]),1])
    d2 = PMT_pos
    d3 = d2 - d1
    # cons beyond shell 
    lmbd = (-2*np.sum(d3*d1,1) \
        + np.sqrt(4*np.sum(d3*d1,1)**2 \
        - 4*np.sum(d3**2,1)*(-np.abs((np.sum(d1**2,1)-shell**2))))) \
        /(2*np.sum(d3**2,1))
    lmbd[lmbd>=1] = 1
    return lmbd

def TimeProfile(time_array, distance, tau_d, t):
    time_correct = time_array - distance/(c/n)*1e9 - t
    time_correct[time_correct<=-8] = -8
    p_time = TimeUncertainty(time_correct, tau_d)
    return p_time

def TimeUncertainty(tc, tau_d):
    a1 = np.exp(((TTS**2 - tc*tau_d)**2-tc**2*tau_d**2)/(2*TTS**2*tau_d**2))
    a2 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))
    a3 = np.exp(((TTS**2 - tc*tau_d)**2 - tc**2*tau_d**2)/(2*TTS**2*tau_d**2))*special.erf((tc*tau_d-TTS**2)/(np.sqrt(2)*tau_d*TTS))
    a4 = np.exp(((TTS**2*(tau_d+tau_r) - tc*tau_d*tau_r)**2 - tc**2*tau_d**2*tau_r**2)/(2*TTS**2*tau_d**2*tau_r**2))*special.erf((tc*tau_d*tau_r-TTS**2*(tau_d+tau_r))/(np.sqrt(2)*tau_d*tau_r*TTS))
    p_time  = np.log(tau_d + tau_r) - 2*np.log(tau_d) + np.log(a1-a2+a3-a4)
    
    return p_time

def cons_t():
    cons = ({'type': 'ineq', 'fun': lambda x: (x[0] - 1)*(100 - x[0])})
    return cons

def recon_drc(time_array, fired_PMT, recon_vertex):
    time_corr = time_array - np.sum(PMT_pos[fired_PMT,1:4]-np.tile(recon_vertex[0,1:4],[len(fired_PMT),1]))/(3*10**8)
    index = np.argsort(time_corr)
    fired_PMT_sorted = fired_PMT[index]
    fired_PMT_sorted = fired_PMT_sorted[0:int(np.floor(len(fired_PMT_sorted)/10))]
    drc = np.sum(PMT_pos[fired_PMT_sorted,1:4],0)/len(fired_PMT_sorted)
    return drc

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

def fit(pe_array, time_array, fired_PMT, PMT_pos, recondata):
    result_vertex = np.empty((0,6)) # reconstructed vertex
    # initial value x[0] = [1,6]

    # Constraints
    E_min = -10
    E_max = 10
    tau_min = 0.01
    tau_max = 100
    t0_min = -300
    t0_max = 300

    # initial value
    x0 = np.zeros((1,4))
    x0[0][0] = pe_array.sum()/300
    x0[0][1] = np.sum(pe_array*PMT_pos[:,0])/np.sum(pe_array)
    x0[0][2] = np.sum(pe_array*PMT_pos[:,1])/np.sum(pe_array)
    x0[0][3] = np.sum(pe_array*PMT_pos[:,2])/np.sum(pe_array)

    # Constraints
    # x0 = np.sum(PE*PMT_pos,axis=0)/np.sum(PE)
    theta0 = np.array([1,0.1,0.1,0.1])
    theta0[0] = x0[0][0]
    theta0[1] = x0[0][1]
    theta0[2] = x0[0][2]
    theta0[3] = x0[0][3]
    record = np.zeros((1,4))
    result = minimize(Likelihood_Sph, theta0, method='SLSQP', bounds=((E_min, E_max), (-shell-0.01, shell+0.01), (None, None), (None, None)), 
                      args = (coeff, PMT_pos, pe_array, cut))
    # record[0,:] = np.array(result.x, dtype=float)
    # result_total = np.vstack((result_total,record))

    v = r2c(result.x[1:4])

    # result
    recondata['x_sph'] = v[0]
    recondata['y_sph'] = v[1]
    recondata['z_sph'] = v[2]
    recondata['l0_sph'] = result.x[0]
    recondata['success_sph'] = result.success

    vertex = result.x[1:4]
    dis = np.sqrt(np.sum((PMT_pos - vertex)**2, axis=1))
    time_array = time_array - dis[fired_PMT ]/3e8/1.5*1e9
    t0 = np.array((26, np.mean(time_array)))
    try:
        Timeleft = np.min(time_array[time_array > np.mean(time_array)-100])
        time_array = time_array[time_array > Timeleft]
        time_array = time_array[time_array < Timeleft + 150]

        result = minimize(Likelihood_Tau, t0, constraints=cons_t(), method='SLSQP', args = time_array)
        # print(result.x)
        recondata['tau_d'] = result.x[0]
        recondata['t0'] = result.x[1]
    except:
        recondata['tau_d'] = 26
        recondata['t0'] = -1

    recondata.append()

def recon(fid, fout, PMT_pos):
    PMT_pos
    # global shell,PE,time_array,PMT_pos, fired_PMT
    '''
    reconstruction

    fid: root reference file
    fout: output file
    '''
    # Create the output file and the group
    print(fid) # filename
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)    # EventNo
        t0 = tables.Float16Col(pos=4)       # time offset
        tau_d = tables.Float16Col(pos=6)    # decay time constant
        x_sph = tables.Float16Col(pos=8)        # x position
        y_sph = tables.Float16Col(pos=9)        # y position
        z_sph = tables.Float16Col(pos=10)        # z position
        l0_sph = tables.Float16Col(pos=11)        # energy
        success_sph = tables.Int64Col(pos=12)    # recon failure
    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event
    '''
    h = tables.open_file(fid,'r')
    rawdata = h.root.RawData
    EventID = rawdata[:]['EventID']
    ChannelID = rawdata[:]['ChannelID']
    Time = rawdata[:]['Time']
    '''
    f = uproot.open(fid)
    a = f['SimpleAnalysis']
    for chl, PEl, Pkl in zip(a.array("ChannelInfo.ChannelId"),
                            a.array("ChannelInfo.PE"),
                            a.array("ChannelInfo.PeakLoc")):
        pe_array = np.zeros(np.size(PMT_pos[:,1])) # Photons on each PMT (PMT size * 1 vector)
        fired_PMT = np.zeros(0)     # Hit PMT (PMT Seq can be repeated)
        time_array = np.zeros(0, dtype=int)    # Time info (Hit number)
        for ch, pe, pk in zip(chl, PEl, Pkl):
            if ch >= 30:
                continue
            pe_array[ch] = pe
            time_array = np.hstack((time_array, pk))
            fired_PMT = np.hstack((fired_PMT, ch*np.ones(np.size(pk))))
        fired_PMT = fired_PMT.astype(int)
        # initial result

        fit(pe_array, time_array, fired_PMT, PMT_pos, recondata)

    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', help="input and selection table output")
args = psr.parse_args()

recon(args.ipt, args.opt, ReadPMT())
