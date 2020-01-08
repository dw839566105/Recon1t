import numpy as np 
import h5py, sys
import matplotlib.pyplot as plt
import tables
import ROOT, uproot
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy.optimize import minimize

def ReconSph():
    fun = lambda vertex: calib(vertex)
    return fun

def calib(vertex):
    global coeff, PMT_pos, event_pe, cut
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

    k = np.zeros((np.size(coeff[0,:])))
    #print(np.size(coeff[0,:]))
    for i in np.arange(0,np.size(coeff[0,:])):
        # polyfit
        # fitfun = np.poly1d(coeff[:,i])
        # k[i] = fitfun(z)
        # cubic interp

        xx = np.arange(-1,1,0.01)
        yy = np.zeros(np.size(xx))
        # print(np.where(np.abs(x+0.63)<1e-5))
        # print(np.where(np.abs(x-0.64)<1e-5))
        # z = y[np.where(np.abs(x+0.63)<1e-5):np.where(np.abs(x-0.64)<1e-5)]
        # z = y[37:164]
        # print(z.shape)
        # y[np.where(np.where(np.abs(x+0.63)<1e-5)):np.where(np.where(np.abs(x-0.64)<1e-5))] = 
        yy[37:164] = coeff[:,i]
        if z>0.99:
            z = 0.99
        elif z<-0.99:
            z = -0.99
        # print(z)
        f = interpolate.interp1d(xx, yy, kind='cubic')
        k[i] = f(z)
        k[0] = k[0] + np.log(vertex[0])
    # print(k) 
    # print('haha')
    expect = np.exp(np.dot(x,k))
    L = - np.sum(np.sum(np.log((expect**y)*np.exp(-expect))))
    return L

def con():
    cons = ({'type': 'ineq', 'fun': lambda x: x[0] - 0.01},\
    {'type': 'ineq', 'fun': lambda x: 0.65**2 - (x[1]**2 + x[2]**2+x[3]**2)})
    return cons

def recon(fid, fout):
    global PMT_pos, coeff, event_pe, event_count
    '''
    reconstruction

    fid: root reference file
    fout: output file in this step
    '''

    # Create the output file and the group

    rootfile = ROOT.TFile(fid)
    #TruthChain = rootfile.Get('SimTriggerInfo')
    print(fid)
    '''
    class ChargeData(tables.IsDescription):
        ChannelID = tables.Float64Col(pos=0)
        Time = tables.Float16Col(pos=1)
        PE = tables.Float16Col(pos=2)
        Charge = tables.Float16Col(pos=2)
    '''
    class ReconData(tables.IsDescription):
        EventID = tables.Int64Col(pos=0)
        x = tables.Float16Col(pos=1)
        y = tables.Float16Col(pos=2)
        z = tables.Float16Col(pos=3)
        t0 = tables.Float16Col(pos=4)
        E = tables.Float16Col(pos=5)
        tau_d = tables.Float16Col(pos=6)
        success = tables.Int64Col(pos=7)

    # Create the output file and the group
    h5file = tables.open_file(fout, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"

    # Create tables
    '''
    ChargeTable = h5file.create_table(group, "Charge", ChargeData, "Charge")
    Charge = ChargeTable.row
    '''
    ReconTable = h5file.create_table(group, "Recon", ReconData, "Recon")
    recondata = ReconTable.row
    '''
    # Loop for ROOT files. 
    data = ROOT.TChain("SimpleAnalysis")
    data.Add(fid)
    '''
    # Loop for event
    '''
    for event in data:
        print(event)
        EventID = event.TriggerNo
        print(EventID)
    '''  
    '''  
    psr = argparse.ArgumentParser()
    psr.add_argument("-o", dest='opt', help="output")
    psr.add_argument('ipt', help="input")
    args = psr.parse_args()

    f = uproot.open(args.ipt)
    '''
    result_total = np.empty((1,4))
    record = np.zeros((1,4))
    h = h5py.File('../JP_python/version3/calib/coeff_corr.h5','r')
    coeff = h['coeff_corr'][...]
    f = uproot.open(fid)
    a = f['SimpleAnalysis']
    for tot, chl, PEl, Pkl, nPl in zip(a.array("TotalPE"),
                    a.array("ChannelInfo.ChannelId"),
                    a.array('ChannelInfo.PE'),
                    a.array('ChannelInfo.PeakLoc'),
                    a.array('ChannelInfo.nPeaks')):

    #print("=== TotalPE: {} ===".format(tot))
    #for ch, PE, pk, np in zip(chl, PEl, Pkl, nPl):
    #   print(ch, PE, pk, np)
        CH = np.zeros(np.size(PMT_pos[:,1]))
        PE = np.zeros(np.size(PMT_pos[:,1]))
        fired_PMT = np.zeros(0)
        TIME = np.zeros(0)
        for ch, pe, pk, npk in zip(chl, PEl, Pkl, nPl):
            PE[ch] = pe
            TIME = np.hstack((TIME, pk))
            fired_PMT = np.hstack((fired_PMT, ch*np.ones(np.size(pk))))
        # print(TIME, fired_PMT)
        fired_PMT = fired_PMT.astype(int)
        time_array = TIME
        
        '''
        for ChannelInfo in event.ChannelInfo:
            Charge['ChannelID'] = ChargeInfo.ChannelID
            Charge['Time'] =  ChannelInfo.Peak
            Charge['PE'] =  ChannelInfo.PE
            Charge['Charge'] =  ChannelInfo.Charge
            Charge.append()

            PE = ChannelInfo.nPeaks
            Time =  ChannelInfo.Peak
            ChannelID = ChargeInfo.ChannelID
        '''

        result_recon = np.empty((0,6))
        result_drc = np.empty((0,3))
        result_tdrc = np.empty((0,3))

        # initial value
        x0 = np.zeros((1,4))
        x0[0][0] = PE.sum()/300
        x0[0][1] = np.sum(PE*PMT_pos[:,0])/np.sum(PE)
        x0[0][2] = np.sum(PE*PMT_pos[:,1])/np.sum(PE)
        x0[0][3] = np.sum(PE*PMT_pos[:,2])/np.sum(PE)

        # Constraints
        event_pe = PE
        # x0 = np.sum(PE*PMT_pos,axis=0)/np.sum(PE)
        theta0 = np.array([1,0.1,0.1,0.1])
        theta0[0] = x0[0][0]
        theta0[1] = x0[0][1]
        theta0[2] = x0[0][2]
        theta0[3] = x0[0][3]
        cons = con()
        result = minimize(ReconSph(),theta0, method='SLSQP',constraints=cons)  
        record[0,:] = np.array(result.x, dtype=float)
        result_total = np.vstack((result_total,record))
        
        # result
        print(event_count, result.x, result.success)
        event_count = event_count + 1
        recondata['EventID'] = event_count
        recondata['x'] = result.x[1]
        recondata['y'] = result.x[2]
        recondata['z'] = result.x[3]
        recondata['E'] = result.x[0]
        recondata['success'] = result.success
        recondata.append()

        # print(np.sum(result_drc*truth_px)/np.sqrt(np.sum(result_drc**2)*np.sum(truth_px**2)))

    # Flush into the output file
    # ChargeTable.flush()
    ReconTable.flush()
    h5file.close()

## read data from calib files
global PMT_pos, cut, event_count
f = open(r"PMT1t.txt")
line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
PMT_pos = np.array(data_list)

event_count = 0
cut = 7 # Legend order
recon(sys.argv[1],sys.argv[2])
