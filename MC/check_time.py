import tables
import numpy as np
import h5py

h1 = tables.open_file("./template.h5")
tp = h1.root.template[:]
bins = np.vstack((h1.root.x[:],h1.root.y[:], h1.root.z[:])).T
h1.close()

h = tables.open_file('../calib/Time_coeff_1t.h5','r')
coeff_time_in = h.root.coeff_L_in[:]
coeff_time_out = h.root.coeff_L_out[:]
h.close()
cut_time, fitcut_time = coeff_time_in.shape

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

    # legendre coeff by polynomials
    k = np.zeros((1,cut))
    if(z>=bd_pe):
        k[0] = LG.legval(z, coeff_time_out.T)
    else:
        k[0] = LG.legval(z, coeff_time_in.T)
    
    k[0,0] = vertex[4]
    T_i = np.dot(x, np.transpose(k))
    L = np.nansum(Likelihood_quantile(y, T_i[:,0], 0.05, 1.25))
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

def readfile(filename):
    '''
    # Read single file
    # input: filename [.h5]
    # output: EventID, ChannelID, x, y, z
    '''
    h1 = tables.open_file(filename,'r')
    print(filename, flush=True)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    print(h1.root.GroundTruth[:]['PulseTime'])
    exit()
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    
    # The following part is to avoid trigger by dn(dark noise) since threshold is 1
    # These thiggers will be recorded as (0,0,0) by uproot
    # but in root, the truth and the trigger is not one to one
    # If the simulation vertex is (0,0,0), it is ambiguous, so we need cut off (0,0,0) or use data without dn
    # If the simulation set -dn 0, whether the program will get into the following part is not tested
    
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    pin = dn[0] + np.min(EventID)
    if(np.sum(x**2+y**2+z**2>0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in pin:
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    return (EventID, ChannelID, x, y, z)
    
def readchain(radius, path, axis):
    '''
    # This program is to read series files
    # Since root file will recorded as 'filename.root', if too large, it will use '_n' as suffix
    # input: radius: %+.3f, 'str'
    #        path: file storage path, 'str'
    #        axis: 'x' or 'y' or 'z', 'str'
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    for i in np.arange(0, 5):
        if(i == 0):
            # filename = path + '1t_' + radius + '.h5'
            # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030.h5
            filename = '%s1t_%s_%sQ.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%dQ.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, x, y, z

def main(path, radius, axis, sign='+'):
    EventIDx, ChannelIDx, xx, yx, zx = readchain(sign + radius, path, axis)
    x1 = np.array((xx[0], yx[0], zx[0]))
    size = np.size(np.unique(EventIDx))

    EventID = EventIDx
    ChannelID = ChannelIDx

    total = np.zeros((size,3))
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    
    data = tp

    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(bins[:,0]),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        L = np.nansum(-corr + np.log(corr)*event_pe, axis=1)
        index = np.where(L == np.max(L))[0][0]
        
        total[k_index,:] = bins[index]/1000
        
    with h5py.File('./MCmean/file%s%s%s.h5' % (sign, axis, radius),'w') as out:
        out.create_dataset('result', data = total)
        
import sys        
path = sys.argv[1]
radius = sys.argv[2]
axis = sys.argv[3]
sign = sys.argv[4]
main(path, radius, axis, sign)
        
        
        
