import tables
import numpy as np
import h5py

def readmean():
    r1 = np.arange(0.01, 0.40, 0.01)
    r2 = np.arange(0.40, 0.65, 0.002)
    r = np.hstack((r1,r2))
    ra = []
    data = []
    for radius in r:
        h1 = tables.open_file("../mean/file_%.3f+x.h5" % radius)
        try:
            a = h1.root.mean[:]
            data.append(a)
            ra.append(radius)
        except:
            print(radius)
            pass
        h1.close()
    return ra, data

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
            filename = '%s1t_%s_%s.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
                # filename = path + '1t_' + radius + '_n.h5'
                # eg: /mnt/stage/douwei/Simulation/1t_root/2.0MeV_xyz/1t_+0.030_1.h5
                filename = '%s1t_%s_%s_%d.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass

    return EventID, ChannelID, x, y, z

def main(path, radius):
    axis = 'x'
    sign = '+'
    EventIDx, ChannelIDx, xx, yx, zx = readchain(sign + radius, path, axis)
    x1 = np.array((xx[0], yx[0], zx[0]))
    size = np.size(np.unique(EventIDx))

    EventID = EventIDx
    ChannelID = ChannelIDx

    total = np.zeros(size)
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    ra, data = readmean()
    data = np.array(data)
    ra = np.array(ra)
    for k_index, k in enumerate(np.unique(EventID)):
        if not k_index % 1e3:
            print('preprocessing %d-th event' % k_index, flush=True)
        hit = ChannelID[EventID == k]
        tabulate = np.bincount(hit)
        event_pe = np.zeros(30)
        # tabulate begin with 0
        event_pe[0:np.size(tabulate)] = tabulate
        
        rep = np.tile(event_pe,(np.size(ra),1))
        real_sum = np.sum(data, axis=1)
        corr = (data.T/(real_sum/np.sum(event_pe))).T
        
        L = np.sum(-corr + np.log(corr)*event_pe, axis=1)

        index = np.where(L == np.max(L))[0]

        total[k_index] = ra[index]
        
    with h5py.File('../MCmean/file%s.h5' % radius,'w') as out:
        out.create_dataset('result', data = total)
        
import sys        
path = sys.argv[1]
radius = sys.argv[2]
main(path, radius)
        
        
        
