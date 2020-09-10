'''
This easy script is to calculate the difference of PMT
Using points generated at the detector center
Calucalte the mean of the log expect
The base is the exp of the mean
Then the correction is the mean/base to make sure

    sum(corr) = 0

Note that in Poisson regression, just add the corr (not times)

If PMT are all the same, this script can be ignored
'''

import numpy as np
import h5py

def ReadPMT():
    '''
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
    '''
    f = open(r"../PMT_1t.txt")
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
    print(filename)
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
    for i in np.arange(0, 3): # Although we have many files, but the calulation is too slowwwww
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

def CalMean(axis):
    '''
    # a simple program to calculate the mean hit
    # input: axis: str 'x' or 'y' or 'z'
    # output: size: event total number
    '''
    data = []
    PMT_pos = ReadPMT()
    ra = -0.0001 # the sim file is -0.000, QAQ
    EventID, ChannelID, x, y, z = readchain('%+.3f' % ra, '/mnt/stage/douwei/Simulation/1t_root/2.0MeV_dn/', axis)

    size = np.size(np.unique(EventID))
    total_pe = np.zeros(np.size(PMT_pos[:,0])*size)

    for k_index, k in enumerate(np.unique(EventID)):
            if not k_index % 1e4:
                print('preprocessing %d-th event' % k_index)
            hit = ChannelID[EventID == k]
            tabulate = np.bincount(hit)
            event_pe = np.zeros(np.size(PMT_pos[:,0]))
            # tabulate begin with 0
            event_pe[0:np.size(tabulate)] = tabulate
            total_pe[(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = event_pe
            # although it will be repeated, mainly for later EM:
            # vertex[0,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = x[k_index]
            # vertex[1,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = y[k_index]
            # vertex[2,(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = z[k_index]
            # total_pe[(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = event_pe
    data.append(np.mean((np.reshape(total_pe, (30,-1), order='F')), axis=1))
    return size, data

size_x, ax0 = CalMean('x')
size_y, ay0 = CalMean('y')
size_z, az0 = CalMean('z')
# mean of x, y, z
mean = (size_x*ax0 + size_y*ay0 + size_z*az0)/(size_x+size_y+size_z)
# from exp to linear
base = np.exp(np.mean(np.log(mean)))
# relative value
correct = mean/base

# write file
with h5py.File('base.h5','w') as out:
    out.create_dataset('base', data = base)
    out.create_dataset('correct', data = correct)
