import numpy as np
import h5py
import tables

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
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    h1.close()
    #print(x.shape, EventID.shape, np.unique(EventID).shape, np.std(y),np.sum(x**2+y**2+z**2>0.1))
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    #print(np.sum(dn_index))
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
                
                #print(cnt, ID, EventID.shape,(np.unique(EventID)).shape)
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    #print(x.shape, EventID.shape, np.unique(EventID).shape,np.std(y),np.sum(x**2+y**2+z**2>0.1))
    return (EventID, ChannelID, x, y, z)

def readchain(radius, path, axis):
    for i in np.arange(0, 3):
        if(i == 0):
            #filename = path + '1t_' + radius + '.h5'
            filename = '%s1t_%s_%s.h5' % (path, radius, axis)
            EventID, ChannelID, x, y, z = readfile(filename)
        else:
            try:
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
    data = []
    PMT_pos = ReadPMT()
    ra = -0.0001
    EventID, ChannelID, x, y, z = readchain('%+.3f' % ra, '/mnt/stage/douwei/Simulation/1t_root/2.0MeV_dn/', axis)

    size = np.size(np.unique(EventID))
    print(size)
    print(type(size))
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
    print(total_pe.shape, np.size(np.unique(EventID)))
    return np.float(size), data[0]

size_x, ax0 = CalMean('x')
size_y, ay0 = CalMean('y')
size_z, az0 = CalMean('z')

mean = (size_x*ax0 + size_y*ay0 + size_z*az0)/(size_x+size_y+size_z)
base = np.exp(np.mean(np.log(mean)))
correct = mean/base
print(correct)

with h5py.File('base.h5','w') as out:
    out.create_dataset('base', data = base)
    out.create_dataset('correct', data = correct)