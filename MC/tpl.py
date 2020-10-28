#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import tables
import matplotlib.pyplot as plt
import sys
import h5py
np.set_printoptions(precision=4, suppress=True)
def main(radius):
    Q = np.empty(0)
    x = np.empty(0)
    y = np.empty(0)
    z = np.empty(0)
    Ch = np.empty(0)
    Time = np.empty(0)
    Event = np.empty(0)
    for i in np.arange(300):
        if(i==0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+%.3fQ.h5' % radius)
            Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
            x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
            y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
            z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
            Event = np.hstack((Event, np.array(h.root.GroundTruth[:]['EventID'])))
            Ch = np.hstack((Ch, np.array(h.root.GroundTruth[:]['ChannelID'])))
            Time = np.hstack((Time, np.array(h.root.GroundTruth[:]['PulseTime'])))
            
            h.close()
        elif(i>0):
            try:
                h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+%.3f_%dQ.h5' % (radius, i))
                Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
                x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
                y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
                z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
                Event = np.hstack((Event, np.array(h.root.GroundTruth[:]['EventID'])))
                Ch = np.hstack((Ch, np.array(h.root.GroundTruth[:]['ChannelID'])))
                Time = np.hstack((Time, np.array(h.root.GroundTruth[:]['PulseTime'])))
                h.close()
            except:
                break

    Ch.astype('int')
    r = np.sqrt(x**2+y**2+z**2)
    cos_theta = z/(r+1e-6)
    phi = np.arctan2(y,x)
    index1 = np.digitize(cos_theta, np.linspace(-1,1,31)) 
    index2 = np.digitize(phi, np.linspace(-np.pi,np.pi,31))
    data = []
    data1 = []
    xt = []
    yt = []
    zt = []
    plt.figure(dpi=300)
    for i in np.arange(1,31):
        print(f'processing {i}-th cos theta')
        for j in np.arange(1,31):
            Q1 = np.reshape(Q,(-1,30))
            pe = np.mean(Q1[(index1==i) & (index2==j)], axis = 0)
            data.append(pe)
            time = np.zeros(30)
            
            Ev = np.unique(Event)[(index1==i) & (index2==j)]
            for k in np.arange(30):
                Ei = np.flatnonzero(np.isin(Event, Ev))
                ci = np.where(Ch==k)[0]
                si = Ei[np.isin(Ei,ci)]
                #print(np.quantile(Time[si], 0.1))
                #print(Ch[si])
                #print(Event[si])
                if np.size(si)>0:
                    time[k] = np.quantile(Time[si], 0.1)
            
            data1.append(time)
            if(np.sum((index1==i) & (index2==j)))==0:
                print(radius, i, j)
            xt.append(np.mean(x[(index1==i) & (index2==j)]))
            yt.append(np.mean(y[(index1==i) & (index2==j)]))
            zt.append(np.mean(z[(index1==i) & (index2==j)]))

            plt.scatter(x[(index1==i) & (index2==j)], y[(index1==i) & (index2==j)], s=1)
            plt.scatter(xt[-1], yt[-1], s=1.5, color='r')
            
        print(np.sum(index1==i))
    data = np.array(data)
    data1 = np.array(data1)
    xt = np.array(xt)
    yt = np.array(yt)
    zt = np.array(zt)
    with h5py.File('./Tpl_%.3f.h5' % radius,'w') as out:
        out.create_dataset('x', data = xt)
        out.create_dataset('y', data = yt)
        out.create_dataset('z', data = zt)
        out.create_dataset('template', data = data)
        out.create_dataset('time', data = data1)
        
main(eval(sys.argv[1]))
