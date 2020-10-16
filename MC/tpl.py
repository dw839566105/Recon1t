#!/usr/bin/python
# -*- coding:utf-8 -*-
import numpy as np
import tables
import matplotlib.pyplot as plt
import sys
import h5py
np.set_printoptions(precision=4, suppress=True)
def main(radius):
    Q = x = y = z = np.empty(0)
    for i in np.arange(300):
        if(i==0):
            h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+%.3fQ.h5' % radius)
            Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
            x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
            y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
            z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
            h.close()
        elif(i>0):
            try:
                h = tables.open_file('/mnt/stage/douwei/Simulation/1t_root/reflection0.05/1t_+%.3f_%dQ.h5' % (radius, i))
                Q = np.hstack((Q, np.array(h.root.PETruthData[:]['Q'])))
                x = np.hstack((x, np.array(h.root.TruthData[:]['x'])))
                y = np.hstack((y, np.array(h.root.TruthData[:]['y'])))
                z = np.hstack((z, np.array(h.root.TruthData[:]['z'])))
                h.close()
            except:
                break


    r = np.sqrt(x**2+y**2+z**2)
    cos_theta = z/(r+1e-6)
    phi = np.arctan2(y,x)
    index1 = np.digitize(cos_theta, np.linspace(-1,1,31)) 
    index2 = np.digitize(phi, np.linspace(-np.pi,np.pi,31))
    data = []
    xt = []
    yt = []
    zt = []
    plt.figure(dpi=300)
    for i in np.arange(1,31):
        for j in np.arange(1,31):
            Q1 = np.reshape(Q,(-1,30))
            pe = np.mean(Q1[(index1==i) & (index2==j)], axis = 0)
            data.append(pe)
            if(np.sum((index1==i) & (index2==j)))==0:
                print(radius, i, j)
            xt.append(np.mean(x[(index1==i) & (index2==j)]))
            yt.append(np.mean(y[(index1==i) & (index2==j)]))
            zt.append(np.mean(z[(index1==i) & (index2==j)]))
            #print(x[(index1==i) & (index2==j)])
            #print(y[(index1==i) & (index2==j)])
            #print(z[(index1==i) & (index2==j)])
            #plt.scatter(np.sqrt(x[(index1==i) & (index2==j)]**2 + y[(index1==i) & (index2==j)]**2), z[(index1==i) & (index2==j)], s=0.2)
            plt.scatter(x[(index1==i) & (index2==j)], y[(index1==i) & (index2==j)], s=1)
            plt.scatter(xt[-1], yt[-1], s=1.5, color='r')
            #print(i, j, np.array(xt[-1]), np.array(yt[-1]), np.array(zt[-1]))
        print(np.sum(index1==i))
    data = np.array(data)
    xt = np.array(xt)
    yt = np.array(yt)
    zt = np.array(zt)
    with h5py.File('./Tpl_%.3f.h5' % radius,'w') as out:
        out.create_dataset('x', data = xt)
        out.create_dataset('y', data = yt)
        out.create_dataset('z', data = zt)
        out.create_dataset('template', data = data)
        
main(eval(sys.argv[1]))
