import tables
import h5py
import numpy as np
import matplotlib.pyplot as plt
import sys

print('Get data')
data = np.zeros((30,0))
vertex = np.zeros((3,0))

No = eval(sys.argv[1])
fout = sys.argv[2]
for i in np.arange((No-1)*10, No*10):
    h = tables.open_file('file%d.h5' % (i+1))
    if(np.size(h.root.mean)/30 == np.size(h.root.vertex[:])/3):
        data = np.hstack((data, h.root.mean[:].T))
        vertex = np.hstack((vertex, h.root.vertex[:].T))
    else:
        print(i+1)
    h.close()
    
vertex = vertex.T
data = data.T

v = np.zeros_like(vertex)
v[:,0] = np.sqrt(np.sum(vertex**2,axis=1))/1000
v[:,1] = np.arccos(vertex[:,2]/1000/(v[:,0]+1e-6))
v[:,2] = np.arctan(vertex[:,1]/(vertex[:,0]+1e-6)) + (vertex[:,0]<0)*np.pi

N = 30
H_r, edges_r = np.histogram(v[:,0]**3, bins = N)
H_t, edges_t = np.histogram(np.cos(v[:,1]), bins = N)
H_p, edges_p = np.histogram(v[:,2], bins = N)

print('hist')
bins = np.zeros((N**3, 3))
mean = np.zeros((N**3, 31))
cnt = 0
for i1, i in enumerate(edges_r[0:-1]):
    for j1, j in enumerate(edges_t[0:-1]):
        for k1, k in enumerate(edges_p[0:-1]):
            bins[cnt, :] = np.array((i,j,k))

            if not cnt % 10:
                print(cnt)
            index = (v[:,0]**3>edges_r[i1]) & (v[:,0]**3<edges_r[i1+1]) & \
                (np.cos(v[:,1])>edges_t[j1]) & (np.cos(v[:,1])<edges_t[j1+1]) & \
                (v[:,2]>edges_p[k1]) & (v[:,2]<edges_p[k1+1])
            mean[cnt, 0:-1] = np.sum(data[index],axis=0)
            mean[cnt, -1] = np.sum(index)
            cnt = cnt + 1

with h5py.File(fout,'w') as out:   
    out.create_dataset('mean', data = mean)
    out.create_dataset('vertex', data = bins)