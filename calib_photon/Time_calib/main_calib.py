import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt

def Calib(theta, *args):
    ChannelID, flight_time, PMT_pos, cut, LegendreCoeff = args
    y = flight_time
    T_i = np.dot(LegendreCoeff, theta)
    # quantile regression
    # quantile = 0.01
    L0 = Likelihood_quantile(y, T_i, 0.01, 0.3)
    # L = L0
    L = L0 + np.sum(np.abs(theta))
    return L

def Likelihood_quantile(y, T_i, tau, ts):
    less = T_i[y<T_i] - y[y<T_i]
    more = y[y>=T_i] - T_i[y>=T_i]
    R = (1-tau)*np.sum(less) + tau*np.sum(more)
    #log_Likelihood = exp
    return R

def Legendre_coeff(PMT_pos, vertex, cut):
    size = np.size(PMT_pos[:,0])
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos,axis=1) \
            /np.sqrt(np.sum(vertex**2)*np.sum(PMT_pos**2,axis=1))
    else:
        cos_theta = np.ones(size)
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0, cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    return x  

def hessian(x, *args):
    ChannelID, PETime, PMT_pos, cut, LegendreCoeff = args
    H = np.zeros((len(x),len(x)))
    h = 1e-3
    k = 1e-3
    for i in np.arange(len(x)):
        for j in np.arange(len(x)):
            if (i != j):
                delta1 = np.zeros(len(x))
                delta1[i] = h
                delta1[j] = k
                delta2 = np.zeros(len(x))
                delta2[i] = -h
                delta2[j] = k


                L1 = - Calib(x + delta1, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L2 = - Calib(x - delta1, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L3 = - Calib(x + delta2, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L4 = - Calib(x - delta2, *(total_pe, PMT_pos, cut, LegendreCoeff))
                H[i,j] = (L1+L2-L3-L4)/(4*h*k)
            else:
                delta = np.zeros(len(x))
                delta[i] = h
                L1 = - Calib(x + delta, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L2 = - Calib(x - delta, *(total_pe, PMT_pos, cut, LegendreCoeff))
                L3 = - Calib(x, *(total_pe, PMT_pos, cut, LegendreCoeff))
                H[i,j] = (L1+L2-2*L3)/h**2                
    return H

def readfile(filename):
# read files by table
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    PETime = truthtable[:]['PETime']
    photonTime = truthtable[:]['photonTime']
    PulseTime = truthtable[:]['PulseTime']
    dETime = truthtable[:]['dETime']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    
    h1.close()
    
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
                PETime = PETime[~(EventID == ID)]
                photonTime = photonTime[~(EventID == ID)]
                PulseTime = PulseTime[~(EventID == ID)]
                dETime = dETime[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
                
        x = x[~dn_index]
        y = y[~dn_index]
        z = z[~dn_index]
    
    return (EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z)

def readchain(radius, path, axis):
    for i in np.arange(0,5):
        if(i == 0):
            #filename = path + '1t_' + radius + '.h5'
            filename = '%s1t_%s_%s.h5' % (path, radius, axis)
            EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z = readfile(filename)
        else:
            try:
                filename = '%s1t_%s_%s_%d.h5' % (path, radius, axis, i)
                EventID1, ChannelID1, PETime1, photonTime1, PulseTime1, dETime1, x1, y1, z1 = readfile(filename)
                EventID = np.hstack((EventID, EventID1))
                ChannelID = np.hstack((ChannelID, ChannelID1))
                PETime = np.hstack((PETime,PETime1))
                photonTime = np.hstack((photonTime, photonTime1))
                PulseTime = np.hstack((PulseTime, PulseTime1))
                dETime = np.hstack((dETime, dETime1))
                x = np.hstack((x, x1))
                y = np.hstack((y, y1))
                z = np.hstack((z, z1))
            except:
                pass
    return EventID, ChannelID, PETime, photonTime, PulseTime, dETime, x, y, z

def main_Calib(radius, path, fout, cut_max):
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5' 
    with h5py.File(fout,'w') as out:
        # read files by table
        if(eval(radius) != 0.00):
            EventIDx, ChannelIDx1, PETimex, photonTimex, PulseTimex, dETimex, xx, yx, zx = readchain('+' + radius, path, 'x')
            EventIDy, ChannelIDy1, PETimey, photonTimey, PulseTimey, dETimey, xy, yy, zy = readchain('+' + radius, path, 'y')
            EventIDz, ChannelIDz1, PETimez, photonTimez, PulseTimez, dETimez, xz, yz, zz = readchain('+' + radius, path, 'z')

            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)

            x1 = np.array((xx[0], yx[0], xz[0]))
            y1 = np.array((xy[0], yy[0], zy[0]))
            z1 = np.array((xz[0], yz[0], zz[0])) 

            # dark noise (np.unique EventID should not change since the pure trigger by dark noise has been filtered!)
            flight_timex = PulseTimex - dETimex        
            EventIDx = EventIDx[~(flight_timex==0)]
            ChannelIDx1 = ChannelIDx1[~(flight_timex==0)]
            flight_timex = flight_timex[~(flight_timex==0)]

            flight_timey = PulseTimey - dETimey       
            EventIDy = EventIDy[~(flight_timey==0)]
            ChannelIDy1 = ChannelIDy1[~(flight_timey==0)]
            flight_timey = flight_timey[~(flight_timey==0)]

            flight_timez = PulseTimez - dETimez        
            EventIDz = EventIDz[~(flight_timez==0)]
            ChannelIDz1 = ChannelIDz1[~(flight_timez==0)]
            flight_timez = flight_timez[~(flight_timez==0)]

            EventID1 = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID1 = np.hstack((ChannelIDx1, ChannelIDy1, ChannelIDz1))
            PETime1 = np.hstack((PETimex, PETimey, PETimez))
            photonTime1 = np.hstack((photonTimex, photonTimey, photonTimez))
            PulseTime1 = np.hstack((PulseTimex, PulseTimey, PulseTimez))
            dETime1 = np.hstack((dETimex, dETimey, dETimez))
            flight_time1 = np.hstack((flight_timex, flight_timey, flight_timez))

            # read files by table
            EventIDx, ChannelIDx2, PETimex, photonTimex, PulseTimex, dETimex, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy2, PETimey, photonTimey, PulseTimey, dETimey, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz2, PETimez, photonTimez, PulseTimez, dETimez, xz, yz, zz = readchain('-' + radius, path, 'z')

            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)

            x2 = np.array((xx[0], yx[0], xz[0]))
            y2 = np.array((xy[0], yy[0], zy[0]))
            z2 = np.array((xz[0], yz[0], zz[0])) 

            # dark noise (np.unique EventID should not change since the pure trigger by dark noise has been filtered!)
            flight_timex = PulseTimex - dETimex        
            EventIDx = EventIDx[~(flight_timex==0)]
            ChannelIDx2 = ChannelIDx2[~(flight_timex==0)]
            flight_timex = flight_timex[~(flight_timex==0)]

            flight_timey = PulseTimey - dETimey       
            EventIDy = EventIDy[~(flight_timey==0)]
            ChannelIDy2 = ChannelIDy2[~(flight_timey==0)]
            flight_timey = flight_timey[~(flight_timey==0)]

            flight_timez = PulseTimez - dETimez        
            EventIDz = EventIDz[~(flight_timez==0)]
            ChannelIDz2 = ChannelIDz2[~(flight_timez==0)]
            flight_timez = flight_timez[~(flight_timez==0)]

            EventID2 = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID2 = np.hstack((ChannelIDx2, ChannelIDy2, ChannelIDz2))
            PETime2 = np.hstack((PETimex, PETimey, PETimez))
            photonTime2 = np.hstack((photonTimex, photonTimey, photonTimez))
            PulseTime2 = np.hstack((PulseTimex, PulseTimey, PulseTimez))
            dETime2 = np.hstack((dETimex, dETimey, dETimez))
            flight_time2 = np.hstack((flight_timex, flight_timey, flight_timez))

            x = np.hstack((xx, xy, xz))
            y = np.hstack((yx, yy, yz))
            z = np.hstack((xz, yz, zz))
            
            EventID = np.hstack((EventID1, EventID2))
            ChannelID = np.hstack((ChannelID1, ChannelID2))
            PETime = np.hstack((PETime1, PETime2))
            photonTime = np.hstack((photonTime1, photonTime2))
            PulseTime = np.hstack((PulseTime1, PulseTime2))
            dETime = np.hstack((dETime1, dETime2))
            flight_time = np.hstack((flight_time1, flight_time2))
            
            print('begin processing legendre coeff')
            # this part for the same vertex
            tmp_x_p = Legendre_coeff(PMT_pos,x1/1e3, cut_max)
            tmp_x_p = tmp_x_p[ChannelIDx1]
            tmp_y_p = Legendre_coeff(PMT_pos,y1/1e3, cut_max)
            tmp_y_p = tmp_y_p[ChannelIDy1]
            tmp_z_p = Legendre_coeff(PMT_pos,z1/1e3, cut_max)
            tmp_z_p = tmp_z_p[ChannelIDz1]  
            tmp_x_n = Legendre_coeff(PMT_pos,x2/1e3, cut_max)
            tmp_x_n = tmp_x_n[ChannelIDx2]
            tmp_y_n = Legendre_coeff(PMT_pos,y2/1e3, cut_max)
            tmp_y_n = tmp_y_n[ChannelIDy2]
            tmp_z_n = Legendre_coeff(PMT_pos,z2/1e3, cut_max)
            tmp_z_n = tmp_z_n[ChannelIDz2]
            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p,tmp_x_n, tmp_y_n, tmp_z_n))
        else:
            EventIDx, ChannelIDx1, PETimex, photonTimex, PulseTimex, dETimex, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy1, PETimey, photonTimey, PulseTimey, dETimey, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz1, PETimez, photonTimez, PulseTimez, dETimez, xz, yz, zz = readchain('-' + radius, path, 'z')

            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)

            x1 = np.array((xx[0], yx[0], xz[0]))
            y1 = np.array((xy[0], yy[0], zy[0]))
            z1 = np.array((xz[0], yz[0], zz[0])) 

            # dark noise (np.unique EventID should not change since the pure trigger by dark noise has been filtered!)
            flight_timex = PulseTimex - dETimex        
            EventIDx = EventIDx[~(flight_timex==0)]
            ChannelIDx1 = ChannelIDx1[~(flight_timex==0)]
            flight_timex = flight_timex[~(flight_timex==0)]

            flight_timey = PulseTimey - dETimey       
            EventIDy = EventIDy[~(flight_timey==0)]
            ChannelIDy1 = ChannelIDy1[~(flight_timey==0)]
            flight_timey = flight_timey[~(flight_timey==0)]

            flight_timez = PulseTimez - dETimez        
            EventIDz = EventIDz[~(flight_timez==0)]
            ChannelIDz1 = ChannelIDz1[~(flight_timez==0)]
            flight_timez = flight_timez[~(flight_timez==0)]

            EventID1 = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID1 = np.hstack((ChannelIDx1, ChannelIDy1, ChannelIDz1))
            PETime1 = np.hstack((PETimex, PETimey, PETimez))
            photonTime1 = np.hstack((photonTimex, photonTimey, photonTimez))
            PulseTime1 = np.hstack((PulseTimex, PulseTimey, PulseTimez))
            dETime1 = np.hstack((dETimex, dETimey, dETimez))
            flight_time1 = np.hstack((flight_timex, flight_timey, flight_timez))
            
            EventID = EventID1
            ChannelID = ChannelID1
            PETime = PETime1
            photonTime = photonTime1
            PulseTime = PulseTime1
            dETime = dETime1
            flight_time = flight_time1

            print('begin processing legendre coeff')
            # this part for the same vertex
            tmp_x_p = Legendre_coeff(PMT_pos,x1/1e3, cut_max)
            tmp_x_p = tmp_x_p[ChannelIDx1]
            tmp_y_p = Legendre_coeff(PMT_pos,y1/1e3, cut_max)
            tmp_y_p = tmp_y_p[ChannelIDy1]
            tmp_z_p = Legendre_coeff(PMT_pos,z1/1e3, cut_max)
            tmp_z_p = tmp_z_p[ChannelIDz1]
            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p))
            
        print(ChannelID.shape, LegendreCoeff.shape)
        # this part for EM
        '''
        LegendreCoeff = np.zeros((0,cut_max))
        for k_index, k in enumerate(np.unique(EventID)):
            tmp = Legendre_coeff(PMT_pos,np.array((x[k_index], y[k_index], z[k_index]))/1e3, cut_max)
            single_index = ChannelID[EventID==k]
            LegendreCoeff = np.vstack((LegendreCoeff,tmp[single_index]))
       '''
        print('finish calc coeff')
        
        for cut in np.arange(5,cut_max + 5,5):

            theta0 = np.zeros(cut) # initial value
            theta0[0] = np.mean(flight_time) - 26
            result = minimize(Calib,theta0, method='SLSQP',args = (ChannelID, flight_time, PMT_pos, cut, LegendreCoeff[:,0:cut]))  
            record = np.array(result.x, dtype=float)
            print(result.x)
            
            out.create_dataset('coeff' + str(cut), data = record)


f = open(r'./PMT_1t.txt')
line = f.readline()
data_list = []
while line:
    num = list(map(float,line.split()))
    data_list.append(num)
    line = f.readline()
f.close()
PMT_pos = np.array(data_list)

# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3], eval(sys.argv[4]))
