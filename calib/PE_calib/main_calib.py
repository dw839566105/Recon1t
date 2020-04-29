import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt

def Calib(theta, *args):
    total_pe, PMT_pos, cut, LegendreCoeff = args
    y = total_pe
    '''
    print(LegendreCoeff.shape)
    print(theta.shape)
    print(np.dot(LegendreCoeff, theta))
    print((np.dot(LegendreCoeff, theta)).shape)
    '''
    # Poisson regression
    L0 = - np.sum(np.sum(np.transpose(y)*np.transpose(np.dot(LegendreCoeff, theta)) \
        - np.transpose(np.exp(np.dot(LegendreCoeff, theta)))))

    L = L0 + np.exp(np.sum(np.abs(theta)))
    # L = L0 + 0.01*2e5*np.exp(np.sum(np.abs(theta)))
    # L = L0 + np.sum(np.abs(theta)) # standard L-0 norm
    # L = L0 * np.exp(np.sum(np.abs(theta)))
    return L

def Legendre_coeff(PMT_pos, vertex, cut):
    size = np.size(PMT_pos[:,0])
    if(np.sum(vertex**2) > 1e-6):
        cos_theta = np.sum(vertex*PMT_pos,axis=1)\
            /np.sqrt(np.sum(vertex**2)*np.sum(PMT_pos**2,axis=1))
    else:
        cos_theta = np.ones(size)
    x = np.zeros((size, cut))
    # legendre coeff
    for i in np.arange(0,cut):
        c = np.zeros(cut)
        c[i] = 1
        x[:,i] = LG.legval(cos_theta,c)
    return x  

def hessian(x, *args):
    total_pe, PMT_pos, cut, LegendreCoeff= args
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
    h1 = tables.open_file(filename,'r')
    print(filename)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
        
    h1.close()

    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    if(np.sum(x**2+y**2+z**2<0.1)>0):
        cnt = 0        
        for ID in np.arange(np.min(EventID), np.max(EventID)+1):
            if ID in dn[0] + np.min(EventID):
                cnt = cnt+1
                #print('Trigger No:', EventID[EventID==ID])
                #print('Fired PMT', ChannelID[EventID==ID])
                
                ChannelID = ChannelID[~(EventID == ID)]
                EventID = EventID[~(EventID == ID)]
                
                #print(cnt, ID, EventID.shape,(np.unique(EventID)).shape)
    x = x[~dn_index]
    y = y[~dn_index]
    z = z[~dn_index]
    return (EventID, ChannelID, x, y, z)
    
def readchain(radius, path, axis):
    for i in np.arange(0,20):
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
    
def main_Calib(radius, path, fout, cut_max):
    print('begin read file')
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5'
    with h5py.File(fout,'w') as out:
        # read files by table
        EventIDx, ChannelIDx, xx, yx, zx = readchain(radius, path, 'x')
        EventIDy, ChannelIDy, xy, yy, zy = readchain(radius, path, 'y')
        EventIDz, ChannelIDz, xz, yz, zz = readchain(radius, path, 'z')

        EventIDy = EventIDy + np.max(EventIDx)
        EventIDz = EventIDz + np.max(EventIDy)
        
        EventID = np.hstack((EventIDx, EventIDy, EventIDz))
        ChannelID = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
        x = np.hstack((xx, xy, xz))
        y = np.hstack((yx, yy, yz))
        z = np.hstack((xz, yz, zz))

        # written total_pe into a column
        sizex = np.size(np.unique(EventIDx))
        sizey = np.size(np.unique(EventIDy))
        sizez = np.size(np.unique(EventIDz))
        size = np.size(np.unique(EventID))
        #print(sizex,sizey,sizez,size)
        total_pe = np.zeros(np.size(PMT_pos[:,0])*size)
        vertex = np.zeros((3,np.size(PMT_pos[:,0])*size))
        print('total event: %d' % np.size(np.unique(EventID)))
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
            total_pe[(k_index) * np.size(PMT_pos[:,0]) : (k_index + 1) * np.size(PMT_pos[:,0])] = event_pe

        print('begin processing legendre coeff')
        # this part for the same vertex
        tmp_x = Legendre_coeff(PMT_pos,np.array((xx[0], yx[0], zx[0]))/1e3, cut_max)
        tmp_x = np.tile(tmp_x, (sizex,1))
        tmp_y = Legendre_coeff(PMT_pos,np.array((xy[0], yy[0], zy[0]))/1e3, cut_max)
        tmp_y = np.tile(tmp_y, (sizey,1))
        tmp_z = Legendre_coeff(PMT_pos,np.array((xz[0], yz[0], zz[0]))/1e3, cut_max)
        tmp_z = np.tile(tmp_z, (sizez,1))
        LegendreCoeff = np.vstack((tmp_x, tmp_y, tmp_z))
        # print(np.size(np.unique(EventID)), total_pe.shape, LegendreCoeff.shape)
               
        # this part for EM
        '''
        LegendreCoeff = np.zeros((0,cut_max))           
        for k in np.arange(size):
            LegendreCoeff = np.vstack((LegendreCoeff,Legendre_coeff(PMT_pos,np.array((x[k], y[k], z[k]))/1e3, cut_max)))
       ''' 
        print('begin get coeff')
        
        for cut in np.arange(5,cut_max,5):
            
            theta0 = np.zeros(cut) # initial value
            theta0[0] = 0.8 + np.log(2)
            #print(total_pe.shape, total_pe)
            #print(LegendreCoeff.shape, LegendreCoeff)
            result = minimize(Calib, theta0, method='SLSQP', args = (total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut]))  
            record = np.array(result.x, dtype=float)
            
            H = hessian(result.x, *(total_pe, PMT_pos, cut, LegendreCoeff[:,0:cut]))
            H_I = np.linalg.pinv(np.matrix(H))
            vs = np.reshape(total_pe[0:sizex*30],(-1,30), order='C')
            mean = np.mean(vs, axis=0)
            args = (total_pe, PMT_pos, cut)
            predict = [];
            predict.append(np.exp(np.dot(LegendreCoeff[0:30,0:cut], result.x)))
            print(mean)
            print(predict)
            predict = np.transpose(predict)
            #chi2sq = 2*np.sum(- total_pe + predict + np.nan_to_num(total_pe*np.log(total_pe/predict)), axis=1)/(np.max(EventID)-30)

            # print(np.dot(x, result.x) - mean)
            # print(np.size(total_pe,1))

            print(record)

            out.create_dataset('coeff' + str(cut), data = record)
            out.create_dataset('mean' + str(cut), data = mean)
            out.create_dataset('predict' + str(cut), data = predict)
            out.create_dataset('rate' + str(cut), data = np.size(total_pe)/30)
            out.create_dataset('hinv' + str(cut), data = H_I)
            #out.create_dataset('chi' + str(cut), data = chi2sq)

## read data from calib files
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
