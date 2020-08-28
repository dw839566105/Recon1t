import numpy as np 
import scipy, h5py
import tables
import sys
from scipy.optimize import minimize
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
np.set_printoptions(formatter={'float': '{: 0.3f}'.format})

def LoadBase():
    path = './base.h5'
    h1 = tables.open_file(path)
    base = h1.root.correct[:]
    h1.close()
    return base

#base = np.log(LoadBase())

def ReadPMT():
    # Read PMT position
    # output: 2d PMT position 30*3 (x, y, z)
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

def Calib(theta, *args):
    total_pe, PMT_pos, cut, LegendreCoeff = args
    y = total_pe
    '''
    print(LegendreCoeff.shape)
    print(theta.shape)
    print(np.dot(LegendreCoeff, theta))
    print((np.dot(LegendreCoeff, theta)).shape)
    '''
    #print(np.dot(LegendreCoeff, theta).shape)
    #a = np.tile(base, (1, np.int(np.size(LegendreCoeff)/np.size(base)/np.size(theta))))[0,:]
    #print(a.shape)
    # corr = np.dot(LegendreCoeff, theta) + np.tile(base, (1, np.int(np.size(LegendreCoeff)/np.size(base)/np.size(theta))))[0,:]
    corr = np.dot(LegendreCoeff, theta)
    # Poisson regression
    L0 = - np.sum(np.sum(np.transpose(y)*np.transpose(corr) \
        - np.transpose(np.exp(corr))))

    # L = L0 + np.exp(np.sum(np.abs(theta)))
    # L = L0 + 0.01*2e5*np.exp(np.sum(np.abs(theta)))
    L = L0/(2*np.size(y)) + 0.001 * np.sum(np.abs(theta)) # standard L-0 norm
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
    try:
        h1 = tables.open_file(filename,'r')
    except:
        exit()
    print(filename)
    truthtable = h1.root.GroundTruth
    EventID = truthtable[:]['EventID']
    ChannelID = truthtable[:]['ChannelID']
    
    x = h1.root.TruthData[:]['x']
    y = h1.root.TruthData[:]['y']
    z = h1.root.TruthData[:]['z']
    #print(x.shape)
    #print(np.sum(x**2+y**2+z**2<0.1))
    #exit()
    h1.close()
    #print(x.shape, EventID.shape, np.unique(EventID).shape, np.std(y),np.sum(x**2+y**2+z**2>0.1))
    dn = np.where((x==0) & (y==0) & (z==0))
    dn_index = (x==0) & (y==0) & (z==0)
    #print(np.sum(dn_index))
    try:
        pin = dn[0] + np.min(EventID)
    except:
        pin = dn[0]
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
                exit()
    #print(x, y, z)
    return EventID, ChannelID, x, y, z
    
def main_Calib(radius, path, fout, cut_max):
    print('begin read file')
    #filename = '/mnt/stage/douwei/Simulation/1t_root/1.5MeV_015/1t_' + radius + '.h5'
    with h5py.File(fout,'w') as out:
        # read files by table
        # positive direction
        if(eval(radius) != 0.00):
            EventIDx, ChannelIDx, xx, yx, zx = readchain('+' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('+' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('+' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            try:
                x1 = np.array((xx[0], yx[0], xz[0]))
                y1 = np.array((xy[0], yy[0], zy[0]))
                z1 = np.array((xz[0], yz[0], zz[0]))
            except:
                exit()
            sizex_p = np.size(np.unique(EventIDx))
            sizey_p = np.size(np.unique(EventIDy))
            sizez_p = np.size(np.unique(EventIDz))

            EventID_p = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID_p = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_p = np.hstack((xx, xy, xz))
            y_p = np.hstack((yx, yy, yz))
            z_p = np.hstack((xz, yz, zz))

            # negative direction
            EventIDx, ChannelIDx, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('-' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            x2 = np.array((xx[0], yx[0], xz[0]))
            y2 = np.array((xy[0], yy[0], zy[0]))
            z2 = np.array((xz[0], yz[0], zz[0]))

            EventID_n = np.hstack((EventIDx, EventIDy, EventIDz)) + np.max(EventID_p)
            ChannelID_n = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_n = np.hstack((xx, xy, xz))
            y_n = np.hstack((yx, yy, yz))
            z_n = np.hstack((xz, yz, zz))

            # written total_pe into a column
            sizex_n = np.size(np.unique(EventIDx))
            sizey_n = np.size(np.unique(EventIDy))
            sizez_n = np.size(np.unique(EventIDz))

            # gather
            EventID = np.hstack((EventID_p, EventID_n))
            ChannelID = np.hstack((ChannelID_p, ChannelID_n))
            x = np.hstack((x_p, x_n))
            y = np.hstack((y_p, y_n))
            z = np.hstack((z_p, z_n))         

            #print(sizex_p,sizex_n,sizey_p,sizey_n,sizez_p, sizez_n)

            size = np.size(np.unique(EventID))
            #print(size)
        else:
            EventIDx, ChannelIDx, xx, yx, zx = readchain('-' + radius, path, 'x')
            EventIDy, ChannelIDy, xy, yy, zy = readchain('-' + radius, path, 'y')
            EventIDz, ChannelIDz, xz, yz, zz = readchain('-' + radius, path, 'z')
            EventIDy = EventIDy + np.max(EventIDx)
            EventIDz = EventIDz + np.max(EventIDy)
            x1 = np.array((xx[0], yx[0], xz[0]))
            y1 = np.array((xy[0], yy[0], zy[0]))
            z1 = np.array((xz[0], yz[0], zz[0]))
            sizex_p = np.size(np.unique(EventIDx))
            sizey_p = np.size(np.unique(EventIDy))
            sizez_p = np.size(np.unique(EventIDz))

            EventID_p = np.hstack((EventIDx, EventIDy, EventIDz))
            ChannelID_p = np.hstack((ChannelIDx, ChannelIDy, ChannelIDz))
            x_p = np.hstack((xx, xy, xz))
            y_p = np.hstack((yx, yy, yz))
            z_p = np.hstack((xz, yz, zz))
            
            EventID = EventID_p
            ChannelID = ChannelID_p
            x = x_p
            y = y_p
            z = z_p 
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
        
        if(eval(radius) != 0.00):
            tmp_x_p = Legendre_coeff(PMT_pos, x1/1e3, cut_max)
            tmp_x_p = np.tile(tmp_x_p, (sizex_p,1))
            tmp_y_p = Legendre_coeff(PMT_pos, y1/1e3, cut_max)
            tmp_y_p = np.tile(tmp_y_p, (sizey_p,1))
            tmp_z_p = Legendre_coeff(PMT_pos, z1/1e3, cut_max)
            tmp_z_p = np.tile(tmp_z_p, (sizez_p,1))
            tmp_x_n = Legendre_coeff(PMT_pos, x2/1e3, cut_max)
            tmp_x_n = np.tile(tmp_x_n, (sizex_n,1))
            tmp_y_n = Legendre_coeff(PMT_pos, y2/1e3, cut_max)
            tmp_y_n = np.tile(tmp_y_n, (sizey_n,1))
            tmp_z_n = Legendre_coeff(PMT_pos, z2/1e3, cut_max)
            tmp_z_n = np.tile(tmp_z_n, (sizez_n,1))              
            #print(xx[0],xy[0],xz[0])
            #print(yx[0],yy[0],yz[0])
            #print(zx[0],zy[0],zz[0])
            #print(tmp_x_p.shape)
            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p, tmp_x_n, tmp_y_n, tmp_z_n))
            #print(tmp_x_p.shape, tmp_y_p.shape, tmp_z_p.shape, tmp_x_n.shape, tmp_y_n.shape, tmp_z_n.shape)
        else:
            tmp_x_p = Legendre_coeff(PMT_pos, x1/1e3, cut_max)
            tmp_x_p = np.tile(tmp_x_p, (sizex_p,1))
            tmp_y_p = Legendre_coeff(PMT_pos, y1/1e3, cut_max)
            tmp_y_p = np.tile(tmp_y_p, (sizey_p,1))
            tmp_z_p = Legendre_coeff(PMT_pos, z1/1e3, cut_max)
            tmp_z_p = np.tile(tmp_z_p, (sizez_p,1))
            LegendreCoeff = np.vstack((tmp_x_p, tmp_y_p, tmp_z_p))
 
        # print(np.size(np.unique(EventID)), total_pe.shape, LegendreCoeff.shape)
               
        # this part for EM
        '''
        LegendreCoeff = np.zeros((0,cut_max))           
        for k in np.arange(size):
            LegendreCoeff = np.vstack((LegendreCoeff,Legendre_coeff(PMT_pos,np.array((x[k], y[k], z[k]))/1e3, cut_max)))
       ''' 
        print('begin get coeff')
        print(total_pe.shape, total_pe)
        print(LegendreCoeff.shape, LegendreCoeff)
        
        vs = np.reshape(total_pe[0:sizex_p*30],(-1,30), order='C')
        mean = np.mean(vs, axis=0)        

        for cut in np.arange(2,cut_max,1):
            X = LegendreCoeff[:,0:cut]
            y = total_pe
            alpha = 0.001
            reg = TweedieRegressor(power=1, alpha=alpha, link='log',fit_intercept=False, max_iter=1000,tol=1e-8 )
            reg.fit(X, y)
            prd = reg.predict(X[0:30,0:cut+1])
            print('%d-th intercept:\n' % cut, reg.intercept_,'\n')
            print('%d-th coeff:\n' % cut, reg.coef_,'\n')
            print('%d-th predict:\n' % cut, prd,'\n')
            print('Mean hit:\n', mean,'\n')
            
            corr = np.dot(LegendreCoeff[:,0:cut+1], np.hstack((reg.intercept_, reg.coef_)))
            L0 = np.transpose(y)*np.transpose(corr) \
                - np.transpose(np.exp(corr))
            aic = 2*(cut) - 2*np.sum(L0)
            print('%d-th AIC:\n' % cut, aic,'\n')
            print('Saving file...')
            coeff = np.hstack((reg.intercept_,reg.coef_))
            out.create_dataset('coeff' + str(cut), data = coeff)
            out.create_dataset('mean' + str(cut), data = mean)
            out.create_dataset('aic' + str(cut), data = mean)
            out.create_dataset('predict' + str(cut), data = prd)
## read data from calib files
PMT_pos = ReadPMT()
# sys.argv[1]: '%s' radius
# sys.argv[2]: '%s' path
# sys.argv[3]: '%s' output
# sys.argv[4]: '%d' cut
main_Calib(sys.argv[1],sys.argv[2], sys.argv[3], eval(sys.argv[4]))
