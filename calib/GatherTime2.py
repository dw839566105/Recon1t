import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def readtime_sparse(path):
    coeff_time = []
    ra = np.arange(0.01,0.56,0.01)
    for i in ra:
        h = tables.open_file(path + '/file_%+.3f.h5' % i)
        coeff_time.append(h.root.coeff9[:])
        h.close()
    coeff_time = np.array(coeff_time)
    return ra, coeff_time.T

def readtime_compact(path):
    coeff_time = []
    ra = np.arange(0.55,0.650,0.002)
    for i in ra:
        h = tables.open_file(path + '/file_%+.3f.h5' % i)
        coeff_time.append(h.root.coeff9[:])
        h.close()
    coeff_time = np.array(coeff_time)
    return ra, coeff_time.T

def main(order=5, fit_order=10, qt=0.01):
    rd1, coeff_time1 = readtime_sparse('./coeff_time_1t_reflection0.05_2MeV_%s_1' % qt)
    rd2, coeff_time2 = readtime_compact('./coeff_time_1t_compact_%s' % qt)
    
    rd = np.hstack((rd1[:-1], rd2))
    coeff_time2 = (coeff_time2.T - coeff_time2[:,0] + coeff_time1[:,-1]).T
    coeff_time = np.hstack((coeff_time1[:,:-1], coeff_time2))
    coeff_L = np.zeros((order, fit_order + 1))

    for i in np.arange(order):
        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                  np.hstack((coeff_time[i], coeff_time[i])), \
                                                  deg = fit_order, full = True)

        y = np.polynomial.legendre.legval(rd/np.max(rd), B)

        coeff_L[i] = B

        plt.figure(num = i+1, dpi = 300)
        #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
        plt.plot(rd, coeff_time[i], 'r.', label='real',linewidth=2)
        plt.plot(rd, y, label = 'Legendre')
        plt.xlabel('radius/m')
        plt.ylabel('Time Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        plt.legend()
        plt.savefig('coeff_Time2_%d.png' % i)
        plt.close()
    return coeff_L

coeff_L = main(eval(sys.argv[1]), eval(sys.argv[2]), sys.argv[3])  
with h5py.File('./Time_coeff2_1t_%s.h5' % sys.argv[3],'w') as out:
    out.create_dataset('coeff_L', data = coeff_L)
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)
