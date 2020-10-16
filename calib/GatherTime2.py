import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def readtime(path):
    coeff_time = []
    ra = np.arange(0.01,0.640,0.01)
    for i in ra:
        h = tables.open_file(path + '/file_%+.3f.h5' % i)
        coeff_time.append(h.root.coeff5[:])
        h.close()
    coeff_time = np.array(coeff_time)
    '''
    for i in np.arange(5):
        plt.figure(num=i+1, dpi=300)
        plt.plot(ra[1:-2], coeff_time[1:-2,i])
        plt.xlabel('Radius/m')
        plt.ylabel('Legendre Coefficient of Time Info')
        plt.title('Time %d-th order, quantile = %.2f' % (i,0.01) )
        plt.savefig('/mnt/stage/douwei/Simulation/1t_root/fig/Time%d.pdf' % i)
        plt.show()
    '''
    return ra, coeff_time.T

def main(order=5, fit_order=10, qt=0.01):
    rd, coeff_pe = readtime('./coeff_time_1t_8.0MeV_shell_%s' % qt)
    rd = np.array(rd)
    coeff_pe = np.array(coeff_pe)
    coeff_L = np.zeros((order, fit_order + 1))

    for i in np.arange(order):
        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))), \
                                                  np.hstack((coeff_pe[i], coeff_pe[i])), \
                                                  deg = fit_order, full = True)

        y = np.polynomial.legendre.legval(rd/np.max(rd), B)

        coeff_L[i] = B

        plt.figure(num = i+1, dpi = 300)
        #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
        plt.plot(rd, coeff_pe[i], 'r.', label='real',linewidth=2)
        plt.plot(rd, y, label = 'Legendre')
        plt.xlabel('radius/m')
        plt.ylabel('Time Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        plt.legend()
        plt.savefig('coeff_Time2_%d.png' % i)
    return coeff_L

coeff_L = main(eval(sys.argv[1]), eval(sys.argv[2]), sys.argv[3])  
with h5py.File('./Time_coeff2_1t_%s.h5' % sys.argv[3],'w') as out:
    out.create_dataset('coeff_L', data = coeff_L)
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)
