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

def main(order=5, fit_order=10, qt=0.1):
    #rd, coeff_pe = readtime('./coeff_time_1t_shell_200000_%s' % qt)
    rd, coeff_pe = readtime('./coeff_time_1t_reflection0.05_%s_2' % qt)
    coeff_L_in = np.zeros((order, fit_order + 1))
    coeff_L_out = np.zeros((order, fit_order + 1))
    coeff_p_in = np.zeros((order, fit_order + 1))
    coeff_p_out = np.zeros((order, fit_order + 1))
    bd = 0.88
    d = np.where(np.abs(rd-np.max(rd)*bd) == np.min(np.abs(rd-np.max(rd)*bd)))
    deg = fit_order

    for i in np.arange(order):
        index_in = (rd<=rd[d[0][0]])
        index_out = (rd>=rd[d[0][0]])

        if not i % 2:
            w_in = np.ones_like(rd[index_in])
            w_out = np.ones_like(rd[index_out])
            w_in[-1] = 1000
            w_out[0] = 1000
            if(i==0):
                    w_in[-1] = 1000
            # Legendre coeff

            B_in, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])), \
                                                      deg = deg, w = np.hstack((w_in,w_in)), full = True)

            y1_in = np.polynomial.legendre.legval(rd[index_in]/np.max(rd), B_in)

            B_out, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])), \
                                                      deg = deg, w = np.hstack((w_out,w_out)), full = True)
            y1_out = np.polynomial.legendre.legval(rd[index_out]/np.max(rd), B_out)
            # polynormial coeff
            C_in = np.polyfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))), \
                              np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])), \
                              deg=fit_order)
            y2_in = np.polyval(C_in, rd[index_in]/np.max(rd))

            C_out = np.polyfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))), \
                      np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])), \
                      deg=fit_order)
            y2_out = np.polyval(C_out, rd[index_out]/np.max(rd))

        else:
            B_in, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_in], -coeff_pe[i,index_in])), \
                                                      deg=fit_order, w = np.hstack((w_in,w_in)), full=True)
            y1_in = np.polynomial.legendre.legval(rd[index_in]/np.max(rd), B_in)

            B_out, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_out], -coeff_pe[i,index_out])), \
                                                      deg=fit_order, w = np.hstack((w_out,w_out)), full=True)
            y1_out = np.polynomial.legendre.legval(rd[index_out]/np.max(rd), B_out)
            C_in = np.polyfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))), \
                      np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])), \
                      deg=fit_order)
            y2_in = np.polyval(C_in, rd[index_in]/np.max(rd))

            C_out = np.polyfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))), \
                      np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])), \
                      deg=fit_order)
            y2_out = np.polyval(C_out, rd[index_out]/np.max(rd))
 
        coeff_L_in[i] = B_in
        coeff_L_out[i] = B_out
        coeff_p_in[i] = C_in
        coeff_p_out[i] = C_out
        plt.figure(num = i+1, dpi = 300)
        #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
        plt.plot(rd, coeff_pe[i], 'r.', label='real',linewidth=2)
        plt.plot(rd, np.hstack((y1_in, y1_out[1:])), label = 'Legendre')
        plt.xlabel('radius/m')
        plt.ylabel('Time Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        plt.legend()
        plt.savefig('coeff_Time_%d_%s.png' % (i,qt))
    return coeff_L_in, coeff_L_out, coeff_p_in, coeff_p_out, bd

coeff_L_in, coeff_L_out, coeff_p_in, coeff_p_out, bd = main(eval(sys.argv[1]), eval(sys.argv[2]), sys.argv[3])  
with h5py.File('./Time_coeff_1t_%s.h5' % sys.argv[3],'w') as out:
    out.create_dataset('coeff_L_in', data = coeff_L_in)
    out.create_dataset('coeff_L_out', data = coeff_L_out)
    out.create_dataset('coeff_p_in', data = coeff_p_in)
    out.create_dataset('coeff_p_out', data = coeff_p_out)
    out.create_dataset('bd', data = bd) 
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)
