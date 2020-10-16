import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py
from numpy.polynomial import legendre as LG
def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon(path, order):
    ra = np.arange(0.01, 0.65, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe
def main(order=15, fit_order=10):
    rd, coeff_pe = main_photon('coeff_pe_1t_reflection0.00_30/',order)
    rd = np.array(rd)
    coeff_pe = np.array(coeff_pe)
    #coeff_pe[1][51:58] = np.interp(rd[51:58],np.array((rd[51], rd[57])), np.array((coeff_pe[1][51],coeff_pe[1][57])))
    coeff_L_in = np.zeros((order, fit_order + 1))
    coeff_L_out = np.zeros((order, fit_order + 1))
    coeff_p_in = np.zeros((order, fit_order + 1))
    coeff_p_out = np.zeros((order, fit_order + 1))
    bd = 0.91
    d = np.where(np.abs(rd-np.max(rd)*bd) == np.min(np.abs(rd-np.max(rd)*bd)))

    for i in np.arange(order):
        index_in = (rd<=rd[d[0][0]])
        index_out = (rd>=rd[d[0][0]])
        if not i % 2:
            w_in = np.ones_like(rd[index_in])
            w_out = np.ones_like(rd[index_out])
            w_in[-1] = 1000
            w_out[0] = 1000
            #if(i==0):
            #    w_in[-2:] = 1000
            # Legendre coeff
            B_in, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_in]/np.max(rd),-rd[index_in]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_in], coeff_pe[i,index_in])), \
                                                      deg = fit_order, w = np.hstack((w_in,w_in)), full = True)

            y1_in = np.polynomial.legendre.legval(rd[index_in]/np.max(rd), B_in)

            B_out, tmp = np.polynomial.legendre.legfit(np.hstack((rd[index_out]/np.max(rd),-rd[index_out]/np.max(rd))), \
                                                      np.hstack((coeff_pe[i,index_out], coeff_pe[i,index_out])), \
                                                      deg = fit_order, w = np.hstack((w_out,w_out)), full = True)
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
        #plt.plot(rd, np.hstack((y1_in, y1_out[1:])), label = 'Legendre')
        plt.axvline(rd[index_out][0])
        plt.plot(rd[index_in], np.hstack((y1_in)), label = 'Legendre')
        plt.plot(rd[index_out], np.hstack((y1_out)), label = 'Legendre')
        
        tmp = []
        for z in np.arange(0,0.64,0.001):
            if(z/0.64>=bd):
                k = LG.legval(z/0.64, B_out)
            else:
                k = LG.legval(z/0.64, B_in)
            #print(k)
            tmp.append(k)
        tmp = np.array(tmp)
        plt.plot(np.arange(0,0.64,0.001), tmp)
        plt.xlabel('radius/m')
        plt.ylabel('PE Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        #plt.legend()
        plt.savefig('coeff_PE_%d.png' % i)
    return coeff_L_in, coeff_L_out, coeff_p_in, coeff_p_out, bd

coeff_L_in, coeff_L_out, coeff_p_in, coeff_p_out, bd = main(eval(sys.argv[1]), eval(sys.argv[2]))  
with h5py.File('./PE_coeff_1t.h5','w') as out:
    out.create_dataset('coeff_L_in', data = coeff_L_in)
    out.create_dataset('coeff_L_out', data = coeff_L_out)
    out.create_dataset('coeff_p_in', data = coeff_p_in)
    out.create_dataset('coeff_p_out', data = coeff_p_out)
    out.create_dataset('bd', data = bd) 
#A = np.polynomial.legendre.Legendre.fit(np.array((-0.5,0.4)), np.array((0.5,0.6)),deg=5)
