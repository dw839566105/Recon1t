import os, sys
import tables
import numpy as np
import h5py
from scipy.optimize import curve_fit

#def odd_func(x, a, b, c, d, e):
#    return a * x**1 + b * x**3 + c * x**5 + d * x**7 + e * x**9
#
#def even_func(x, a, b, c, d, e):
#    return a * x**0 + b * x**2 + c * x**4 + d * x**6 + e * x**8

# fit odd order, even order is 0
def odd_func(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**1 + b * x**3 + c * x**5 + d * x**7 + e * x**9 + f * x**11 + g * x**13 + h * x**15 + i*x**17 + j*x**19
# fit even order, even order is 0
def even_func(x, a, b, c, d, e, f, g, h, i, j):
    return a * x**0 + b * x**2 + c * x**4 + d * x**6 + e * x**8 + f * x**10 + g * x**12 + h * x**14 + i*x**16 + j*x**18

def findfile(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    
    coeff = 'coeff' + str(order)
   
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    if(eval(radius)<0):
        data[1::2] = - data[1::2]
    return data

def main(path, upperlimit, lowerlimit, order_max):
    
    ra = np.arange(upperlimit + 1e-5, lowerlimit, -0.01)
    for order in np.arange(5, order_max, 5):
        coeff = []
        rd = []
        for radius in ra:
            str_radius = '%.2f' % radius
            k = findfile(path, str_radius, order)
            rd.append(np.array(radius))
            coeff = np.hstack((coeff, k))

        coeff = np.reshape(coeff,(-1,np.size(rd)),order='F')
        #print(coeff)
        #ft= np.reshape(ft,(-1,np.size(ra)),order='F')
        #ch= np.reshape(ch,(-1,np.size(ra)),order='F')
        #predict = np.reshape(predict,(-1,np.size(ra)),order='F')

        N_max = np.size(coeff[:,0])
        bd_1 = 0.85
        #bd_2l = 0.50 
        #bd_2r = 0.80
        bd_3 = 0.8

        fit_max = 10
        k1 = np.zeros((N_max+1, fit_max))
        k2 = np.zeros((N_max+1, fit_max))
        for i in np.arange(np.size(coeff[:,0])):
            data = np.nan_to_num(coeff[i,:])
            x = ra/0.65

            index1 = (x<=bd_1) & (x>=-bd_1) & (x!=0)

            if(i%2==1):
                popt1, pcov = curve_fit(odd_func, x[index1], data[index1])
                output1 = odd_func(x[index1], *popt1)
            else:
                popt1, pcov = curve_fit(even_func, x[index1], data[index1])
                output1 = even_func(x[index1], *popt1)
            '''
            index2 = (np.abs(x)<=bd_2r) & (np.abs(x)>=bd_2l)
            if(i%2==1):
                popt2, pcov = curve_fit(odd_func, x[index2], data[index2])
                output2 = odd_func(x[index2], *popt2)
            else:
                popt2, pcov = curve_fit(even_func, x[index2], data[index2])
                output2 = even_func(x[index2], *popt2)
            '''
            index3 = (x >= bd_3) | (x <= - bd_3)
            if(i%2==1):
                popt3, pcov = curve_fit(odd_func, x[index3], data[index3])
                output3 = odd_func(x[index3], *popt3)
            else:
                popt3, pcov = curve_fit(even_func, x[index3], data[index3])
                output3 = even_func(x[index3], *popt3)

            #x_total = np.hstack((x[index1],x[index2],x[index3]))
            #y_total = np.hstack((output1,output2,output3))
            x_total = np.hstack((x[index1],x[index3]))
            y_total = np.hstack((output1,output3))
            index = np.argsort(x_total)

            k1[i,:] = popt1
            k2[i,:] = popt3
            
        with h5py.File('./Time_coeff_1t' + str(order) + '.h5','w') as out:
            out.create_dataset('coeff', data = coeff)
            out.create_dataset('poly_in', data = k1)
            out.create_dataset('poly_out', data = k2)
    
path = sys.argv[1]
upperlimit = eval(sys.argv[2])
lowerlimit = eval(sys.argv[3])
order_max = eval(sys.argv[4])
main(path, upperlimit, lowerlimit, order_max)
