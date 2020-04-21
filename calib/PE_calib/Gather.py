import os, sys
import tables
import numpy as np
import h5py
from scipy.optimize import curve_fit

def odd_func(x, a, b, c, d, e):
    return a * x**1 + b * x**3 + c * x**5 + d * x**7 + e * x**9

def even_func(x, a, b, c, d, e):
    return a * x**0 + b * x**2 + c * x**4 + d * x**6 + e * x**8

def findfile(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    
    coeff = 'coeff' + str(order)
    mean = 'mean' + str(order)
    predict = 'predict' +str(order)
    rate = 'rate' + str(order)
    hinv = 'hinv' + str(order)
    chi = 'chi' + str(order)
    
    a = eval('np.array(h.root.'+ coeff + '[:])')
    b = eval('np.array(h.root.'+ mean + '[:])')
    c = eval('np.array(h.root.'+ predict + '[:])')
    try:
        d = eval('h.root.'+ rate + '[:])')
    except:
        d = np.array(0)
    e = eval('np.array(h.root.'+ hinv + '[:])')
    f = eval('np.array(h.root.'+ chi + '[:])')
    
    data.append(np.array(np.array((a,b,c,d,e,f))))
    h.close()
    return data

def main(path, upperlimit, lowerlimit, order_max):
    
    ra = np.arange(upperlimit + 1e-5, lowerlimit, -0.01)
    for order in np.arange(5, order_max, 5):
        coeff = []
        mean = []
        predict = []
        rate = []
        hinv = []
        chi = []
        for radius in ra:
            str_radius = '%+.2f' % radius
            k = findfile(path, str_radius, order)
            k.append(np.array(radius))
            coeff = np.hstack((coeff,np.array(k[0][0])))
            mean = np.hstack((mean,np.array(k[0][1])))
            predict = np.hstack((predict,np.array(k[0][2][:,0])))
            #rate = np.hstack((rate,np.array(k[0][3])))
            #hinv = np.hstack((hinv,np.array(k[0][4])))
            chi = np.hstack((chi,np.array(k[0][5])))

        coeff = np.reshape(coeff,(-1,np.size(ra)),order='F')
        mean = np.reshape(mean,(-1,np.size(ra)),order='F')
        predict = np.reshape(predict,(-1,np.size(ra)),order='F')
        chi = np.reshape(chi,(-1,np.size(ra)),order='F')

        N_max = np.size(coeff[:,0])
        bd_1 = 0.75
        bd_2l = 0.50 
        bd_2r = 0.80
        bd_3 = 0.7

        fit_max = 5
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

            index2 = (np.abs(x)<=bd_2r) & (np.abs(x)>=bd_2l)
            if(i%2==1):
                popt2, pcov = curve_fit(odd_func, x[index2], data[index2])
                output2 = odd_func(x[index2], *popt2)
            else:
                popt2, pcov = curve_fit(even_func, x[index2], data[index2])
                output2 = even_func(x[index2], *popt2)

            index3 = (x >= bd_3) | (x <= - bd_3)
            if(i%2==1):
                popt3, pcov = curve_fit(odd_func, x[index3], data[index3])
                output3 = odd_func(x[index3], *popt3)
            else:
                popt3, pcov = curve_fit(even_func, x[index3], data[index3])
                output3 = even_func(x[index3], *popt3)

            x_total = np.hstack((x[index1],x[index2],x[index3]))
            y_total = np.hstack((output1,output2,output3))
            x_total = np.hstack((x[index1],x[index3]))
            y_total = np.hstack((output1,output3))
            index = np.argsort(x_total)

            k1[i,:] = popt1
            k2[i,:] = popt3
            
        with h5py.File('./PE_coeff_1t' + str(order) + '.h5','w') as out:
            out.create_dataset('coeff', data = coeff)
            out.create_dataset('mean', data = mean)
            out.create_dataset('predict', data = predict)
            out.create_dataset('chi', data = chi)
            out.create_dataset('in', data = k1)
            out.create_dataset('out', data = k2)
    
path = sys.argv[1]
upperlimit = eval(sys.argv[2])
lowerlimit = eval(sys.argv[3])
order_max = eval(sys.argv[4])
main(path, upperlimit, lowerlimit, order_max)