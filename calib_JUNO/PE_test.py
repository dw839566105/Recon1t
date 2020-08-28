from scipy.optimize import minimize
import tables
import numpy as np
import matplotlib.pyplot as plt
import tables
import h5py

def LoadDataPE(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    if(eval(radius)<0):
        data[1::2] = - data[1::2]
    return data

def odd_func_poly(theta, *args):
    x, y, w = args
    size = np.size(theta)
    pw = np.arange(1,2*size+1,2)
    pw[0] = 1
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return np.sum(w*res**2) + 0.0001 * np.sum(np.abs(theta))

def even_func_poly(theta, *args):
    x, y, w = args
    size = np.size(theta)
    pw = np.arange(0,2*size,2)
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return np.sum(w*res**2) + 0.0001 * np.sum(np.abs(theta))

def odd_func_Legendre(theta, *args):
    x, y, w = args
    size = np.size(theta)
    s = np.zeros_like(y)
    for i in np.arange(size):
        k = np.zeros(2*size)
        k[2*i+1] = 1
        X = np.polynomial.legendre.legval(x, k)
        s = s + np.sum(X*theta[i], axis=0)
    res = y - s
    return np.sum(w*res**2)

def even_func_Legendre(theta, *args):
    x, y, w = args
    size = np.size(theta)
    s = np.zeros_like(y)
    for i in np.arange(size):
        k = np.zeros(2*size)
        k[2*i] = 1
        X = np.polynomial.legendre.legval(x, k)
        s = s + np.sum(X*theta[i], axis=0)
    res = y - s
    return np.sum(w*res**2)

def odd_func_poly1(theta, *args):
    x, y = args
    size = np.size(theta)
    pw = np.arange(1,2*size+1,2)
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return s

def even_func_poly1(theta, *args):
    x, y = args
    size = np.size(theta)
    pw = np.arange(0,2*size,2)
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return s

def odd_func_Legendre1(theta, *args):
    x, y = args
    size = np.size(theta)
    s = np.zeros_like(y)
    for i in np.arange(size):
        k = np.zeros(2*size)
        k[2*i+1] = 1
        X = np.polynomial.legendre.legval(x, k)
        s = s + np.sum(X*theta[i], axis=0)
    res = y - s
    return s

def even_func_Legendre1(theta, *args):
    x, y = args
    size = np.size(theta)
    s = np.zeros_like(y)
    for i in np.arange(size):
        k = np.zeros(2*size)
        k[2*i] = 1
        X = np.polynomial.legendre.legval(x, k)
        s = s + np.sum(X*theta[i], axis=0)
    res = y - s
    return s

## gather the data
path = './coeff_pe_1t_2.0MeV_dns_Lasso_els4/'
ra = np.hstack((np.arange(0.01, 0.40, 0.01), np.arange(0.40, 0.65, 0.002)))

order = 15
coeff_pe = []
rd = []
for radius in ra:
    str_radius = '%.3f' % radius
    k = LoadDataPE(path, str_radius, order)
    rd.append(np.array(radius))
    coeff_pe = np.hstack((coeff_pe, k))

coeff_pe = np.reshape(coeff_pe,(-1,np.size(ra)),order='F')
with h5py.File('./PE_coeff_1t_seg.h5','w') as out:
    for i in np.arange(np.size(coeff_pe[:,0])):
        print('Processing %d-th order' % i)
        plt.figure(num=i)
        data = np.nan_to_num(coeff_pe[i,:])
        x = ra/0.65
        plt.plot(x, data,'.',label='Raw')
        for j_index, j in enumerate(data):
            if((np.abs(data[j_index])>0.03) & (np.sum(np.abs(data[0:j_index])>0.03)==0)):
                if(j_index>0):
                    break
        if (j_index != np.size(data)-1):
            for k in np.arange(j_index, 0, -1):
                if((np.abs(data[k])<0.02)):
                #if((np.abs(data[k])<0.015) & (np.sum(np.abs(data[0:k])>0.02)==0)):
                    plt.axvline(x[k-1],color='red', label='Sparse')
                    break
        # 1-st diff 
        # plt.plot(x[0:-1],np.diff(data)/np.diff(x)/1e2)
        coeff = np.zeros_like(x)
        if (j_index == np.size(data)-1):
            k=1
        x1 = x[0:k-1]
        theta0 = np.zeros(10)
        error = np.zeros(np.size(np.arange(k+10 ,np.size(data)-10,1)))
        a = np.diff(data)
        b = np.diff(x)
        diff = a[1:]*a[:-1] /(b[1:]*b[:-1])
        min_diff = np.argsort(diff)

        r_set = []
        for h in np.arange(np.sum(diff<0)):
            k2 = min_diff[h]
            if(h>0):
                if(np.sum(np.abs(min_diff[0:h] - min_diff[h])<15)==0):            
                    #plt.axvline(x[k2],color='green')
                    if((min_diff[h]>k+5) & (min_diff[h] - np.size(min_diff)<-20)):
                        plt.axvline(x[k2],color='green', label='Segmented')
                        r_set.append(k2)
            else:
                if((min_diff[h]>k+5) & (min_diff[h] - np.size(min_diff)<-20)):
                    plt.axvline(x[k2],color='green', label='Segmented')
                    r_set.append(k2)
        r_set_index = np.argsort(x[r_set])
        output = []
        xx = []

        poly_r = x[k]
        poly_coeff = [];
        for l in np.arange(np.size(r_set)+1):
            if (l == 0):
                left = k
                right = r_set[r_set_index[l]]
            elif (l == np.size(r_set)):
                left = r_set[r_set_index[l-1]]
                right = np.size(data)-1
            else:
                left = r_set[r_set_index[l-1]]
                right = r_set[r_set_index[l]]
            #print(x[left],x[right])
            x2 = x[left:right]
            a2 = data[left:right]
            if not i%2:
                f = even_func_poly
                f1 = even_func_poly1
                #print(i, f.__name__)
            else:
                f = odd_func_poly
                f1 = odd_func_poly1
                #print(i, f.__name__)
            w2 = np.ones_like(x2)
            w2[0] = 100
            if(l < np.size(r_set)-1):
                w2[-1] = 100
            result = minimize(f, theta0, method='SLSQP', args = (x2, a2, w2))
            output = np.hstack((output,f1(result.x, *(x2, a2))))
            xx = np.hstack((xx,x2))
            if(l < np.size(r_set)):
                poly_r = np.hstack((poly_r, x[r_set[r_set_index[l]]]))
                poly_coeff = np.hstack((poly_coeff, result.x))
            else:
                poly_r = np.hstack((poly_r, 1))
                poly_coeff = np.hstack((poly_coeff, result.x))

        poly_coeff = np.reshape(poly_coeff, (-1,10))
        
        out.create_dataset('poly_r_%d' % i, data = poly_r)
        out.create_dataset('poly_coeff_%d' % i, data = poly_coeff)
        
        plt.plot(xx, output)
        plt.xlabel('Relative Radius')
        plt.ylabel('Coefficients')
        plt.title(str(i)+'-th Legendre coeff')
        plt.legend()
        plt.savefig('PE_fig%d.png' % i)