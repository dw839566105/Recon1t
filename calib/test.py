from scipy.optimize import minimize
import tables
import numpy as np
import matplotlib.pyplot as plt
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
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return np.sum(w*res**2)

def even_func_poly(theta, *args):
    x, y, w = args
    size = np.size(theta)
    pw = np.arange(0,2*size,2)
    s = np.zeros_like(y)
    for i in np.arange(np.size(theta)):
        s = s + theta[i]*(x**pw[i])
    res = y - s
    return np.sum(w*res**2)

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
    pw = np.arange(0,2*size+1,2)
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

for i in np.arange(np.size(coeff_pe[:,0])):
    print('Processing %d-th order' % i)
    plt.figure(num=i+1)
    data = np.nan_to_num(coeff_pe[i,:])
    x = ra/0.65
    plt.plot(x, data,'.')
    for j_index, j in enumerate(data):
        if((np.abs(data[j_index])>0.03) & (np.sum(np.abs(data[0:j_index])>0.03)==0)):
            if(j_index>0):
                break
    if (j_index != np.size(data)-1):
        for k in np.arange(j_index, 0, -1):
            if((np.abs(data[k])<0.02)):
            #if((np.abs(data[k])<0.015) & (np.sum(np.abs(data[0:k])>0.02)==0)):
                plt.axvline(x[k-1],color='red')
                break
    # 1-st diff 
    # plt.plot(x[0:-1],np.diff(data)/np.diff(x)/1e2)
    coeff = np.zeros_like(x)
    if (j_index == np.size(data)-1):
        k=1
    x1 = x[0:k-1]
    theta0 = np.zeros(10) + 1e-1
    error = np.zeros(np.size(np.arange(k+10 ,np.size(data)-10,1)))
    for k2_index, k2 in enumerate(np.arange(k+10,np.size(data)-10,1)):
        x2 = x[k-1:k2+1]
        a2 = data[k-1:k2+1]
        
        x3 = x[k2::]
        a3 = data[k2::]

        if not i%2:
            #f = even_func_poly
            #f1 = even_func_poly1
            f = even_func_Legendre
            f1 = even_func_Legendre1
            #print(i, f.__name__)
        else:
            f = odd_func_Legendre
            f1 = odd_func_Legendre1
            #print(i, f.__name__)
        w2 = np.ones_like(x2)
        w3 = np.ones_like(x3)
        w2[0] = 100
        w2[-1] = 100
        w3[0] = 100
        result2 = minimize(f, theta0, method='SLSQP', args = (x2, a2, w2))
        result3 = minimize(f, theta0, method='SLSQP', args = (x3, a3, w3))

        output2 = f1(result2.x, *(x2, a2))
        output3 = f1(result3.x, *(x3, a3))
        error[k2_index] = np.sum((output2 - a2)**2) + np.sum((output3 - a3)**2)
        #error[k2_index] = np.sum((output2 - a2)**2/((x2-0.8)**2+0.0001)) +  np.sum((output3 - a3)**2/((x3-0.8)**2+0.0001))
        
    min_index = np.where(error == np.min(error))
    k2 = min_index[0][0]
    x2 = x[k-1:k2+1]
    a2 = data[k-1:k2+1]
    
    x3 = x[k2::]
    a3 = data[k2::]
    w2 = np.ones_like(x2)
    w3 = np.ones_like(x3)
    w2[0] = 100
    w2[-1] = 100
    w3[0] = 100
    result2 = minimize(f, theta0, method='SLSQP', args = (x2, a2, w2))
    result3 = minimize(f, theta0, method='SLSQP', args = (x3, a3, w3))
    output2 = f1(result2.x, *(x2, a2))
    output3 = f1(result3.x, *(x3, a3))

    plt.plot(x2, output2)
    plt.plot(x3, output3)
    plt.xlabel('Relative Radius')
    plt.ylabel('Coefficients')
    plt.title(str(i)+'-th Legendre coeff')
    plt.legend(['raw'])
    plt.savefig('fig%d.png' % i)