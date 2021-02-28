#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.55, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe

def main(order=5, fit_order=10):
    rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
    rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
    coeff_pe2[0] = coeff_pe2[0] + np.log(20000/4285)
    rd = np.hstack((rd1, rd2))
    coeff_pe = np.hstack((coeff_pe1, coeff_pe2))
    coeff_L = np.zeros((order, fit_order + 1))

    for i in np.arange(order):
        if not i%2:
            B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                       np.hstack((coeff_pe[i], coeff_pe[i])),                                                       deg = fit_order, full = True)
        else:
            B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                       np.hstack((coeff_pe[i], -coeff_pe[i])),                                                       deg = fit_order, full = True)

        y = np.polynomial.legendre.legval(rd/np.max(rd), B)

        coeff_L[i] = B

        plt.figure(num = i+1, dpi = 300)
        #plt.plot(rd, np.hstack((y2_in, y2_out[1:])), label='poly')
        plt.plot(rd, coeff_pe[i], 'r.', label='real',linewidth=2)
        plt.plot(rd, y, label = 'Legendre')
        plt.xlabel('radius/m')
        plt.ylabel('PE Legendre coefficients')
        plt.title('%d-th, max fit = %d' %(i, fit_order))
        plt.legend()

    return coeff_L

coeff_L = main(5,80)


# In[31]:


import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.55, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 5
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe1[0] -= np.log(20000/4285)


# In[33]:


from scipy.optimize import minimize
# optimize the piecewise function
# inner: 30, outer: 80
N_in = 30
N_out = 90
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1, coeff_pe2 = args
    y1 = np.polynomial.legendre.legval(rd1/0.65, theta[0:N_in])
    L1 = np.sum(coeff_pe1[1]-y1)**2
    y2 = np.polynomial.legendre.legval(rd2/0.65, theta[-N_out:])
    L2 = np.sum(coeff_pe2[1]-y2)**2
    return L1 + L2

result_in = minimize(loss, theta, method='SLSQP',args = (rd1, rd2, coeff_pe1, coeff_pe2))

plt.plot(rd1/0.65, np.polynomial.legendre.legval(rd1/0.65, result_in.x[0:N_in]))
plt.plot(rd2/0.65, np.polynomial.legendre.legval(rd2/0.65, result_in.x[-N_out:]))

plt.plot(rd1/0.65, coeff_pe1[1],'.')
plt.plot(rd2/0.65, coeff_pe2[1],'.')


# In[14]:


result_in


# In[92]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.56, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe1[0] -= np.log(20000/4285)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 30
N_out = 80
order = 25
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1, coeff_pe2 = args
    y1 = np.polynomial.legendre.legval(rd1/0.65, theta[0:N_in])
    L1 = np.sum((coeff_pe1[order]-y1)**2)
    y2 = np.polynomial.legendre.legval(rd2/0.65, theta[-N_out:])
    L2 = np.sum((coeff_pe2[order]-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((coeff_pe1[order]-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((coeff_pe2[order]-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((-coeff_pe1[order]-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((-coeff_pe2[order]-y2)**2)
    return L1 + L2

eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(np.max(rd1)/0.65, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]), 
                                      #np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]) - \
                                      #   np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]),# 0-th
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/((np.max(rd1)/0.65)**2-1)) - 
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/((np.max(rd1)/0.65)**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd1, rd2, coeff_pe1, coeff_pe2))

plt.figure(dpi=300)

rd = np.arange(0.53,0.56,0.001)
plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[0:N_in]), label='inner fit')
plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[-N_out:]), label='outer fit')
plt.axvline(np.max(rd1)/0.65, color='red', label='Discontinuity')

plt.xlabel('Relative R')
plt.ylabel('Legendre Coefficient')
plt.legend()


# In[ ]:





# In[85]:


print((np.polynomial.legendre.legval(0.55/0.65, result.x[0:N_in]) - np.polynomial.legendre.legval(0.54999/0.65, result.x[0:N_in]))/0.00001*0.65)
#print(np.arange(N_in-1)*(np.max(rd1)/0.65*np.polynomial.legendre.legval(np.max(rd1)/0.65, result.x[1:N_in]) - \
#                 np.polynomial.legendre.legval(np.max(rd1)/0.65, result.x[0:N_in-1])))
x = np.max(rd1)/0.65
coeff = result.x.copy()
#coeff[0] = 0
print()
print(np.polynomial.legendre.legval(0.55/0.65, result.x[-N_out:]) - np.polynomial.legendre.legval(0.551/0.65, result.x[-N_out:]))


# In[86]:





# In[74]:


size = 5
coeff = np.ones(5)
x = 0.5
print(np.sum(
        (x * np.polynomial.legendre.legval(x, np.diag(coeff[0:size]))[1:size] - \
             np.polynomial.legendre.legval(x, np.diag(coeff[1:size]))) * \
                np.arange(1,size)/(x**2-1)))


# In[76]:


np.polynomial.legendre.legval(x, np.diag(coeff[0:size]))[1:size]


# In[77]:


np.polynomial.legendre.legval(x, np.diag(coeff[1:size]))


# In[81]:


(x * np.polynomial.legendre.legval(x, np.diag(coeff[0:size]))[1:size] -              np.polynomial.legendre.legval(x, np.diag(coeff[1:size]))) *                 np.arange(1,size)/(x**2-1)


# In[82]:


size = 5
coeff = np.arange(size)
x = 0.5
print(np.sum(
        (x * np.polynomial.legendre.legval(x, np.diag(coeff[0:size]))[1:size] - \
             np.polynomial.legendre.legval(x, np.diag(coeff[1:size]))) * \
                np.arange(1,size)/(x**2-1)))

(x * np.polynomial.legendre.legval(x, np.diag(coeff[0:size]))[1:size] -              np.polynomial.legendre.legval(x, np.diag(coeff[1:size]))) *                 np.arange(1,size)/(x**2-1)


# In[193]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.56, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe1[0] -= np.log(20000/4285)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 30
N_out = 80

theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1_row, coeff_pe2_row, order = args
    y1 = np.polynomial.legendre.legval(rd1/0.65, theta[0:N_in])
    L1 = np.sum((coeff_pe1_row-y1)**2)
    y2 = np.polynomial.legendre.legval(rd2/0.65, theta[-N_out:])
    L2 = np.sum((coeff_pe2_row-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((coeff_pe1_row-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((coeff_pe2_row-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rd1/0.65, theta[0:N_in])
        L1 += np.sum((-coeff_pe1_row-y1)**2)
        y2 = np.polynomial.legendre.legval(-rd2/0.65, theta[-N_out:])
        L2 += np.sum((-coeff_pe2_row-y2)**2)
    return L1 + L2

eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(np.max(rd1)/0.65, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]), 
                                      #np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]) - \
                                      #   np.polynomial.legendre.legval(np.max(rd1)/0.65, x[-N_out:]),# 0-th
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/((np.max(rd1)/0.65)**2-1)) - 
                                      np.sum((np.max(rd1)/0.65 * np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(np.max(rd1)/0.65, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/((np.max(rd1)/0.65)**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

for i in np.arange(29):    
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd1, rd2, coeff_pe1[i], coeff_pe2[i], i))
    plt.figure(dpi=300)
    rd = np.arange(0.01,0.65,0.001)
    plt.plot(rd1/0.65, coeff_pe1[i], 'r.', alpha = 0.5, label='inner data')
    plt.plot(rd2/0.65, coeff_pe2[i], 'r.', alpha = 0.5, label='outer data')
    plt.plot(rd1/0.65, np.polynomial.legendre.legval(rd1/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    plt.plot(rd2/0.65, np.polynomial.legendre.legval(rd2/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)
    plt.savefig('Fit%d.pdf' % i)
    plt.figure(dpi=300)
    rd = np.arange(0.54,0.56,0.001)
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[0:N_in]), label='inner fit')
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[-N_out:]), label='outer fit')
    plt.axvline(np.max(rd1)/0.65, color='red', label='Discontinuity')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit local' % i)
    plt.savefig('Fit%d_local.pdf' % i)


# In[101]:


coeff_pe1[i].shape


# In[103]:


plt.plot(coeff_pe1[i],'.')


# In[51]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.56, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.645, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe1[0] -= np.log(20000/4285)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1_row, coeff_pe2_row, order, bnd = args
    
    rd = np.hstack((rd1, rd2))
    coeff_pe_row = np.hstack((coeff_pe1_row, coeff_pe2_row))

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]

    y1 = np.polynomial.legendre.legval(rda/0.65, theta[0:N_in])
    L1 = np.sum((ya-y1)**2)
    y2 = np.polynomial.legendre.legval(rdb/0.65, theta[-N_out:])
    L2 = np.sum((yb-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((yb-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((-ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((-yb-y2)**2)
    return L1 + L2

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

for i in np.arange(3):
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd1, rd2, coeff_pe1[i], coeff_pe2[i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]
    plt.figure(dpi=300)
    rd = np.arange(0.00,0.65,0.001)
    
    rd = np.hstack((rd1, rd2))
    coeff_pe_row = np.hstack((coeff_pe1[i], coeff_pe2[i]))

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]
    plt.plot(rda/0.65, ya, 'r.', alpha = 0.5, label='inner data')
    plt.plot(rdb/0.65, yb, 'r.', alpha = 0.5, label='outer data')
    plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    plt.plot(rdb/0.65, np.polynomial.legendre.legval(rdb/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)
    plt.savefig('Fit%d.pdf' % i)
    plt.figure(dpi=300)
    rd = np.arange(bnd*0.65-0.02,bnd*0.65+0.02,0.001)
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[0:N_in]), label='inner fit')
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[-N_out:]), label='outer fit')
    plt.axvline(bnd, color='red', label='Discontinuity')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit local' % i)
    plt.savefig('Fit%d_local.pdf' % i)


# In[40]:


coeff_in


# In[26]:


coeff_out.shape


# In[55]:


y1 = np.polynomial.legendre.legval(-rda/0.65, coeff_in[0,0:N_in])
plt.plot(-rda/0.65, y1)
y1 = np.polynomial.legendre.legval(rda/0.65, coeff_in[0,0:N_in])
plt.plot(rda/0.65, y1)

y1 = np.polynomial.legendre.legval(-rda/0.65, coeff_in[1,0:N_in])
plt.plot(-rda/0.65, y1)
y1 = np.polynomial.legendre.legval(rda/0.65, coeff_in[1,0:N_in])
plt.plot(rda/0.65, y1)


# In[29]:


with h5py.File('./PE_coeff_1t_new.h5','w') as out:
    out.create_dataset('coeff_in', data = coeff_in)
    out.create_dataset('coeff_out', data = coeff_out)
    out.create_dataset('bd', data = bnd*0.65)
h = tables.open_file('../calib/PE_coeff_1t_new.h5','r')
h.root.coeff_in[:]


# In[ ]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.56, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.645, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe1[0] -= np.log(20000/4285)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd1, rd2, coeff_pe1_row, coeff_pe2_row, order, bnd = args
    
    rd = np.hstack((rd1, rd2))
    coeff_pe_row = np.hstack((coeff_pe1_row, coeff_pe2_row))

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]

    y1 = np.polynomial.legendre.legval(rda/0.65, theta[0:N_in])
    L1 = np.sum((ya-y1)**2)
    y2 = np.polynomial.legendre.legval(rdb/0.65, theta[-N_out:])
    L2 = np.sum((yb-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((yb-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((-ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((-yb-y2)**2)
    return L1 + L2

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

for i in np.arange(3):
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd1, rd2, coeff_pe1[i], coeff_pe2[i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]
    plt.figure(dpi=300)
    rd = np.arange(0.00,0.65,0.001)
    
    rd = np.hstack((rd1, rd2))
    coeff_pe_row = np.hstack((coeff_pe1[i], coeff_pe2[i]))

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]
    plt.plot(rda/0.65, ya, 'r.', alpha = 0.5, label='inner data')
    plt.plot(rdb/0.65, yb, 'r.', alpha = 0.5, label='outer data')
    plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    plt.plot(rdb/0.65, np.polynomial.legendre.legval(rdb/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)
    plt.savefig('Fit%d.pdf' % i)
    plt.figure(dpi=300)
    rd = np.arange(bnd*0.65-0.02,bnd*0.65+0.02,0.001)
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[0:N_in]), label='inner fit')
    plt.plot(rd/0.65, np.polynomial.legendre.legval(rd/0.65, result.x[-N_out:]), label='outer fit')
    plt.axvline(bnd, color='red', label='Discontinuity')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit local' % i)
    plt.savefig('Fit%d_local.pdf' % i)


# In[56]:


np.arange(5)%2


# In[86]:


import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.02, 0.65, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe


rd, coeff_pe = main_photon_sparse('coeff_pe_1t_point_10_track_30/',order)


# In[87]:


for i in np.arange(coeff_pe[0].shape[0]):
    plt.figure(dpi=200)
    plt.plot(rd, coeff_pe[i])
    plt.show()


# In[83]:


r = np.arange(0.30,0.31,0.01)
b = []
for rd in r:
    h = tables.open_file('./coeff_pe_1t_point_10_track_30/file_+%.3f.h5' % rd)
    a = []
    for order in np.arange(5,30):
        a.append(eval('h.root.AIC%d[()]' % order))
    b.append(np.arange(5,30)[np.where(np.array(a)==np.min(a))[0]])
    h.close()
plt.plot(r, b)


# In[84]:


a


# In[85]:


r = 0.25
h = tables.open_file('./coeff_pe_1t_point_10_track_30/file_+%.3f.h5' % rd)
a = []
for order in np.arange(5,30):
    a.append(eval('h.root.AIC%d[()]' % order))
a = np.array(a)
plt.plot(np.arange(5,30), a)
plt.axvline(np.arange(5,30)[np.where(a==np.min(a))[0]])
plt.show()


# In[77]:


(a==np.min(a))


# In[111]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.02, 0.63, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff))
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F').T
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd, coeff_pe = main_photon_sparse('coeff_pe_1t_point_10_track_30/',order)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd, coeff_pe_row, order, bnd = args

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]

    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]

    y1 = np.polynomial.legendre.legval(rda/0.65, theta[0:N_in])
    L1 = np.sum((ya-y1)**2)
    y2 = np.polynomial.legendre.legval(rdb/0.65, theta[-N_out:])
    L2 = np.sum((yb-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((yb-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((-ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((-yb-y2)**2)
    return L1 + L2

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

for i in np.arange(30):
    print(rd.shape, coeff_pe.shape)
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd, coeff_pe[:,i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]
    plt.figure(dpi=300)
    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe[rd<bnd*0.65, i] 
    yb = coeff_pe[rd>=bnd*0.65, i]
    plt.plot(rda/0.65, ya, 'r.', alpha = 0.5, label='inner data')
    plt.plot(rdb/0.65, yb, 'r.', alpha = 0.5, label='outer data')
    plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    plt.plot(rdb/0.65, np.polynomial.legendre.legval(rdb/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)
    plt.savefig('Fit%d.pdf' % i)
    plt.figure(dpi=300)
    
    _rd = np.arange(bnd*0.65-0.02,bnd*0.65+0.02,0.001)
    plt.plot(_rd/0.65, np.polynomial.legendre.legval(_rd/0.65, result.x[0:N_in]), label='inner fit')
    plt.plot(_rd/0.65, np.polynomial.legendre.legval(_rd/0.65, result.x[-N_out:]), label='outer fit')
    plt.axvline(bnd, color='red', label='Discontinuity')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit local' % i)
    plt.savefig('Fit%d_local.pdf' % i)


# In[104]:


coeff_pe[i].shape


# In[101]:


rd.shape


# In[184]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.02, 0.63, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff))
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F').T
    return np.array(rd), np.array(coeff_pe)

# load coeff
order = 29
rd, coeff_pe = main_photon_sparse('coeff_pe_1t_point_10_track_30/',order)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

def loss(theta, *args):
    rd, coeff_pe_row, order, bnd = args

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]

    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]

    y1 = np.polynomial.legendre.legval(rda/0.65, theta[0:N_in])
    L1 = np.sum((ya-y1)**2)
    y2 = np.polynomial.legendre.legval(rdb/0.65, theta[-N_out:])
    L2 = np.sum((yb-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((yb-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((-ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((-yb-y2)**2)
    return L1 + L2

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):

plt.subplots(2,3,dpi=300, figsize=(20,10), sharex=True)
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.suptitle('Legendre Coefficient', fontsize=35)
for i in np.arange(5):
    print(rd.shape, coeff_pe.shape)

    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd, coeff_pe[:,i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]
    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe[rd<bnd*0.65, i] 
    yb = coeff_pe[rd>=bnd*0.65, i]
    plt.subplot(2,3,i+1)
    plt.plot(rda/0.65, ya, 'r.', alpha = 0.5, label='Coefficient')
    plt.plot(rdb/0.65, yb, 'r.', alpha = 0.5)
    #plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]), 'b-', label='inner fit')
    #plt.plot(rdb/0.65, np.polynomial.legendre.legval(rdb/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R',fontsize=25)
    # plt.ylabel('Legendre Coefficient', fontsize=25)
    # plt.legend(fontsize=25)
    plt.title('%d-th order' % i, fontsize=25)
plt.subplot(2,3,6)
plt.axis('off')
plt.savefig('coeff.png')


# In[180]:


import tables
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sys
import h5py
import seaborn as sns
def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.01, 0.55, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe

def main_photon_compact(path, order):
    ra = np.arange(0.55, 0.65, 0.002)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff)) 
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F')
    return rd, coeff_pe

order=5
fit_order=80
rd1, coeff_pe1 = main_photon_sparse('coeff_pe_1t_reflection0.00_30/',order)
rd2, coeff_pe2 = main_photon_compact('coeff_pe_1t_compact_30/',order)
coeff_pe2[0] = coeff_pe2[0] + np.log(20000/4285)
rd = np.hstack((rd1, rd2))
coeff_pe = np.hstack((coeff_pe1, coeff_pe2))
coeff_L = np.zeros((order, fit_order + 1))
'''
plt.subplots(2,3,dpi=300, figsize=(20,10), sharex=True)
plt.subplots_adjust(hspace=0.4, wspace=0.2)
plt.suptitle('Legendre Coefficient', fontsize=35)
'''
#spec = mpl.gridspec.GridSpec(ncols=3, nrows=2)
plt.rcParams['legend.title_fontsize'] = '15'
fig = plt.figure(dpi=300)
spec = mpl.gridspec.GridSpec(ncols=3, nrows=2, wspace=0.25,hspace=0.5)
ax1 = fig.add_subplot(spec[0,0])
ax2 = fig.add_subplot(spec[0,1])
ax3 = fig.add_subplot(spec[0,2])
ax4 = fig.add_subplot(spec[1,0])
ax5 = fig.add_subplot(spec[1,1])
for i in np.arange(5):
    if not i%2:
        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                   np.hstack((coeff_pe[i], coeff_pe[i])),                                                   deg = fit_order, full = True)
    else:
        B, tmp = np.polynomial.legendre.legfit(np.hstack((rd/np.max(rd),-rd/np.max(rd))),                                                   np.hstack((coeff_pe[i], -coeff_pe[i])),                                                   deg = fit_order, full = True)

    y = np.polynomial.legendre.legval(rd/np.max(rd), B)

    coeff_L[i] = B

    ax = eval('ax%d' % (i+1))
    ax.plot(rd/0.65, coeff_pe[i], 'r.', label='Raw data',linewidth=1, markersize=0.8)
    ax.set_xlabel('Relative R',fontsize=8)
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    for tick in ax.yaxis.get_major_ticks():
        tick.label.set_fontsize(6)
    ax.legend(fontsize=7)
    ax.set_title('%d-th order fit' % i, fontsize=8)
    
plt.savefig('coeff.png')
'''
plt.plot(rd, y, label = 'Fit')
plt.xlabel('Relative R',fontsize=25)
# plt.ylabel('Legendre Coefficient', fontsize=25)
leg = plt.legend()
#plt.setp(ax.get_legend().get_texts(), fontsize='22') 
leg.get_texts()[0].set_fontsize('25')
plt.title('%d-th order fit' % i, fontsize=25)

if(i==5):
    plt.delaxes(plt.gca)
'''


# In[141]:





# In[1]:


from scipy.optimize import minimize

import tables
import matplotlib.pyplot as plt
import numpy as np
import sys
import h5py

def LoadDataPE_TW(path, radius, order):
    data = []
    filename = path + 'file_' + radius + '.h5'
    h = tables.open_file(filename,'r')
    coeff = 'coeff' + str(order)
    hess = 'hess' + str(order)
    data = eval('np.array(h.root.'+ coeff + '[:])')
    h.close()
    return data

def main_photon_sparse(path, order):
    ra = np.arange(0.02, 0.63, 0.01)
    rd = []
    coeff_pe = []
    for radius in ra:
        str_radius = '%+.3f' % radius
        coeff= LoadDataPE_TW(path, str_radius, order)
        rd.append(np.array(radius))
        coeff_pe = np.hstack((coeff_pe, coeff))
    coeff_pe = np.reshape(coeff_pe,(-1,np.size(rd)),order='F').T
    return np.array(rd), np.array(coeff_pe)



def loss(theta, *args):
    rd, coeff_pe_row, order, bnd = args

    rda = rd[rd<bnd*0.65]
    rdb = rd[rd>=bnd*0.65]

    ya = coeff_pe_row[rd<bnd*0.65] 
    yb = coeff_pe_row[rd>=bnd*0.65]

    y1 = np.polynomial.legendre.legval(rda/0.65, theta[0:N_in])
    L1 = np.sum((ya-y1)**2)
    y2 = np.polynomial.legendre.legval(rdb/0.65, theta[-N_out:])
    L2 = np.sum((yb-y2)**2)
    
    if not order%2:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((yb-y2)**2)
    else:
        y1 = np.polynomial.legendre.legval(-rda/0.65, theta[0:N_in])
        L1 += np.sum((-ya-y1)**2)
        y2 = np.polynomial.legendre.legval(-rdb/0.65, theta[-N_out:])
        L2 += np.sum((-yb-y2)**2)
    return L1 + L2

# load coeff
order = 29
rd, coeff_pe = main_photon_sparse('coeff_pe_1t_point_10_track_30/',order)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):
plt.figure(dpi=300)
for i in np.arange(8,9):
    print(rd.shape, coeff_pe.shape)
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd, coeff_pe[:,i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]

    rda = rd[rd<bnd*0.65]
    #rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe[rd<bnd*0.65, i] 
    #yb = coeff_pe[rd>=bnd*0.65, i]
    plt.plot(rda/0.65, ya, 'r.', alpha = 0.5, label='without acrylic reflection')
    # plt.plot(rdb/0.65, yb, 'r.', alpha = 0.5)
    plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]), linewidth=0.5, 'r-', label='fit, without acrylic reflection')
    plt.xlabel('Relative R')
    plt.ylabel('Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order fit' % i)

# load coeff
order = 29
rd, coeff_pe = main_photon_sparse('coeff_pe_1t_point_30/',order)

# optimize the piecewise function
# inner: 30, outer: 80
N_in = 20
N_out = 40
coeff_in = np.zeros((order, N_in))
coeff_out = np.zeros((order, N_out))
theta = np.zeros(N_in + N_out)

bnd = 0.572/0.65
#bnd = np.max(rd1)/0.65
eq_cons = {'type':'eq',
           'fun': lambda x: np.array([np.polynomial.legendre.legval(bnd, x[0:N_in]) - \
                                         np.polynomial.legendre.legval(bnd, x[-N_out:]), 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[0:N_in]))[1:N_in] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[1:N_in]))) * \
                                         np.arange(1,N_in)/(bnd**2-1)) - 
                                      np.sum((bnd * np.polynomial.legendre.legval(bnd, np.diag(x[-N_out:]))[1:N_out] - \
                                         np.polynomial.legendre.legval(bnd, np.diag(x[-N_out+1:]))) * \
                                         np.arange(1,N_out)/(bnd**2-1))
                                     ]), # 1-st
            }

#def cons(theta, *args):
for i in np.arange(8,9):
    print(rd.shape, coeff_pe.shape)
    result = minimize(loss, theta, method='SLSQP', constraints=eq_cons, args = (rd, coeff_pe[:,i], i, bnd))
    coeff_in[i] = result.x[:N_in]
    coeff_out[i] = result.x[-N_out:]

    rda = rd[rd<bnd*0.65]
    #rdb = rd[rd>=bnd*0.65]
    ya = coeff_pe[rd<bnd*0.65, i] 
    #yb = coeff_pe[rd>=bnd*0.65, i]
    plt.plot(rda/0.65, ya, 'b.', alpha = 0.5, label='with acrylic reflection')
    #plt.plot(rdb/0.65, yb, 'b.', alpha = 0.5)
    plt.plot(rda/0.65, np.polynomial.legendre.legval(rda/0.65, result.x[0:N_in]),linewidth=0.5, 'b-', label='fit, with acrylic reflection')
    #plt.plot(rdb/0.65, np.polynomial.legendre.legval(rdb/0.65, result.x[-N_out:]), 'g-', label='outer fit')
    plt.xlabel('Relative R')
    plt.ylabel('PE Legendre Coefficient')
    plt.legend()
    plt.title('%d-th order' % i)


plt.savefig('Acrylic_fit.png')


# In[212]:


h = tables.open_file('coeff_pe_1t_point_10_track_30/file_+0.020.h5')
print(h.root)


# In[213]:


h = tables.open_file('coeff_pe_1t_point_10_track_30/file_+0.020.h5')


# In[215]:


h.root.std10[:]


# In[ ]:




