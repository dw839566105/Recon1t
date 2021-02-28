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


# In[19]:


coeff_pe1[0].shape


# In[ ]:




