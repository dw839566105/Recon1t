import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot, argparse
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from scipy.linalg import norm
from scipy.stats import norm as normpdf
from scipy.spatial import distance
import warnings

class ReconData(tables.IsDescription):
    EventID = tables.Int64Col(pos=0)    # EventNo
    # inner recon
    E_sph_in = tables.Float16Col(pos=1)        # Energy
    x_sph_in = tables.Float16Col(pos=2)        # x position
    y_sph_in = tables.Float16Col(pos=3)        # y position
    z_sph_in = tables.Float16Col(pos=4)        # z position
    t0_in = tables.Float16Col(pos=5)          # time offset
    success_in = tables.Int64Col(pos=6)        # recon status   
    Likelihood_in = tables.Float16Col(pos=7)

    # outer recon
    E_sph_out = tables.Float16Col(pos=8)         # Energy
    x_sph_out = tables.Float16Col(pos=9)         # x position
    y_sph_out = tables.Float16Col(pos=10)        # y position
    z_sph_out = tables.Float16Col(pos=11)        # z position
    t0_out = tables.Float16Col(pos=12)          # time offset
    success_out = tables.Int64Col(pos=13)        # recon status 
    Likelihood_out = tables.Float16Col(pos=14)

    # truth info
    x_truth = tables.Float16Col(pos=15)        # x position
    y_truth = tables.Float16Col(pos=16)        # y position
    z_truth = tables.Float16Col(pos=17)        # z position
    E_truth = tables.Float16Col(pos=18)        # z position
        
def readtpl(filename='../MC/template.h5'):
    # Read MC grid recon result
    h = tables.open_file(filename)
    tp = h.root.template[:]
    bins = np.vstack((h.root.x[:], h.root.y[:], h.root.z[:])).T
    h.close()
    return tp, bins

def load_coeff(PEFile = '../calib/PE_coeff_1t_29_80.h5', TimeFile = '../calib/Time_coeff2_1t_0.1.h5'):
    # spherical harmonics coefficients for time and PEmake 
    h = tables.open_file(PEFile, 'r')
    coeff_pe = h.root.coeff_L[:]
    h.close()
    cut_pe, fitcut_pe = coeff_pe.shape

    h = tables.open_file(Timefile,'r')
    coeff_time = h.root.coeff_L[:]
    h.close()
    cut_time, fitcut_time = coeff_time.shape
    return coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time

def r2c(c):
    v = np.zeros(3)
    v[2] = c[0] * np.cos(c[1]) #z
    rho = c[0] * np.sin(c[1])
    v[0] = rho * np.cos(c[2]) #x
    v[1] = rho * np.sin(c[2]) #y
    return v

def c2r(c):
    v = np.zeros(3)
    v[0] = norm(c)
    v[1] = np.arccos(c[2]/(v[0]+1e-6))
    #v[2] = np.arctan(c[1]/(c[0]+1e-6)) + (c[0]<0)*np.pi
    v[2] = np.arctan2(c[1],c[0])
    return v

class Likelihood_1:
    print('Using method: do not fit energy, energy is estimated by normalized')
    def Likelihood(vertex, *args):
        '''
        vertex[1]: r
        vertex[2]: theta
        vertex[3]: phi
        '''
        coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_tpl, PE = args
        z, x = Calc_basis(vertex, PMT_pos, np.max((cut_time, cut_pe)))
        L1, E = Likelihood_PE(z, x, coeff_pe, pe_array, cut_pe, N, pdf_tpl)
        L2 = Likelihood_Time(z, x, vertex[4], coeff_time, fired_PMT, time_array, cut_time, PE)
        return L1 + L2

    def Calc_basis(vertex, PMT_pos, cut):
        # boundary
        v = r2c(vertex[1:4])
        z = norm(v)
        if z > 1-1e-3:
            z = 1-1e-3
        # calculate cos theta
        cos_theta = np.dot(v, PMT_pos.T) / (norm(v)*norm(PMT_pos,axis=1))
        ### Notice: Here may not continuous! ###
        cos_theta[np.isnan(cos_theta)] = 1 # for v in detector center    

        # Generate Legendre basis
        x = LG.legval(cos_theta, np.diag((np.ones(cut)))).T   
        return z, x
    
    def Likelihood_PE(z, x, coeff, pe_array, cut, N, pdf_tpl):
        # Recover coefficient
        k = LG.legval(z, coeff_pe.T)
        # Recover expect
        expect = np.exp(np.dot(x,k))
        # Energy fit 
        nml = np.sum(expect)/np.sum(pe_array)
        expect = expect/nml
        k[0] = k[0] - np.log(nml) # 0-th

        # Poisson likelihood
        # p(q|lambda) = sum_n p(q|n)p(n|lambda)
        #         = sum_n Gaussian(q, n, sigma_n) * exp(-expect) * expect^n / n!
        # int p(q|lambda) dq = sum_n exp(-expect) * expect^n / n! = 1
        a0 = np.atleast_2d(expect).T ** N / (scipy.special.factorial(N))
        a1 = np.nansum(a0 * pdf_tpl, axis=1)
        a2 = np.exp(-expect)

        # -ln Likelihood
        L = - np.sum(np.sum(np.log(a1*a2)))
        # avoid inf (very impossible vertex) 
        if(np.isinf(L) or L>1e20):
            L = 1e20
        return L, k[0]
    
    def Likelihood_Time(z, x, T0, coeff, fired_PMT, time_array, cut, PE):
        x = x[fired_PMT][:,:cut]

        # Recover coefficient
        k = np.atleast_2d(LG.legval(z, coeff_time.T)).T
        k[0,0] = T0

        # Recover expect
        T_i = np.dot(x, k)

        # Likelihood
        L = - np.nansum(Likelihood_quantile(time_array, T_i[:,0], 0.1, 2.6, PE))
        return L

    def Likelihood_quantile(y, T_i, tau, ts, PE):
        # less = T_i[y<T_i] - y[y<T_i]
        # more = y[y>=T_i] - T_i[y>=T_i]    
        # R = (1-tau)*np.sum(less) + tau*np.sum(more)

        # since lucy ddm is not sparse, use PE as weight
        L = (T_i-y) * (y<T_i) * (1-tau) + (y-T_i) * (y>=T_i) * tau
        nml = tau*(1-tau)/ts**PE
        L_norm = np.exp(-np.atleast_2d(L).T * PE) * nml / ts
        L = np.log(np.sum(L_norm, axis=1))
        return L_norm

class initial:
    def ChargeWeighted():
        pass
    def MCGrid():
        pass
    def FitGrid():
        pass
    

    
        
