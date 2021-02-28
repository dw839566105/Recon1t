import sys
import pub
import numpy as np

import tables, h5py
import argparse
import time


'''
from scipy.optimize import minimize
from sklearn.linear_model import Lasso
from sklearn.linear_model import TweedieRegressor
from numpy.polynomial import legendre as LG
import matplotlib.pyplot as plt
import statsmodels.api as sm
import scipy
from zernike import RZern
import h2o
'''

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = pub.ReadJPPMT()



def main_Calib(filename, output, alg, basis, order, figure, verbose):
    '''
    # main program
    # input: radius: %+.3f, 'str' (in makefile, str is default)
    #        path: file storage path, 'str'
    #        fout: file output name as .h5, 'str' (.h5 not included')
    #        cut_max: cut off of Legendre
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    print('begin reading file', flush=True)
    with h5py.File(output,'w') as out:
        EventID, Q, x, y, z = pub.ReadFile(filename)
        VertexTruth = (np.vstack((x, y, z))/1e3).T

        print('total event: %d' % np.size(np.unique(EventID)), flush=True)
        print('begin processing legendre coeff', flush=True)
        # this part for the same vertex
        
        tmp = time.time()
        EventNo = np.size(np.unique(EventID))
        PMTNo = np.size(PMTPos[:,0])
        PMTPosRep = np.tile(PMTPos, (EventNo,1))
        vertex = np.repeat(VertexTruth, PMTNo, axis=0) # unit to meter

        if basis == 'Legendre':
            basis, cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=True)
        elif basis == 'Zernike':
            from zernike import RZern
            cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=False)

            cart = RZern(order)
            nk = cart.nk
            
            rho = np.linalg.norm(vertex, axis=1)/0.6
            theta = np.arccos(cos_theta)
            
            rho_total = np.hstack((rho, rho))
            theta_total = np.hstack((theta, np.pi*2 - theta))
            basis = np.zeros((rho_total.shape[0], nk))

            for i in np.arange(nk):
                if not i % 5:
                    print(f'process {i}-th event')
                basis[:,i] = cart.Zk(i, rho_total, theta_total)
                
        print(f'use {time.time() - tmp} s')
        if verbose:
            print(f'the basis shape is {basis.shape}, and the dependent variable shape is {Q.shape}') 
        
        # Regression methods:
        if (alg == 'sms'):
            import statsmodels.api as sm
            X = basis
            y = Q
            model = sm.GLM(y, X, family=sm.families.Poisson())
            result = model.fit()
            
            if verbose:
                print(result.summary())
            AIC = result.aic
            coef_ = result.params
            std = result.bse
            
        elif (alg == 'custom'):
            from scipy.optimize import minimize
            X = basis
            y = Q
            x0 = np.zeros(order) # initial value
            x0[0] = 0.8 + np.log(2) # intercept is much more important
            result = minimize(pub.Calib, x0=x0, method='SLSQP', args = (Q, PMTPos, X)) 
            coef_ = np.array(result.x, dtype=float)
            if verbose:
                print(result.message)
            AIC = np.zeros_like(coef_)
            std = np.zeros_like(coef_)

            H1 = pub.MyHessian(result.x, *(Q, PMTPos, X))
            std = 1/np.sqrt(-np.diag(np.linalg.pinv(H1)))
            print(coef_)
            print(std)
            print('Waring! No AIC and std value, std is testing')
        
        elif (alg == 'sk'):            
            from sklearn.linear_model import TweedieRegressor
            X = basis
            y = Q
            alpha = 0.001
            reg = TweedieRegressor(power=1, alpha=alpha, link='log', max_iter=1000, tol=1e-6, fit_intercept=False)
            reg.fit(X, y)
            
            # just for point data
            # pred = reg.predict(X[0:30,0:cut+1])

            print('coeff:\n', reg.coef_,'\n')
            
            coef_ = reg.coef_ 
            
            AIC = np.zeros_like(coef_)
            std = np.zeros_like(coef_)
            print('Waring! No AIC and std value')
        
        elif (alg == 'h2o'):
            import h2o
            from h2o.estimators.gbm import H2OGradientBoostingEstimator
            from h2o.estimators.glm import H2OGeneralizedLinearEstimator           
                        
            # symmetry
            Q = np.atleast_2d(np.hstack((Q,Q))).T
            data = np.hstack((basis, Q, np.ones_like(Q)))
            
            h2o.init() 
            hf = h2o.H2OFrame(data)
            predictors = hf.columns[0:-2]
            response_col = hf.columns[-2]
            
            #offset_col = hf.columns[-1]
            glm_model = H2OGeneralizedLinearEstimator(family= "poisson",
                #offset_column = offset_col,
                lambda_ = 0,
                compute_p_values = True)
            glm_model.train(predictors, response_col, training_frame=hf)
            coef_table = glm_model._model_json['output']['coefficients_table']
            
            if verbose:
                print(glm_model.coef())
                print(f'Regession coef shape is f{np.array(coef_).shape}, Zernike shape is {nk}')
            coef_ = coef_table['coefficients']
            std = coef_table['std_error']
            AIC = glm_model.aic()

            h2o.cluster().shutdown()
            
            if (figure=='ON'):
                import matplotlib.pyplot as plt
                L, K = 500, 500
                ddx = np.linspace(-1.0, 1.0, K)
                ddy = np.linspace(-1.0, 1.0, L)
                xv, yv = np.meshgrid(ddx, ddy)
                cart.make_cart_grid(xv, yv)
                # normal scale
                # im = plt.imshow(np.exp(cart.eval_grid(np.array(coef_), matrix=True)), origin='lower', extent=(-1, 1, -1, 1))
                # log scale
                im = plt.imshow(cart.eval_grid(np.array(coef_), matrix=True), origin='lower', extent=(-1, 1, -1, 1))
                plt.colorbar()
                plt.savefig('test.png')
        else:
            print('error regression algorithm')
        
        out.create_dataset('coeff' + str(order), data = coef_)
        out.create_dataset('std' + str(order), data = std)
        out.create_dataset('AIC' + str(order), data = AIC)

parser = argparse.ArgumentParser(description='Process template construction')
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='the filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='the output filename [*.h5] to save')

parser.add_argument('-v', '--verbose', type=int, choices=[0,1], default=0, 
                    help='output debug info')

parser.add_argument('--alg', dest='alg', type=str, choices=['sms', 'sk', 'custom', 'h2o'], default='sms',
                    help='Algorithm to be used in regression')

parser.add_argument('--basis', dest='basis', type=str, choices=['Legendre', 'Zernike'],default='Legendre',
                    help='Basis to be used in regression')

parser.add_argument('--order', dest='order', metavar='N', type=int, default=10,
                    help='the max cutoff order')

parser.add_argument('--figure', dest='figure', type=str, choices=['ON', 'OFF'], default='OFF',
                    help='whether plot figure')

parser.add_argument('--offset', dest='offset', type=str, choices=['ON', 'OFF'], default='OFF',
                    help='whether use offset data')

args = parser.parse_args()
print(args.filename)

main_Calib(args.filename, args.output, args.alg, args.basis, args.order, args.figure, args.verbose)
