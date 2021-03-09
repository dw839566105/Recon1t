import sys
import pub
import numpy as np

import tables, h5py
import argparse
from argparse import RawTextHelpFormatter
import argparse, textwrap
import time

np.set_printoptions(formatter={'float': '{: 0.3f}'.format})
# global:
PMTPos = pub.ReadJPPMT()

def main_Calib(filename, output, mode, alg, basis, order, figure, verbose, offset):
    '''
    # main program
    # input: radius: %+.3f, 'str' (in makefile, str is default)
    #        path: file storage path, 'str'
    #        fout: file output name as .h5, 'str' (.h5 not included')
    #        cut_max: cut off of Legendre
    # output: the gathered result EventID, ChannelID, x, y, z
    '''
    print('begin reading file', flush=True)

    EventID, ChannelID, Q, PETime, photonTime, PulseTime, dETime, x, y, z = pub.ReadFile(filename)
    VertexTruth = (np.vstack((x, y, z))/1e3).T
    if(offset):
        off = pub.LoadBase(offset)
    else:
        off = np.zeros_like(PMTPos[:,0])
    print('total event: %d' % np.size(np.unique(EventID)), flush=True)
    print('begin processing legendre coeff', flush=True)
    # this part for the same vertex

    tmp = time.time()
    EventNo = np.size(np.unique(EventID))
    PMTNo = np.size(PMTPos[:,0])
    if mode == 'PE':
        PMTPosRep = np.tile(PMTPos, (EventNo,1))
        vertex = np.repeat(VertexTruth, PMTNo, axis=0)
    elif mode == 'time':
        counts = np.bincount(EventID)
        counts = counts[counts!=0]
        PMTPosRep = PMTPos[ChannelID]
        vertex = np.repeat(VertexTruth, counts, axis=0)
    elif mode == 'combined':
        PMTPosRep = np.tile(PMTPos, (EventNo,1))
        vertex = np.repeat(VertexTruth, PMTNo, axis=0)

    if basis == 'Legendre':
        X, cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=True)
    elif basis == 'Zernike':
        from zernike import RZern
        cos_theta = pub.LegendreCoeff(PMTPosRep, vertex, order, Legendre=False)
        cart = RZern(order)
        nk = cart.nk
        m = cart.mtab
        n = cart.ntab
        rho = np.linalg.norm(vertex, axis=1)/0.65
        theta = np.arccos(cos_theta)        
        X = np.zeros((rho.shape[0], nk))
        for i in np.arange(nk):
            if not i % 5:
                print(f'process {i}-th event')
            X[:,i] = cart.Zk(i, rho, theta)
        X = X[:,m>=0]
        print(f'rank: {np.linalg.matrix_rank(X)}')    
    print(f'use {time.time() - tmp} s')

    # which info should be used
    if mode == 'PE':
        y = Q
    elif mode == 'time':
        y = PulseTime 
    elif mode == 'combined':
        PulseTime = PulseTime/np.max(PulseTime)
        bins = np.arange(0,1,0.1)

        N = 10
        x = pub.legval(bins, np.eye(N).reshape(N, N, 1))
        Y = np.tile(x, len(np.unique(EventID))*len(np.unique(ChannelID))).T
        #basis = np.zeros_like(X)
        
        X = np.repeat(X, bins.shape[0], axis=0)
        y = np.zeros((len(np.unique(EventID)), len(np.unique(ChannelID)), len(bins)))

        for k_index, k in enumerate(np.unique(EventID)): # event begin with 1
            if not k % 100:
                print(k)
            index = EventID == k
            CID = ChannelID[index]
            Pulse_t = PulseTime[index]
            for i in np.unique(CID): # PMT begin with 0
                y[k_index, i, 1:], _ = np.histogram(Pulse_t[CID==i], bins=bins)
        y = np.reshape(y,(-1))

    print(y.shape,X.shape)

    if verbose:
        print(f'the basis shape is {X.shape}, and the dependent variable shape is {y.shape}') 

    # Regression methods:
    if (alg == 'sms'):
        import statsmodels.api as sm
        if mode != 'combined':
            if mode == 'PE':
                model = sm.GLM(y, X, family=sm.families.Poisson())
                result = model.fit()

                if verbose:
                    print(result.summary())
                AIC = result.aic
                coef_ = result.params
                std = result.bse

            elif mode == 'time':
                import pandas as pd
                data = pd.DataFrame(data = np.hstack((X, np.atleast_2d(y).T)))                
                strs = 'y ~ '
                start = data.keys().start
                stop = data.keys().stop
                step = data.keys().step

                cname = []
                cname.append('X0')
                for i in np.arange(start+1, stop, step):
                    if i == start + 1:
                        strs += 'X%d ' % i
                    elif i == stop - step:
                        pass
                    else:
                        strs += ' + X%d ' % i                      

                    if i == stop - step:
                        cname.append('y')
                    else:
                        cname.append('X%d' % i)
                data.columns = cname

                mod = sm.formula.quantreg(strs, data[cname])

                res = mod.fit(q=0.1,)
                coef_ = res.params
                AIC = np.zeros_like(coef_)
                std = np.zeros_like(coef_)
        elif mode == 'combined':

            basis = np.zeros((X.shape[0], X.shape[1]*Y.shape[1]))

            for i_index, i in enumerate(np.arange(X.shape[1])):
                for j_index, j in enumerate(np.arange(Y.shape[1])):
                    total_index = i_index*Y.shape[1] + j_index
                    if not total_index % 10:
                        print(total_index)
                    basis[:, total_index] = X[:,i_index]*Y[:,j_index]
            # data = pd.DataFrame(data = np.hstack((basis, np.atleast_2d(y).T)))  
            with h5py.File(output,'w') as out:        
                out.create_dataset('X', data = basis)
                out.create_dataset('Y', data = y)
            print(basis.shape, y.shape)
            breakpoint()
            print('begin...')
            model = sm.GLM(y, basis, family=sm.families.Poisson())
            result = model.fit()

            if verbose:
                print(result.summary())
            AIC = result.aic

            print('Waring! No AIC and std value')
        if verbose:
            print(res.summary())

    elif (alg == 'custom'):
        from scipy.optimize import minimize
        x0 = np.zeros_like(X[0]) # initial value (be careful of Zernike order)
        
        if mode == 'PE':
            x0[0] = 0.8 + np.log(2) # intercept is much more important
            result = minimize(pub.CalibPE, x0=x0, method='SLSQP', args = (y, PMTPos, X))
        elif mode == 'time':
            x0[0] = np.mean(y)
            qt = 0.1
            ts = 2.6
            result = minimize(pub.CalibTime, x0=x0, method='SLSQP', args = (np.hstack((EventID, EventID)), y, X, qt, ts))
        elif mode == 'combined':
            x0[0] = 0.8 + np.log(2) # intercept is much more important
            result = minimize(pub.CalibPE, x0=x0, method='SLSQP', args = (y, PMTPos, X))

        coef_ = np.array(result.x, dtype=float)
        if verbose:
            print(result.message)
        AIC = np.zeros_like(coef_)
        std = np.zeros_like(coef_)

        H = pub.MyHessian(result.x, pub.CalibPE, *(y, PMTPos, X))
        # H = pub.MyHessian(result.x, *(Q, PMTPos, X, pub.CalibTime))
        # std = 1/np.sqrt(-np.diag(np.linalg.pinv(H1)))
        print(coef_)
        # print(std)
        print('Waring! No AIC and std value, std is testing')

    elif alg == 'sk':
        from sklearn.linear_model import TweedieRegressor
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

    elif alg == 'h2o':
        import h2o
        from h2o.estimators.gbm import H2OGradientBoostingEstimator
        from h2o.estimators.glm import H2OGeneralizedLinearEstimator           
        if mode != 'combined':
            y = np.atleast_2d(y).T
            data = np.hstack((X, y, np.ones_like(y)))

            h2o.init()
            hf = h2o.H2OFrame(data)
            predictors = hf.columns[0:-2]
            response_col = hf.columns[-2]

            if mode == 'PE':
                #offset_col = hf.columns[-1]
                glm_model = H2OGeneralizedLinearEstimator(family= "poisson",
                    #offset_column = offset_col, 
                    lambda_ = 0,
                    compute_p_values = True)

                glm_model.train(predictors, response_col, training_frame=hf)

                coef_table = glm_model._model_json['output']['coefficients_table']
                coef_ = glm_model.coef()

            elif mode == 'time':
                gbm = H2OGradientBoostingEstimator(distribution="quantile", seed = 1234,
                                                  stopping_metric = "mse", stopping_tolerance = 1e-4)
                gbm.train(x = predictors, y = response_col, training_frame = hf)
                breakpoint()
                print(gbm)
                exit()
        elif mode == 'combined':
            y = np.atleast_2d(y).T
            data = np.hstack((X, Y, y, np.ones_like(y)))

            h2o.init() 
            hf = h2o.H2OFrame(data)
            breakpoint()
            predictors = hf.columns[0:-2]
            response_col = hf.columns[-2]           

        if verbose:
            print(coef_)
            if basis == 'Zernike':
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
            
    with h5py.File(output,'w') as out:        
        out.create_dataset('coeff' + str(order), data = coef_)
        out.create_dataset('std' + str(order), data = std)
        out.create_dataset('AIC' + str(order), data = AIC)

parser = argparse.ArgumentParser(description='Process template construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('-v', '--verbose', type=int, choices=[0,1], default=0, 
                    help='Output debug info')

parser.add_argument('--mode', dest='mode', type=str, choices=['PE', 'time','combined'], default='PE',
                    help='Which info should be used')

parser.add_argument('--alg', dest='alg', type=str, choices=['sms', 'sk', 'h2o', 'custom'], default='sms',
                    help=textwrap.dedent('''Algorithm to be used in regression.
                    PE    -  "sms", "sk", "h2o", "custom".
                    time  -  "sms", "custom".'''))

parser.add_argument('--basis', dest='basis', type=str, choices=['Legendre', 'Zernike'],default='Legendre',
                    help=textwrap.dedent('''Basis to be used in regression.
                    Legendre  -  For data with the same radius. (1d)
                    Zernike  -  For data random distributed. (2d)'''))

parser.add_argument('--order', dest='order', metavar='N', type=int, default=10,
                    help=textwrap.dedent('''The max cutoff order. 
                    For Zernike is (N+1)*(N+2)/2'''))

parser.add_argument('--figure', dest='figure', metavar='filename[*.png]', type=str, default=False,
                    help=textwrap.dedent('''Whether plot figure, default is "OFF".
                    Figure is a unit diskonly effect for Zernike now'''))

parser.add_argument('--offset', dest='offset', metavar='filename[*.h5]', type=str, default=False,
                    help='Whether use offset data, default is 0')

args = parser.parse_args()
print(args.filename)

main_Calib(args.filename, args.output, args.mode, args.alg, args.basis, args.order, args.figure, args.verbose, args.offset)
