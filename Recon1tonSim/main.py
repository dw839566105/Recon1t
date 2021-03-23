# recon range: [-1,1], need * detector radius
import numpy as np
import scipy, h5py
import scipy.stats as stats
import os,sys
import tables
import scipy.io as scio
import matplotlib.pyplot as plt
import uproot
import awkward as ak
import argparse, textwrap
from argparse import RawTextHelpFormatter
from scipy.optimize import minimize
from scipy import interpolate
from numpy.polynomial import legendre as LG
from scipy import special
from scipy.linalg import norm
from scipy.stats import norm as normpdf
from scipy.spatial import distance
import warnings
warnings.filterwarnings('ignore')
np.set_printoptions(precision=3, suppress=True)

# boundaries
shell = 0.65
Gain = 164
sigma = 40

import pub

def Recon(filename, output, pe, time, PE, TIME, offset, types, initial, MC, method):

    '''
    reconstruction

    fid: root reference file convert to .h5
    fout: output file
    '''
    # Create the output file and the group
    print(filename) # filename
    PMT_pos = 
    # Create the output file and the group
    h5file = tables.open_file(output, mode="w", title="OneTonDetector",
                            filters = tables.Filters(complevel=9))
    group = "/"
    # Create tables
    ReconTable = h5file.create_table(group, "Recon", pub.ReconData, "Recon")
    recondata = ReconTable.row
    # Loop for event

    f = uproot.open(filename)
    data = f['SimTriggerInfo']
    if types == 'root':
        PMTId = ak.to_numpy(ak.flatten(data['PEList.PMTId'].array()))
        Time = ak.to_numpy(ak.flatten(data['PEList.HitPosInWindow'].array()))
        Charge = ak.to_numpy(ak.flatten(data['PEList.Charge'].array()))    
        SegmentId = ak.to_numpy(ak.flatten(data['truthList.SegmentId'].array()))
        VertexId = ak.to_numpy(ak.flatten(data['truthList.VertexId'].array()))
        x = ak.to_numpy(ak.flatten(data['truthList.x'].array()))
        y = ak.to_numpy(ak.flatten(data['truthList.y'].array()))
        z = ak.to_numpy(ak.flatten(data['truthList.z'].array()))
        E = ak.to_numpy(ak.flatten(data['truthList.EkMerged'].array()))
    
        for pmt, time_array, pe_array, sid, vid, xt, yt, zt, Et in zip(PMTId, Time, Charge, SegmentId, VertexId, x, y, z, E):
            recondata['x_truth'] = xt
            recondata['y_truth'] = yt
            recondata['z_truth'] = zt
            recondata['E_truth'] = Et
            
            # PMT order: 0-29
            # PE /= Gain
            # pe_array, cid = np.histogram(pmt, bins=np.arange(31)-0.5, weights=PE)

            # For hit info
            pe_array, cid = np.histogram(pmt, bins=np.arange(31)) 
            # For very rough estimate
            #pe_array = np.round(pe_array)
        
            time_array = time
            fired_PMT = pmt

            # calculate pdf template
            '''
            ## DO NOT USE IN LUCY DDM 
            N0 = np.atleast_2d(np.round(PE/Gain)).T \
                - np.atleast_2d(np.arange(-3,3)) # range: -10:10
            sigma_array = sigma/Gain*np.sqrt(N0)
            pdf_weight = normpdf.pdf(np.atleast_2d(PE/Gain).T,\
                N0, \
                np.atleast_2d(sigma_array)+1e-6 \
                )
            pdf_weight[N0<0] = 0
            N0[N0<0] = 0
            '''
        
            '''
            # Possible N for pe_array
            N = np.atleast_2d(np.round(pe_array)).T \
                - np.atleast_2d(np.arange(-10,10)) # range: -10:10
            # relative sigma_N
            sigma_array = sigma/Gain*np.sqrt(N)
            # Gaussian pdf
            pdf_pe = normpdf.pdf(np.atleast_2d(pe_array).T, 
                N, \
                np.atleast_2d(sigma_array)+1e-6 \
                )
            # cut negative
            pdf_pe[N<0] = 0
            N[N<0] = 0
            '''
            
            if np.sum(pe_array)!=0:
                # Use MC to find the initial value
                # x0 = pub.initial.wa
                x0 = np.zeros(6)
                # inner recon
                # initial value
                # Energy recon will be removed later
                x0_in = np.zeros(5)
                x0_in[0] = 0.8 + np.log(np.sum(pe_array)/60)
                x0_in[4] = np.quantile(time_array,0.1)

                footnote{or anything equivalent} reconstruction.
                result_in = minimize(Likelihood, x0_in, method='SLSQP',bounds=((-10, 10), (0, 1), (None, None), (None, None), (None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, PE))
                z, x = Calc_basis(result_in.x, PMT_pos, cut_pe)
                L, E_in = Likelihood_PE(z, x, coeff_pe, pe_array, cut_pe, N, pdf_pe)

            # xyz coordinate
            in2 = r2c(result_in.x[1:4])*shell
            recondata['x_sph_in'] = in2[0]
            recondata['y_sph_in'] = in2[1]
            recondata['z_sph_in'] = in2[2]
            recondata['success_in'] = result_in.success
            recondata['Likelihood_in'] = result_in.fun
            
            # outer recon
            # initial value is a 30*30 grid at 0.92 * radius
            vertex = x0_in[0]
            vertex[1] = 0.92
            y = np.linspace(0, 2*np.pi, 30)
            x = np.linspace(0, np.pi, 30)
            xx, yy = np.meshgrid(x, y, sparse=False)
            mesh = np.hstack((xx.reshape(-1,1), yy.reshape(-1,1)))
            data = np.zeros_like(mesh[:,0])
            for i in np.arange(np.size(data)):
                vertex[2:4] = mesh[i]
                data[i] = Likelihood(vertex, *(coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, PE))
            index = np.where(data==np.min(data)) 
            
            # choose best vertex to outer recon
            x0_out = x0_in[0]
            x0_out[1:4] = np.array((0.92, mesh[index[0][0],0], mesh[index[0][0],1]))
            result_out = minimize(Likelihood, x0_out, method='SLSQP',bounds=((E_min, E_max), (0,1), (None, None), (None, None),(None, None)), args = (coeff_time, coeff_pe, PMT_pos, fired_PMT, time_array, pe_array, cut_time, cut_pe, N, pdf_pe, PE))
            z, x = (Calc_basis(result_out.x, PMT_pos, cut_pe))
            L, E_out = Likelihood_PE(z,x, coeff_pe, pe_array, cut_pe, N, pdf_pe)
            
            out2 = r2c(result_out.x[1:4]) * shell
            recondata['x_sph_out'] = out2[0]
            recondata['y_sph_out'] = out2[1]
            recondata['z_sph_out'] = out2[2]
            recondata['success_out'] = result_out.success
            recondata['Likelihood_out'] = result_out.fun
            
            # 0-th order (Energy intercept)
            base_in = LG.legval(result_in.x[1], coeff_pe.T)
            base_out = LG.legval(result_out.x[1], coeff_pe.T)

            print('-'*60)
            print(f'inner: {np.exp(E_in - base_in[0] + np.log(2))}')
            print(f'outer: {np.exp(E_out - base_out[0] + np.log(2))}')
            template_E = 2/2 * 4285/4285 # template is 2.0 MeV, light yield 4285/MeV
            recondata['E_sph_in'] = np.exp(E_in - base_in[0] + np.log(template_E) + np.log(2))
            recondata['E_sph_out'] = np.exp(E_out - base_out[0] + np.log(template_E) + np.log(2))

            print('inner')
            print(f'Template likelihood: {-np.max(L)}')
            print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, in2[0], in2[1], in2[2], norm(in2), result_in.fun))
            print('outer')
            print('%d vertex: [%+.2f, %+.2f, %+.2f] radius: %+.2f, Likelihood: %+.6f' % (event_count, out2[0], out2[1], out2[2], norm(out2), result_out.fun))
            
        else:
            recondata['x_sph_in'] = 0
            recondata['y_sph_in'] = 0
            recondata['z_sph_in'] = 0
            recondata['E_sph_in'] = 0
            recondata['success_in'] = 0
            recondata['Likelihood_in'] = 0
            
            recondata['x_sph_out'] = 0
            recondata['y_sph_out'] = 0
            recondata['z_sph_out'] = 0
            recondata['E_sph_out'] = 0
            recondata['success_out'] = 0
            recondata['Likelihood_out'] = 0
            print('empty event!')
            print('-'*60)
        recondata.append()
        event_count = event_count + 1

    # Flush into the output file
    ReconTable.flush()
    h5file.close()

# Automatically add multiple root files created a program with max tree size limitation.



parser = argparse.ArgumentParser(description='Process Reconstruction construction', formatter_class=RawTextHelpFormatter)
parser.add_argument('-f', '--filename', dest='filename', metavar='filename[*.h5]', type=str,
                    help='The filename [*Q.h5] to read')

parser.add_argument('-o', '--output', dest='output', metavar='output[*.h5]', type=str,
                    help='The output filename [*.h5] to save')

parser.add_argument('--pe', dest='pe', metavar='PECoeff[*.h5]', type=str,
                    help='The pe coefficients file [*.h5] to be loaded')

parser.add_argument('--time', dest='time', metavar='TimeCoeff[*.h5]', type=str,
                    help='The time coefficients file [*.h5] to be loaded')

parser.add_argument('--split', dest='split', choices=[True, False], default=True,
                    help='Whether the coefficients is segmented')

parser.add_argument('--PE', dest='PE', choices=[True, False], default=True,
                    help='Whether use PE info')

parser.add_argument('--TIME', dest='TIME', choices=[True, False], default=True,
                    help='Whether use Time info')

parser.add_argument('--offset', dest='offset', metavar='offset[*.h5]', type=str, default=False,
                    help='Whether use offset data, default is 0')

parser.add_argument('--type', dest='type', choices=['h5', 'root', 'wave'], default='root',
                    help = 'Load file type')

parser.add_argument('--initial', dest='initial', choices=['WA','MC','fit'], default='MC',
                    help = 'initial point method')

parser.add_argument('--MC', dest='MC', metavar='MCGrid[*.h5]', type=str,
                    help='The MC grid file [*.h5] to be loaded')

parser.add_argument('--method', dest='method', choices=['1','2','3'], default='2',
                    help=textwrap.dedent('''Method to be used in reconstruction.
                    '1' - Fit E,
                    '2' - normalized E,
                    '3' - dual annealing'''))
args = parser.parse_args()
print(args.filename)

# Read PMT position
#PMT_pos = np.loadtxt(r"./PMT_1t.txt")

#coeff_pe, coeff_time, cut_pe, fitcut_pe, cut_time, fitcut_time\
#    = load_coeff()
#tp, bins = readtpl()
#args = PMT_pos, tp, bins

Recon(args.filename, args.output, args.pe, args.time, args.PE, args.TIME, args.offset, args.type, args.initial, args.MC, args.method)


