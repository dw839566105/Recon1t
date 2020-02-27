#!/usr/bin/env python3
'''
Energy spectra plot
'''

import h5py, matplotlib as mpl, numpy as np, argparse, pandas as pd
mpl.use('pdf')
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import pyplot as plt, cm

psr = argparse.ArgumentParser()
psr.add_argument("-o", dest='opt', help="output")
psr.add_argument('ipt', nargs="+", help="input")
args = psr.parse_args()

from Readlog import coeff3d
# coeff3d
EE_tmp, radius, coeff = coeff3d()
assert EE_tmp[4]=="1.8", "1.8 MeV curve is missing."

from numpy.polynomial import legendre as lg

c = lg.legfit(radius, coeff[:,0,4], 9)
c[1::2] = 0 # force even function

def ld(f):
    a = pd.read_hdf(f)
    rho2 = a["x_sph"]**2 + a["y_sph"]**2
    r = np.sqrt(a["z_sph"]**2 + rho2)
    E = np.exp(a["l0_sph"] - (lg.legval(r, c) - np.log(1.8)))
    return pd.DataFrame({"E":E, "r":r, "rho2":rho2, "z":a["z_sph"]})

d = pd.concat(map(ld, args.ipt))

rl = np.arange(0.65, 0.25, -0.05)
b = np.arange(0, 9, 0.05)

pp = PdfPages(args.opt)
for r in rl:
    plt.hist(d.query("r<{}".format(r))['E'], bins=b, 
             label="r<{:.2f}m".format(r))
plt.xlabel("E/MeV")
plt.ylabel("count/0.1 MeV")
plt.legend()
pp.savefig()
plt.yscale("log")
pp.savefig()

plt.clf()
ds=d.query("E < 3 & E > 2")
plt.hist2d(ds['rho2'], ds['z'], cmap=cm.PuBu, 
           bins=[np.linspace(0, 0.65*0.65, 50), np.linspace(-0.65, 0.65, 50)])
plt.xlabel("x*x+y*y/m^2")
plt.ylabel("z/m")
plt.colorbar()
pp.savefig()
pp.close()
