#### Calib
This directory is to calib coefficients based on simulation files, files saved under '/mnt/stage/douwei/Simulation/1t_root'
Different dir shows different data
> dns shows no dark noise and same PMT
> dn shows no dark noise
> xyz shows 3 axis data, with dark noise and threshold 1
> 015 is just 1 axis, with dark noise and threshold 25

you need to change file path in make file to get different calib. However, the default(dns) is the most ideal case and is what we want now.

#### Usage:
>>>make or make -f Makefile

#### Algorithm:
##### PE_calib
Use hit information, by Poisson regression, decompose the log execeted hit by Legendre polynomials
##### Time_calib
Use time information, by quantile regression, decompose the flight time by Legendre polynomials

One can choose a suitable order to fit, but it is still under trials.
