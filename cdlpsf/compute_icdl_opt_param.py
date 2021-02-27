#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find optimal ICDL parameters for each test case by optimizing over 768 different parameter combinations.
"""

import os
import sys
import glob
import copy
from socket import gethostname
from functools import partial

import numpy as np
from psutil import cpu_count

import sporco.fft
sporco.fft.pyfftw_threads = 1
from sporco.signal import gaussian
from sporco.util import idle_cpu_count, grid_search

from cdlpsf.cdl import PSFEstimator
from cdlpsf.util import sm_snr



# Define standard integer sampling grid -wp ... wp
wp = 7

# Subpixel estimation factor (common for all runs)
M = 5

# Common noise level
slct_noise = 1.0


# Evaluate PSF estimation performance for specified parameters
def psf_est_performance(prm, K, img, psfref):
    sigma0, lmbdaX, lmbdaG, rhoX, LG = prm
    g0 = gaussian((2*wp+1,) * 2, sigma0)
    g0 /= np.linalg.norm(g0)
    opt = PSFEstimator.Options(
            {'Verbose': False, 'MaxMainIter': 100,
             'XslvIter0': 10, 'GslvIter0': 10,
             'Xslv': {'NonNegCoef': False, 'rho': rhoX},
             'Gslv': {'L': LG}})
    pe = PSFEstimator(img, g0, lmbdaX, lmbdaG, M, K, opt)
    psfgrd = pe.solve()
    return sm_snr(psfref, psfgrd)



# Algorithm parameter ranges
paramranges = {
    'sigma0': np.array([0.5, 1.0, 1.5, 2.0]),
    'lmbdaX': np.array([1e-3, 1e-2, 1e-1, 5e-1]),
    'lmbdaG': np.array([1e-2, 5e-2, 1e-1, 5e-1]),
    'rhoX': np.array([1e0, 1e1, 1e2]),
    'LG': np.array([5e1, 1e2, 5e2, 1e3])
}

prmcmb = np.product([paramranges[v].size for v in paramranges])
print('Evaluating %d parameter combinations per input file' % prmcmb)

# Paths to data files
imgpath = 'data/simulated_images'
psfpath = 'data/reference_psfs'
rsltpath = 'data/icdl_results'
if not os.path.isdir(rsltpath):
    os.makedirs(rsltpath)
rsltfile = os.path.join(rsltpath, 'icdl_opt_param_pps.npz')

# Number of processors to use for parameter evaluation
nproc = cpu_count(logical=False)

results = {}
imgfiles = sorted(glob.glob(os.path.join(imgpath, '*.npz')))
# Iterate over input image files
for imgfile in imgfiles:
    # Load data from simulated image file
    npz = np.load(imgfile)
    img = npz['imgn']
    noise = float(npz['noise'])
    shape = str(npz['shape'])
    pps = int(npz['pixperstar'])

    # Skip input images for different noise levels
    if noise != slct_noise:
        continue

    basename = os.path.basename(imgfile)[0:-4]
    print('Computing parameters for %s' % imgfile)

    # Rescale image range
    imx = img.max()
    img /= imx

    # Select Lanczos kernel order depending on PSF shape
    if shape == 'complex' or shape == 'narrow':
        K = 5
    else:
        K = 10

    # Load reference PSF
    psffile = os.path.join(psfpath, '%s.npz' % shape)
    npz = np.load(psffile, allow_pickle=True)
    psfref = npz['refpsf'].item()[M]

    # Evaluate performance over selected parameter ranges
    perf = partial(psf_est_performance, K=K, img=img, psfref=psfref)
    sprm, sfvl, fvmx, sidx = grid_search(
        perf, (paramranges['sigma0'], paramranges['lmbdaX'],
               paramranges['lmbdaG'], paramranges['rhoX'],
               paramranges['LG']),
        fmin=False, nproc=nproc)

    # Add results for this image to results dict
    results[basename] = {'shape': shape, 'pps': pps, 'noise': noise,
                         'K': K, 'sprm': sprm, 'sfvl': sfvl, 'fvmx': fvmx,
                         'sidx': sidx}

    # Save data to NPZ file
    np.savez(rsltfile, paramranges=paramranges, results=results)
