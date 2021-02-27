#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Find optimal RCA parameters for each test case by optimizing over 9000 different parameter combinations.
"""

import os
import glob
from functools import partial

import numpy as np
from psutil import cpu_count

from sporco.util import grid_search

from cdlpsf.util import sm_snr

from rca import RCA
from rca.utils import rca_format


# NB: Before running (see INSTALL_RCA.rst):
#     export PATH=$PATH:/opt/sparse2d/bin
#     conda activate py38rca


# Define standard integer sampling grid -wp ... wp
wp = 7
base_psf_size = (2.0 * wp + 1) / 2.0

# Subpixel estimation factor for SNR calculations
M = 5

# Common noise level
slct_noise = 1.0


# Evaluate PSF estimation performance for specified parameters
def psf_est_performance(prm, stars, pos, psfref):
    n_comp, ksig, n_scales, ksig_init, psf_size, n_eigenvects = prm
    try:
        rca = RCA(n_comp=n_comp, ksig=ksig, n_scales=n_scales,
                  ksig_init=ksig_init, verbose=0)
        S, A = rca.fit(rca_format(stars), pos, psf_size=psf_size,
                       n_eigenvects=n_eigenvects)
        cdlpsf = rca.estimate_psf(pos[0:1]).squeeze()
        cdlpsf = np.pad(cdlpsf, (2, 2))
        snr = sm_snr(psfref, cdlpsf)
    except:
        snr = 0.0
    return snr


# Algorithm parameter ranges
paramranges = {
    'n_comp': np.arange(2, 7),
    'ksig': np.arange(1, 5),
    'n_scales': np.arange(1, 4),
    'ksig_init': np.arange(1, 6),
    'psf_size': base_psf_size * np.linspace(0.5, 1.5, 5),
    'n_eigenvects': np.arange(2, 8)
}

prmcmb = np.product([paramranges[v].size for v in paramranges])
print('Evaluating %d parameter combinations per input file' % prmcmb)

# Paths to data files
rcainpath = 'data/rca_input'
psfpath = 'data/reference_psfs'
rsltpath = 'data/rca_results'
if not os.path.isdir(rsltpath):
    os.makedirs(rsltpath)
rsltfile = os.path.join(rsltpath, 'rca_opt_param.npz')

# Number of processors to use for parameter evaluation
nproc = cpu_count(logical=False)

results = {}
# Iterate over input data files
for starfile in sorted(glob.glob(os.path.join(rcainpath, 'rca_stars_*.npy'))):
    basename = os.path.basename(starfile)
    fncmpnt = basename.split('_')
    posfile = os.path.join(rcainpath, '_'.join(
            (['rca', 'pos'] + fncmpnt[2:])))
    shape = fncmpnt[-1][0:-4]
    noise = float(fncmpnt[-2][1:])
    pps = int(fncmpnt[-3][1:])

    # Skip input images for different noise levels
    if noise != slct_noise:
        continue

    basename = '%s_d%03d_n%7.1e.npz' % (shape, pps, noise)
    print('Computing parameters for %s' % starfile)

    stars = np.load(starfile)
    pos = np.load(posfile)

    # Load reference PSF
    psffile = os.path.join(psfpath, '%s.npz' % shape)
    npz = np.load(psffile, allow_pickle=True)
    psfref = npz['refpsf'].item()[M]

    # Evaluate performance over selected parameter ranges
    perf = partial(psf_est_performance, stars=stars, pos=pos, psfref=psfref)
    sprm, sfvl, fvmx, sidx = grid_search(
        perf, (paramranges['n_comp'], paramranges['ksig'],
               paramranges['n_scales'], paramranges['ksig_init'],
               paramranges['psf_size'], paramranges['n_eigenvects']),
        fmin=False, nproc=nproc)

    # Add results for this image to results dict
    results[basename] = {'shape': shape, 'pps': pps, 'noise': noise,
                         'sprm': sprm, 'sfvl': sfvl, 'fvmx': fvmx,
                         'sidx': sidx}

    # Save data to NPZ file
    np.savez(rsltfile, paramranges=paramranges, results=results)
