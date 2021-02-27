#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute RCA solutions for each test case using optimal parameters found by compute_rca_opt_param.py.
"""

import os
import glob
import copy

import numpy as np

from rca import RCA
from rca.utils import rca_format


# Define standard integer sampling grid -wp ... wp
wp = 7

# Common noise level
slct_noise = 1.0


# Paths to data files
rcainpath = 'data/rca_input'
rsltpath = 'data/rca_results'

# Get optimal parameters found via brute-force search
prmfile = os.path.join(rsltpath, 'rca_opt_param.npz')
prm = dict(np.load(prmfile, allow_pickle=True)['results'].item())

# Iterate over test images
for starfile in sorted(glob.glob(os.path.join(rcainpath, 'rca_stars_*.npy'))):
    basename = os.path.basename(starfile)
    fncmpnt = basename.split('_')
    posfile = os.path.join(rcainpath, '_'.join(
            (['rca', 'pos'] + fncmpnt[2:])))
    shape = fncmpnt[-1][0:-4]
    noise = float(fncmpnt[-2][1:])
    if noise != slct_noise:
            continue
    pps = int(fncmpnt[-3][1:])
    basename = '%s_d%03d_n%7.1e.npz' % (shape, pps, noise)
    rsltfile = os.path.join(rsltpath, basename)
    if not os.path.exists(rsltfile):

        print('Computing %s' % rsltfile)

        stars = np.load(starfile)
        pos = np.load(posfile)

        optprm = prm[basename]['sprm']
        n_comp, ksig, n_scales, ksig_init, psf_size, n_eigenvects = optprm
        n_comp = int(n_comp)
        n_scales = int(n_scales)
        n_eigenvects = int(n_eigenvects)

        try:
            rca = RCA(n_comp=n_comp, ksig=ksig, n_scales=n_scales,
                      ksig_init=ksig_init, verbose=0)
            S, A = rca.fit(rca_format(stars), pos,
                           psf_size=psf_size,
                           n_eigenvects=n_eigenvects)
            psf = rca.estimate_psf(pos[0:1]).squeeze()
        except:
            psf = None

        if psf is not None:
            np.savez(rsltfile, noise=noise, shape=shape,
                     pixperstar=pps, optprm=optprm, psf=psf)
