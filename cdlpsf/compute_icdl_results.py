#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Compute ICDL solutions using parameter selection heuristics described in appendix of extended version (arXiv) of paper.
"""

import os
import sys
import glob

import numpy as np

from sporco.signal import gaussian
from sporco.fft import fftconv
from sporco.plot import imview

from cdlpsf.cdl import PSFEstimator
from cdlpsf.util import translatescale, sm_snr
from cdlpsf.plot import plot_solver_stats, plot_psf_sections, plot_psf_contours



# Define standard integer sampling grid -wp ... wp
wp = 7

# Subpixel estimation factor (common for all runs)
M = 5

# Common noise level
slct_noise = 1.0

# Paths to data files
imgpath = 'data/simulated_images'
rsltpath = 'data/icdl_results'
if not os.path.isdir(rsltpath):
    os.makedirs(rsltpath)


# If input file specified on command line run just that case, otherwise
# run all input files for which saved results not found
if len(sys.argv) > 1:
    imgfiles = [os.path.join(imgpath, sys.argv[1]),]
    runall = False
else:
    imgfiles = sorted(glob.glob(os.path.join(imgpath, '*.npz')))
    runall = True

# Iterate over input image files
for imgfile in imgfiles:
    basename = os.path.basename(imgfile)
    rsltfile = os.path.join(rsltpath, basename)
    if not runall or not os.path.exists(rsltfile):

        print('Computing %s' % rsltfile)

        # Load data from simulated image file
        npz = np.load(imgfile)
        img = npz['imgn']
        noise = float(npz['noise'])
        shape = str(npz['shape'])
        pps = int(npz['pixperstar'])

        # Skip input images for different noise levels
        if noise != slct_noise:
            continue

        # Rescale input image
        imx = img.max()
        img /= imx

        # Set algorithm parameters
        K = 5 if shape == 'complex' or shape == 'narrow' else 10
        sigma0 = 1.0 if shape == 'complex' or shape == 'wide' else 0.5
        lmbdaX = 1e-2 if pps <= 50 else 1e-1
        lmbdaG = 1e-2 if pps <= 1 else 1e-1
        rhoX = 1.0 if pps <= 50 else 10.0
        if pps <= 1:
            LG = 5e1
        elif pps <= 25:
            LG = 1e2
        elif pps <= 50:
            LG = 5e2
        else:
            LG = 1e3
        param = {'K': K, 'sigma0': sigma0, 'lmbdaX': lmbdaX, 'lmbdaG': lmbdaG,
                 'rhoX': rhoX, 'LG': LG}

        # Construct initial solution
        g0 = gaussian((2*wp+1,) * 2, sigma0)
        g0 /= np.linalg.norm(g0)

        # Set options and run solver
        opt = PSFEstimator.Options(
            {'Verbose': (not runall), 'MaxMainIter': 100,
             'XslvIter0': 10, 'GslvIter0': 10,
             'Xslv': {'NonNegCoef': False, 'rho': rhoX},
             'Gslv': {'L': LG}})
        pe = PSFEstimator(img, g0, lmbdaX, lmbdaG, M, K, opt)
        psfgrd = pe.solve()
        psfspx = pe.get_psf(subpixel=True, tkhflt=True)
        itsX = pe.slvX.getitstat()
        itsG = pe.slvG.getitstat()

        # Save parameters and results
        if runall:
            np.savez(rsltfile, noise=noise, shape=shape, pixperstar=pps,
                     M=M, param=param, psfgrd=psfgrd, psfspx=psfspx,
                     itsX=itsX, itsG=itsG)

        # If a specific input image has been selected, print performance
        # metrics and plot results
        if not runall:
            plot_solver_stats(pe)
            psfpath = 'data/reference_psfs'
            psffilename = os.path.join(psfpath, '%s.npz' % shape)
            npz = np.load(psffilename, allow_pickle=True)
            refpsfspx = npz['refpsf'].item()[M]
            print('PSF estimation SNR: %5.2f dB' % sm_snr(refpsfspx, psfgrd))
            psfspx = translatescale(refpsfspx, psfspx)
            spxgrd = pe.get_subpix_grid(np.linspace(-wp, wp, 2*wp+1))
            plot_psf_sections(refpsfspx, psfspx, spxgrd,
                              title='PSF Comparison (sub-pixel grid)')
            plot_psf_contours(refpsfspx, psfspx, spxgrd,
                              title='PSF Comparison (sub-pixel grid)')
            Xhs = np.sum(fftconv(pe.H, pe.X.squeeze(), axes=(0, 1)), axis=-1)
            imview(Xhs)
            input()
