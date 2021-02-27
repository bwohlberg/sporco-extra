#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tabulate RCA results computed by script compute_rca_results.py.
"""

import os
import glob

import numpy as np

from cdlpsf.util import sm_snr


# Function for tabulating results
def print_table(shpnum, ppsnum, tblval, textbl=False):
    shpstr = list(map(lambda x: x[0], sorted(list(shpnum.items()),
                                         key=lambda x: x[1])))
    ppstxt = ['%4.0f' % k for k in ppsnum]
    if textbl:
        print(' ' * 10  + '& ' + ' & '.join(ppstxt) +
              r' \\ \hline')
    else:
        print(' ' * 12 + '   '.join(ppstxt))
    for n in range(len(shpnum)):
        if textbl:
            print(('%-10s' % shpstr[n]) + '& ' +
                  ' & '.join(['%5.2f' % x for x in tblval[n]]) +
                  r' \\ \hline')
        else:
            print(('%-12s' % shpstr[n]) +
                  '  '.join(['%5.2f' % x for x in tblval[n]]))



# Select output format (if true, insert LaTeX formatting in table)
textbl = False


# Define standard integer sampling grid -wp ... wp
wp = 7

# Subpixel estimation factor (common for all runs)
M = 5

# Common noise level
slct_noise = 1.0

# Distinct PSF shapes available
psfshapes = ['complex', 'elong', 'narrow', 'wide']

# Paths to data files
imgpath = 'data/simulated_images'
psfpath = 'data/reference_psfs'
rsltpath = 'data/rca_results'


# Define mappings from parameter values to index values
shpnum = {
    'narrow':  0,
    'wide':    1,
    'elong':   2,
    'complex': 3
}
ppsnum = {
    1.0: 0,
    10.0: 1,
    25.0: 2,
    50.0: 3,
    100.0: 4
    }


snrdict = {}
snrarr = np.zeros((4, 5))

# Iterate over psf shapes
for shape in psfshapes:
    # Load reference psf at common subpixel resolution
    psffilename = os.path.join(psfpath, '%s.npz' % shape)
    npz = np.load(psffilename, allow_pickle=True)
    refpsf = npz['refpsf'].item()[M]

    if shape not in snrdict:
        snrdict[shape] = {}

    # Iterate over cdl interp result files
    for rsltfile in sorted(glob.glob(os.path.join(rsltpath,
                                                  '%s*.npz' % shape))):
        # Load result file data
        npz = np.load(rsltfile, allow_pickle=True)
        noise = float(npz['noise'])
        pps = float(npz['pixperstar'])

        if noise != slct_noise:
            continue

        psfgrd = np.pad(npz['psf'], (2, 2))
        snr = sm_snr(refpsf, psfgrd)

        snrdict[shape][pps] = snr
        snrarr[shpnum[shape], ppsnum[pps]] = snr

# Display table of results
print('Results for individually optimized parameters (SNR in dB):')
print_table(shpnum, ppsnum, snrarr, textbl)
print('Min: %5.2f   Mean: %5.2f   Median: %5.2f   Max: %5.2f' %
      (snrarr.min(), snrarr.mean(), np.median(snrarr), snrarr.max()))
