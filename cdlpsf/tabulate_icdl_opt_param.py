#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tabulate ICDL results computed by script compute_cdl_opt_param.py.
"""

import os
import sys

import numpy as np

from sporco import plot


# Function for tabulating results
def print_table(shpnum, ppsnum, tblval, textbl=False):
    shpstr = list(map(lambda x: x[0], sorted(list(shpnum.items()),
                                         key=lambda x: x[1])))
    if textbl:
        print(' ' * 10  + '& ' + ' & '.join(ppsnum.keys()) +
              r' \\ \hline')
    else:
        print(' ' * 12 + '   '.join(ppsnum.keys()))
    for n in range(len(shpnum)):
        if textbl:
            print(('%-10s' % shpstr[n]) + '& ' +
                  ' & '.join(['%5.2f' % x for x in tblval[n]]) +
                  r' \\ \hline')
        else:
            print(('%-12s' % shpstr[n]) +
                  '  '.join(['%5.2f' % x for x in tblval[n]]))


# Load data
rsltpath = 'data/icdl_results'
if len(sys.argv) > 1:
    rsltfile = sys.argv[1]
else:
    rsltfile = os.path.join(rsltpath, 'icdl_opt_param_pps.npz')
npz = np.load(rsltfile, allow_pickle=True)
results = npz['results'].item()
paramranges = npz['paramranges'].item()

# Select output format (if true, insert LaTeX formatting in table)
textbl = False

# Define mappings from parameter strings to index values
shpnum = {
    'narrow':  0,
    'wide':    1,
    'elong':   2,
    'complex': 3
}
ppsnum = {
    'd001': 0,
    'd010': 1,
    'd025': 2,
    'd050': 3,
    'd100': 4
    }

# Construct array of best SNR values
maxsnr = np.zeros((len(shpnum), len(ppsnum)))
for key in sorted(results.keys()):
    keycmpnt = key[:-4].split('_')
    shape = keycmpnt[0]
    pps = keycmpnt[1]
    maxsnr[shpnum[shape], ppsnum[pps]] = results[key]['sfvl']

# Display table of results
print('Results for individually optimized parameters (SNR in dB):')
print_table(shpnum, ppsnum, maxsnr, textbl)
print('Min: %5.2f   Mean: %5.2f   Median: %5.2f   Max: %5.2f' %
      (maxsnr.min(), maxsnr.mean(), np.median(maxsnr), maxsnr.max()))
