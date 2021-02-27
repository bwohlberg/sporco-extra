#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Construct PSF estimation performance comparisons included in the appendix of the  extended version (arXiv) of the paper.
"""

import os

import numpy as np

from sporco.interp import interpolation_points
from sporco.metric import snr
from sporco import plot

from cdlpsf.util import interpolate
from cdlpsf.util import translatescale


clrs = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 'black', '#9467bd']


def get_psf_arrays(noise, pps, shape, M, wp, psfpath, rcapath, cdlpath):

    if shape == 'complex' or shape == 'narrow':
        K = 5
    else:
        K = 10

    rsp = interpolation_points(M)
    g1d = np.linspace(-wp, wp, 2*wp+1)
    grd = (g1d[:, np.newaxis] + rsp[np.newaxis, :] * np.diff(g1d)[0]).ravel()

    psffile = os.path.join(psfpath, '%s.npz' % shape)
    npz = np.load(psffile, allow_pickle=True)
    refpsf = npz['refpsf'].item()[M]

    rcafile = os.path.join(rcapath, '%s_d%03d_n%7.1e.npz' %
                           (shape, int(pps), noise))
    npz = np.load(rcafile, allow_pickle=True)
    rcapsf = np.pad(npz['psf'], (2, 2))
    rcapsfi = interpolate(rcapsf, M, K)
    rcapsfi = translatescale(refpsf, rcapsfi)

    cdlfile = os.path.join(cdlpath, '%s_d%03d_n%7.1e.npz' %
                           (shape, int(pps), noise))
    npz = np.load(cdlfile, allow_pickle=True)
    cdlpsf = npz['psfgrd']
    cdlpsfi = interpolate(cdlpsf, M, K)
    cdlpsfi = translatescale(refpsf, cdlpsfi)

    return grd, refpsf, rcapsfi, cdlpsfi



def plot_psf_sections(ref, rca, cdl, grd, title=None, maxcnt=True):

    if maxcnt:
        gc, gr = np.unravel_index(ref.argmax(), ref.shape)
    else:
        gc = ref.shape[0] // 2
        gr = ref.shape[1] // 2
    fig, ax = plot.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                            figsize=(16, 5))
    if title is not None:
        fig.suptitle(title, fontsize=14)
    plot.plot(ref[gc], grd, c=clrs[0], lw=2, alpha=0.75, fig=fig, ax=ax[0])
    plot.plot(rca[gc], grd, c=clrs[1], lw=2, alpha=0.75, fig=fig, ax=ax[0])
    plot.plot(cdl[gc], grd, c=clrs[2], lw=2, alpha=0.75, title='Row slice',
              lgnd=('Reference', 'RCA', 'CDL'), fig=fig, ax=ax[0])
    plot.plot(ref[:, gr], grd, c=clrs[0], lw=2, alpha=0.75, fig=fig, ax=ax[1])
    plot.plot(rca[:, gr], grd, c=clrs[1], lw=2, alpha=0.75, fig=fig, ax=ax[1])
    plot.plot(cdl[:, gr], grd, c=clrs[2], lw=2, alpha=0.75,
              title='Column slice', lgnd=('Reference', 'RCA', 'CDL'),
              fig=fig, ax=ax[1])
    fig.show()
    return fig, ax



def plot_psf_section_diffs(ref, rca, cdl, grd, title=None, maxcnt=True):

    if maxcnt:
        gc, gr = np.unravel_index(ref.argmax(), ref.shape)
    else:
        gc = ref.shape[0] // 2
        gr = ref.shape[1] // 2
    fig, ax = plot.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                            figsize=(16, 5))
    if title is not None:
        fig.suptitle(title, fontsize=14)
    plot.plot(rca[gc] - ref[gc], grd, c=clrs[1], lw=2, alpha=0.75,
              fig=fig, ax=ax[0])
    plot.plot(cdl[gc] - ref[gc], grd, c=clrs[2], lw=2, alpha=0.75,
              title='Row slice', lgnd=('RCA - Ref.', 'CDL - Ref.'),
              fig=fig, ax=ax[0])
    plot.plot(rca[:, gr] - ref[:, gr], grd, c=clrs[1], lw=2, alpha=0.75,
              fig=fig, ax=ax[1])
    plot.plot(cdl[:, gr] - ref[:, gr], grd, c=clrs[2], lw=2, alpha=0.75,
              title='Column slice', lgnd=('RCA - Ref.', 'CDL - Ref.'),
              fig=fig, ax=ax[1])
    fig.show()
    return fig, ax




def plot_psf_contours(ref, rca, cdl, grd, v=5, xrng=None, yrng=None,
                      title=None):

    fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(18.15, 5))
    if title is not None:
        fig.suptitle(title, fontsize=14)
    plot.contour(ref, grd, grd, v=v, title='Reference',
                 fig=fig, ax=ax[0])
    plot.contour(rca, grd, grd, v=v, title='RCA',
                 fig=fig, ax=ax[1])
    plot.contour(cdl, grd, grd, v=v, title='CDL',
                 fig=fig, ax=ax[2])
    if xrng is not None or yrng is not None:
        for x in ax:
            if xrng is not None:
                x.set_xlim(xrng)
            if yrng is not None:
                x.set_ylim(yrng)
    fig.show()
    return fig, ax


# Subpixel estimation factor (common for all runs)
M = 5

# Define standard integer sampling grid -wp ... wp
wp = 7

# Paths to data files
psfpath = 'data/reference_psfs'
rcapath = 'data/rca_results'
cdlpath = 'data/icdl_results'


noise = 1.0
pps = 1.0
shape = 'complex'

grd, refpsf, rcapsf, cdlpsf = get_psf_arrays(
    noise, pps, shape, M, wp, psfpath, rcapath, cdlpath)

# The reference complex PSF is different scaling from the other PSFs:
# rescale for plotting
rmax = refpsf.max()
refpsf /= rmax
rcapsf /= rmax
cdlpsf /= rmax

fig, ax = plot_psf_sections(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('complex_d1_n1_section.pdf', bbox_inches='tight')

fig, ax = plot_psf_section_diffs(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('complex_d1_n1_secdiff.pdf', bbox_inches='tight')

fig, ax = plot_psf_contours(refpsf, rcapsf, cdlpsf, grd,
                            v=(0.05, 0.2, 0.4, 0.6, 0.8),
                            xrng=(-5, 4), yrng=(-5, 4))
fig.savefig('complex_d1_n1_contour.pdf', bbox_inches='tight')



noise = 1.0
pps = 1.0
shape = 'elong'

grd, refpsf, rcapsf, cdlpsf = get_psf_arrays(
    noise, pps, shape, M, wp, psfpath, rcapath, cdlpath)

fig, ax = plot_psf_sections(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('elong_d1_n1_section.pdf', bbox_inches='tight')

fig, ax = plot_psf_section_diffs(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('elong_d1_n1_secdiff.pdf', bbox_inches='tight')

fig, ax = plot_psf_contours(refpsf, rcapsf, cdlpsf, grd,
                            v=(0.05, 0.2, 0.4, 0.6, 0.8),
                            xrng=(-4, 4), yrng=(-4, 4))
fig.savefig('elong_d1_n1_contour.pdf', bbox_inches='tight')


noise = 1.0
pps = 1.0
shape = 'narrow'

grd, refpsf, rcapsf, cdlpsf = get_psf_arrays(
    noise, pps, shape, M, wp, psfpath, rcapath, cdlpath)

fig, ax = plot_psf_sections(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('narrow_d1_n1_section.pdf', bbox_inches='tight')

fig, ax = plot_psf_section_diffs(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('narrow_d1_n1_secdiff.pdf', bbox_inches='tight')

fig, ax = plot_psf_contours(refpsf, rcapsf, cdlpsf, grd,
                            v=(0.05, 0.2, 0.4, 0.6, 0.8),
                            xrng=(-4, 4), yrng=(-4, 4))
fig.savefig('narrow_d1_n1_contour.pdf', bbox_inches='tight')

noise = 1.0
pps = 1.0
shape = 'wide'

grd, refpsf, rcapsf, cdlpsf = get_psf_arrays(
    noise, pps, shape, M, wp, psfpath, rcapath, cdlpath)

fig, ax = plot_psf_sections(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('wide_d1_n1_section.pdf', bbox_inches='tight')

fig, ax = plot_psf_section_diffs(refpsf, rcapsf, cdlpsf, grd)
fig.savefig('wide_d1_n1_secdiff.pdf', bbox_inches='tight')

fig, ax = plot_psf_contours(refpsf, rcapsf, cdlpsf, grd,
                            v=(0.05, 0.2, 0.4, 0.6, 0.8))
fig.savefig('wide_d1_n1_contour.pdf', bbox_inches='tight')



input()
