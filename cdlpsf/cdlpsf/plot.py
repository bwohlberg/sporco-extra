# -*- coding: utf-8 -*-
# Author: Brendt Wohlberg <brendt@ieee.org>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

"""Plotting functions."""

import numpy as np

from sporco import plot


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def plot_solver_stats(pe):
    """Plot iteration statistics for a :class:`.cdl.PSFEstimator` solve.

    Parameters
    ----------
    pe : :class:`.cdl.PSFEstimator`
        Solver object for which solver iteration stats are to be plotted

    Returns
    -------
    :class:`matplotlib.figure.Figure` object
        Figure object for this figure
    """

    itsx = pe.slvX.getitstat()
    itsg = pe.slvG.getitstat()
    fig = plot.figure(figsize=(20, 5))
    fig.suptitle('Solver iteration statistics')
    plot.subplot(1, 3, 1)
    N = min(len(itsx.ObjFun), len(itsg.DFid))
    fnvl = np.vstack((itsx.ObjFun[-N:], itsg.ObjFun[-N:],
                      itsx.DFid[-N:], itsg.DFid[-N:])).T
    lgnd = ('X Func.', 'G Func,', 'X D. Fid.', 'G D. Fid.')
    plot.plot(fnvl, ptyp='semilogy', xlbl='Iterations', ylbl='Functional',
              lgnd=lgnd, fig=fig)
    plot.subplot(1, 3, 2)
    plot.plot(np.vstack((itsx.PrimalRsdl, itsx.DualRsdl)).T,
              ptyp='semilogy', xlbl='Iterations', ylbl='X Residual',
              lgnd=['Primal', 'Dual'], fig=fig)
    plot.subplot(1, 3, 3)
    if hasattr(itsg, 'PrimalRsdl'):
        plot.plot(np.vstack((itsg.PrimalRsdl, itsg.DualRsdl)).T,
                  ptyp='semilogy', xlbl='Iterations', ylbl='G Residual',
                  lgnd=['Primal', 'Dual'], fig=fig)
    else:
        plot.plot(itsg.Rsdl, ptyp='semilogy', xlbl='Iterations',
                  ylbl='G Residual', fig=fig)
    fig.show()
    return fig



def plot_psf_sections(ref, est, grd, title='PSF Comparison', maxcnt=True):
    """Plot comparison of PSF cross-sections.

    Parameters
    ----------
    ref : ndarray
        Reference PSF
    est : ndarray
        Estimated PSF
    grd : ndarray
        Sampling grid of PSF
    title : str, optional
        Plot title
    maxcnt : bool, optional
        If True, take cross-sections at maximum value of reference PSF,
        otherwise take them at middle of PSF arrays

    Returns
    -------
    :class:`matplotlib.figure.Figure` object
        Figure object for this figure
    list of :class:`matplotlib.axes.Axes` object
        List of axes objects for this plot
    """

    if maxcnt:
        gc, gr = np.unravel_index(ref.argmax(), ref.shape)
    else:
        gc = ref.shape[0] // 2
        gr = ref.shape[1] // 2
    fig, ax = plot.subplots(nrows=1, ncols=2, sharex=True, sharey=True,
                            figsize=(20, 8))
    fig.suptitle(title, fontsize=14)
    plot.plot(ref[gc], grd, fig=fig, ax=ax[0])
    plot.plot(est[gc], grd, title='Row slice',
              lgnd=('Reference', 'Estimate'), fig=fig, ax=ax[0])
    plot.plot(ref[:, gr], grd, fig=fig, ax=ax[1])
    plot.plot(est[:, gr], grd, title='Column slice',
              lgnd=('Reference', 'Estimate'), fig=fig, ax=ax[1])
    fig.show()
    return fig, ax



def plot_psf_contours(ref, est, grd, title='PSF Comparison'):
    """Plot PSF contour comparison.

    Parameters
    ----------
    ref : ndarray
        Reference PSF
    est : ndarray
        Estimated PSF
    grd : ndarray
        Sampling grid of PSF
    title : str, optional
        Plot title

    Returns
    -------
    :class:`matplotlib.figure.Figure` object
        Figure object for this figure
    list of :class:`matplotlib.axes.Axes` object
        List of axes objects for this plot
    """

    fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12.1, 5))
    fig.suptitle(title, fontsize=14)
    plot.contour(ref, grd, grd, title='Reference',
                 fig=fig, ax=ax[0])
    plot.contour(est, grd, grd, title='Estimate',
                 fig=fig, ax=ax[1])
    fig.show()
    return fig, ax
