# -*- coding: utf-8 -*-
# Author: Brendt Wohlberg <brendt@ieee.org>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license


"""Utility functions."""


import numpy as np
import scipy.signal
import scipy.ndimage

from sporco.metric import snr
from sporco.fft import fftconv
from sporco.interp import lanczos_filters, interpolation_points


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


def subpixel_grid(pixgrd, N):
    """Construct subpixel version of `pixgrd` subsampled `N` times.

    Parameters
    ----------
    pixgrd : ndarray
        Sampling grid
    N : int
        Subsampling factor

    Returns
    -------
    TYPE
        Description
    """

    rsp = interpolation_points(N)
    rspgrd = (pixgrd[:, np.newaxis] + rsp[np.newaxis, :] *
              np.diff(pixgrd)[0])
    return rspgrd.ravel()



def translatescale(x, y):
    """Translate and scale `y` to maximise correlation with `x`.

    Parameters
    ----------
    x : ndarray
        Reference array
    y : ndarray
        Array to be translated and scaled

    Returns
    -------
    ndarray
        Translated and scaled version of `y`
    """

    c = scipy.signal.correlate(y, x, mode='same')
    cmi = np.unravel_index(c.argmax(), c.shape)
    dlt = ((np.array(y.shape) - 1) / 2 - cmi).astype(np.int)

    yr = np.roll(y, shift=tuple(dlt.astype(np.int)), axis=(0, 1))
    yr /= np.sum(yr * x) / np.sum(x * x)

    return yr


def ts_snr(x, y):
    """Maximum SNR attainable via translation and scaling.

    Compute the SNR between `x` and the output of
    :func:`translatescale`(`x`, `y`).

    Parameters
    ----------
    x : ndarray
        Reference array
    y : ndarray
        Comparison array (to be translated and scaled)

    Returns
    -------
    float
        SNR value in dB
    """

    yts = translatescale(x, y)
    return snr(x, yts)



def subsamplematch(x, y):
    """Subsample and scale `x` to to maximize the correlation with `y`.

    Find the subsampling offset in input `x` that maximises the
    correlation with input `y` that is sampled at a lower resolution.
    The output of this function is the best subsampling of `x` after
    scaling to further maximise the correlation with `y`.

    Parameters
    ----------
    x : ndarray
        High resolution signal
    y : ndarray
        Low resolutin signal

    Returns
    -------
    ndarray
        Scaled and subsampled version of `x`
    """

    # Subsampling factor in each axis
    n = tuple(np.array(x.shape) // np.array(y.shape))
    # Input y upsampled to resolution of x
    z = np.zeros(x.shape)
    z[::n[0], ::n[1]] = y
    # Find best translation of x to maximise correlation with z
    c = scipy.signal.correlate(x, z, mode='same')
    cmi = np.unravel_index(c.argmax(), c.shape)
    dlt = ((np.array(x.shape) - 1) / 2 - cmi).astype(np.int)
    # Translate x to selected position
    xr = np.roll(x, shift=tuple(dlt.astype(np.int)), axis=(0, 1))
    # Subsample to match y resolution
    xr = xr[::n[0], ::n[1]]
    # Rescale to maximise correlation
    xr /= np.sum(xr * y) / np.sum(y * y)

    return xr


def sm_snr(x, y):
    """Maximum SNR attainable via subsampling and scaling.

    Compute the SNR between the output of :func:`subsamplematch`(`x`, `y`)
    and `y`.

    Parameters
    ----------
    x : ndarray
        Reference array (to be subsampled and scaled)
    y : ndarray
        Comparison array

    Returns
    -------
    float
        SNR value in dB
    """

    xsm = subsamplematch(x, y)
    return snr(xsm, y)



def interpolate(x, M=5, K=10):
    """Lanczos interpolation of 1D or 2D array.

    Parameters
    ----------
    x : ndarray
        Array to be interpolated
    M : int, optional
        Interpolation factor
    K : int, optional
        Order of Lanczos filters

    Returns
    -------
    ndarray
        Input `x` interpolated to higher resolution
    """

    if x.ndim == 1:
        Hl = lanczos_filters((M,), K, collapse_axes=False)
        xp = np.pad(x, ((0, Hl.shape[0] - 1),), 'constant')
        xrsp = fftconv(Hl, xp[..., np.newaxis], axes=(0,),
                       origin=(K,))[0:x.shape[0],]
        shp = tuple(np.array(x.shape) * M)
        xsub = xrsp.reshape(shp)
    else:
        Hl = lanczos_filters((M, M), K, collapse_axes=False)
        xp = np.pad(x, ((0, Hl.shape[0] - 1), (0, Hl.shape[1] - 1)),
                    'constant')
        xrsp = fftconv(Hl, xp[..., np.newaxis, np.newaxis], axes=(0, 1),
                       origin=(K, K))[0:x.shape[0], 0:x.shape[1]]
        shp = tuple(np.array(x.shape) * M)
        xsub = np.transpose(xrsp, (0, 2, 1, 3)).reshape(shp)
    return xsub
