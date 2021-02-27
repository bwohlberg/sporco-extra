# -*- coding: utf-8 -*-
# Author: Brendt Wohlberg <brendt@ieee.org>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

"""Classes for psf estimation via convolutional sparse representations."""

import copy

import numpy as np

from sporco import cdict
from sporco.fft import fftconv
from sporco.signal import tikhonov_filter
from sporco.interp import lanczos_filters, interpolation_points

from cdlpsf.csc import CSCl1l2
from cdlpsf.ccmod import CnsGrdRegDeconvPGM


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class PSFEstimator(object):
    """PSF estimation via interpolated convolutional dictionary
    learning."""

    class Options(cdict.ConstrainedDict):
        """PSF estimation algorithm options.

        Options:

          ``Verbose`` : Flag determining whether iteration status is
          displayed.

          ``MaxMainIter`` : Maximum main iterations.

          ``Callback`` : Callback function to be called at the end of
          every iteration.
        """

        defaults = {'Verbose': False, 'MaxMainIter': 100,
                    'XslvIter0': 1, 'GslvIter0': 1,
                    'XslvIter': 1, 'GslvIter': 1,
                    'Callback': None}
        defaults.update(
            {'Xslv': copy.deepcopy(CSCl1l2.Options.defaults),
             'Gslv': copy.deepcopy(CnsGrdRegDeconvPGM.Options.defaults)})


        def __init__(self, opt=None):
            """
            Parameters
            ----------
            opt : dict or None, optional (default None)
              Algorithm options
            """

            cdict.ConstrainedDict.__init__(self,
                {'Xslv':  CSCl1l2.Options(
                    {'Verbose': False, 'RelStopTol': 1e-4,
                     'NonNegCoef': True, 'DatFidNoDC': True,
                     'rho': 1e1, 'AutoRho': {'Enabled': False}}),
                 'Gslv': CnsGrdRegDeconvPGM.Options(
                     {'Verbose': False, 'NonNegCoef': True,
                      'DatFidNoDC': True, 'L': 1e2, 'RelStopTol': 1e-4})
                }
            )
            if opt is None:
                opt = {}
            self.update(opt)



    def __init__(self, img, g0, lmbdaX, lmbdaG, M=5, K=5, opt=None):
        """
        Parameters
        ----------
        img : array_like
          Image for which PSF is estimated
        g0 : array_like
          Initial PSF estimate
        lmbdaX : float
          Regularisation parameter for sparse coding stage
        lmbdaG : float
          Regularisation parameter for dictionary update stae
        M : int
          Number of interpolation points on each axis
        K : int
          Lanczos kernel oder parameter
        opt : :class:`PSFEstimator.Options` object
          Algorithm options
        """

        self.img = img

        self.opt = opt
        self.M = M
        self.K = K

        self.g = g0.copy()
        self.g /= np.linalg.norm(self.g)
        self.gshp = self.g.shape

        self.lmbdaX = lmbdaX
        self.lmbdaG = lmbdaG

        H = lanczos_filters((M, M), a=K)
        H = np.pad(H, ((0, img.shape[0] - H.shape[0]),
                       (0, img.shape[1] - H.shape[1]), (0, 0)),
                   'constant')
        self.H = np.roll(H, (-K, -K), (0, 1))

        self.D, self.dn = self.getD(self.g)

        optX = opt['Xslv']
        optX['MaxMainIter'] = opt['XslvIter0']
        self.slvX = CSCl1l2(self.D, img, self.lmbdaX, opt=optX)

        Wsc = np.zeros(img.shape)
        Wsc[0:self.gshp[0], 0:self.gshp[1]] = 1.0
        optG = opt['Gslv']
        optG['MaxMainIter'] = opt['GslvIter0']
        optG['SupportMask'] = Wsc
        optG['Normalize'] = True
        optG['X0'] = np.zeros(img.shape)
        optG['X0'][0:self.gshp[0], 0:self.gshp[1]] = self.g
        Xhs = np.zeros(img.shape)
        self.slvG = CnsGrdRegDeconvPGM(Xhs, img, lmbdaG, opt=optG)


    def solve(self):
        """Start (or re-start) optimisation."""

        # Set up display header if verbose operation enabled
        if self.opt['Verbose']:
            hdr = 'Itn  DFidX     PriResX   DuaResX    DFidG' + \
                  '     ResG     '
            print(hdr)
            print('-' * len(hdr))

        # Main iteration loop
        for n in range(self.opt['MaxMainIter']):

            # At start of 2nd iteration, set the numbers of inner
            # iterations for the X and G solvers from the options
            # object for the outer solver
            if n == 1:
                self.slvX.opt['MaxMainIter'] = self.opt['XslvIter']
                self.slvG.opt['MaxMainIter'] = self.opt['GslvIter']

            # Run the configured number of iterations of the X (CSC)
            # solver and assign the result to X
            self.X = self.slvX.solve()

            # Compute the sum of the subpixel shifts of X
            Xhs = np.sum(fftconv(self.H, self.X.squeeze(), axes=(0, 1)),
                         axis=-1)

            # Set the convolution kernel in the deconvolution solver
            # to the sum of the subpixel shifts of X
            self.slvG.setG(Xhs)
            # Run the configured number of iterations of the G
            # (deconvolution) solver and crop the result to obtain the
            # updated g
            self.g = self.slvG.solve()[0:self.gshp[0], 0:self.gshp[1]]

            # Construct a new dictionary for the X (CSC) solver from
            # the updated psf g
            self.D, self.dn = self.getD(self.g)
            self.slvX.setdict(self.D[..., np.newaxis, np.newaxis, :])

            # Display iteration statistics if verbose operation enabled
            if self.opt['Verbose']:
                itsX = self.slvX.getitstat()
                itsG = self.slvG.getitstat()
                fmt = '%3d  %.3e %.3e %.3e  %.3e %.3e'
                tpl = (n, itsX.DFid[-1], itsX.PrimalRsdl[-1],
                       itsX.DualRsdl[-1], itsG.DFid[-1], itsG.Rsdl[-1])
                print(fmt % tpl)

        # Return the (normalised) psf estimate g
        return self.g / np.linalg.norm(self.g)



    def getD(self, g):
        """Construct the CSC dictionary corresponding to psf `g`."""

        # Zero pad g to avoid boundary effects
        d = np.pad(g, ((0, self.img.shape[0] - g.shape[0]),
                       (0, self.img.shape[1] - g.shape[1])), 'constant')
        # Convolve padded g with set of interpolation filters to
        # construct a set of subpixel shifted versions of g
        D = fftconv(d[..., np.newaxis], self.H, axes=(0, 1))
        # Get dictionary filter norms
        dn = np.sqrt(np.sum(D**2, axis=(0, 1), keepdims=True))

        return D, dn



    def get_psf(self, subpixel=True, tkhflt=False, tkhlmb=1e-3):
        """Get the estimated psf. If parameter `subpixel` is True, the
        subpixel resolution psf is returned, constructed by interpolation
        of the psf estimated at the resolution of the input image.
        """

        if subpixel:
            Hl = lanczos_filters((self.M, self.M), self.K,
                                 collapse_axes=False)
            gp = np.pad(self.g, ((0, self.img.shape[0] - self.gshp[0]),
                                 (0, self.img.shape[1] - self.gshp[1])),
                        'constant')
            grsp = fftconv(Hl, gp[..., np.newaxis, np.newaxis],
                           origin=(self.K, self.K),
                           axes=(0, 1),
                           )[0:self.gshp[0], 0:self.gshp[1]]
            shp = tuple(np.array(self.gshp) * self.M)
            gsub = np.transpose(grsp, (0, 2, 1, 3)).reshape(shp)
            gsub[gsub < 0.0] = 0.0
            if tkhflt:
                gsub, shp = tikhonov_filter(gsub, tkhlmb)
            gsub /= np.linalg.norm(gsub)
            return gsub
        else:
            return self.g / np.linalg.norm(self.g)



    def get_subpix_grid(self, g1d):
        """Get subpixel grid on which the subpixel psf estimate from
        get_psf is defined. Parameter `g1d` is the sampling grid for
        the reference psf at the resolution of the input image.
        """

        rsp = interpolation_points(self.M)
        rspg1d = g1d[:, np.newaxis] + rsp[np.newaxis, :] * np.diff(g1d)[0]
        return rspg1d.ravel()
