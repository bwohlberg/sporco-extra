# -*- coding: utf-8 -*-
# Author: Brendt Wohlberg <brendt@ieee.org>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

"""Classes for solving CCMOD-like problems."""

import copy

import numpy as np

from sporco.fft import rfftn, irfftn, rfftn_empty_aligned, rfl2norm2
from sporco.signal import gradient_filters
from sporco.pgm import pgm


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class CnsGrdRegDeconvPGM(pgm.PGMDFT):
    r"""
    Solve the problem

    .. math::
       \mathrm{argmin}_\mathbf{x} \;
       (1/2) \left\| \mathbf{g} * \mathbf{x} - \mathbf{s} \right\|_2^2 +
         (\lambda / 2) \| D_r \mathbf{x} \| +
         (\lambda / 2) \| D_r \mathbf{x} \|

    such that :math:`\mathbf{x} >= 0` and :math:`\mathbf{x}` has a
    constrained support, where :math:`\mathbf{g}` is a convolution
    kernel, and :math:`D_r` and :math:`D_c` are operators computing
    gradients along image rows and columns respectively.

    The solution is computed via the Proximal Gradient Method.
    """

    # Define iterations statistics fields
    itstat_fields_objfn = ('ObjFun', 'DFid', 'Reg', 'Cnstr')
    hdrtxt_objfn = ('Fnc', 'DFid', 'Reg', 'Cnstr')
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid', 'Reg': 'Reg',
                     'Cnstr': 'Cnstr'}


    class Options(pgm.PGMDFT.Options):
        r"""CnsGrdRegDeconvPGM algorithm options

        Options include all of those defined in
        :class:`.pgm.PGMDFT.Options`, together with
        additional options:

          ``SupportMask`` : A mask array defining the support constraint
          on the solution.
          ``Normalize`` : Flag indicating whether the solution should be
          normalized.
          ``DatFidNoDC`` : Flag indicating whether the frequency domain
          weighting should be applied so that the value of the data
          fidelity term is independent of the DC offset of the signal.
          ``NonNegCoef`` : Flag indicating whether the solution should
          be constrained to be non-negative.
        """

        defaults = copy.deepcopy(pgm.PGMDFT.Options.defaults)
        defaults.update({'SupportMask': None, 'Normalize': False,
                         'DatFidNoDC': False, 'NonNegCoef': False})


    def __init__(self, G, S, lmbda, opt=None, dimN=2):
        """
        Initalize the solver object.

        Parameters
        ----------
        G : array_like
          Convolution kernel array
        S:  array_like
          Signal array
        lmbda: float
           Regularization parameter
        opt: :class:`CnsGrdRegDeconvPGM.Options` object
           Algorithm options
        """

        if opt is None:
            opt = CnsGrdRegDeconvPGM.Options()

        self.dimN = dimN
        self.axes = tuple(range(dimN))

        self.set_dtype(opt, S.dtype)
        self.lmbda = self.dtype.type(lmbda)
        self.set_attr('L', opt['L'], dval=1e1, dtype=self.dtype)

        super(CnsGrdRegDeconvPGM, self).__init__(
            xshape=S.shape, Nv=S.shape, axisN=tuple(range(dimN)),
            dtype=S.dtype, opt=opt)

        self.setS(S)
        self.setG(G)
        self.Df, self.DHDf = gradient_filters(dimN, self.axes, S.shape,
                                              dtype=self.dtype)

        self.Wsm = opt['SupportMask']
        self.nrmflg = opt['Normalize']

        self.Y = self.X
        self.X[:] = self.Y
        self.Vf = rfftn_empty_aligned(self.X.shape, self.axes, self.dtype)
        self.Xf = rfftn(self.X, None, self.axes)
        self.Yf = self.Xf
        self.Yprv = self.Y.copy()
        self.Yfprv = self.Yf.copy() + 1e5



    def setG(self, G):
        """Set the convolution kernel."""

        self.G = G.astype(self.dtype)
        self.Gf = rfftn(self.G)
        if self.opt['DatFidNoDC']:
            if G.ndim == 1:
                self.Gf[0] = 0.0
            else:
                self.Gf[0, 0] = 0.0
        self.GHSf = np.conj(self.Gf) * self.Sf
        self.GHGf = np.conj(self.Gf) * self.Gf



    def setS(self, S):
        """Set the signal."""

        self.S = np.asarray(S, dtype=self.dtype)
        self.Sf = rfftn(self.S)
        if self.opt['DatFidNoDC']:
            if S.ndim == 1:
                self.Sf[0] = 0.0
            else:
                self.Sf[0, 0] = 0.0



    def proximal_step(self, gradf=None):
        """Compute proximal update (gradient descent + constraint).
        Variables are mapped back and forth between input and
        frequency domains.
        """

        if gradf is None:
            gradf = self.eval_grad()

        self.Vf[:] = self.Yf - (1. / self.L) * gradf
        V = irfftn(self.Vf, None, self.axes)

        self.X[:] = self.eval_proxop(V)
        self.Xf = rfftn(self.X, None, self.axes)

        return gradf



    def grad_f(self):
        """Compute gradient in Fourier domain."""

        gradf = (self.GHGf + self.lmbda * self.DHDf) * self.Yf - self.GHSf
        return gradf



    def prox_g(self, V):
        """Compute proximal operator of :math:`g`."""

        return self.cnstr_proj(V)



    def cnstr_proj(self, X):
        """Projection onto constraint set."""

        if self.opt['NonNegCoef']:
            Y = np.clip(X, 0, None)
        else:
            Y = X.copy()
        if self.Wsm is not None:
            Y[self.Wsm == 0.0] = 0.0
        if self.nrmflg:
            yn = np.linalg.norm(Y)
            if yn > 0:
                Y /= yn
        return Y



    def eval_objfn(self):
        """Compute components of objective function as well as total
        contribution to objective function.
        """

        Xf = self.Xf
        Ef = self.Gf * Xf - self.Sf
        dfd = np.sum((irfftn(Ef, self.S.shape, axes=self.axes))**2) / 2.0
        reg = np.sum(irfftn(self.Df * Xf[..., np.newaxis],
                            self.S.shape, axes=self.axes)**2)
        obj = dfd + self.lmbda * reg
        cns = np.linalg.norm(self.X - self.cnstr_proj(self.X))
        return (obj, dfd, reg, cns)



    def obfn_f(self, Xf=None):
        r"""Compute data fidelity term :math:`(1/2) \| \sum_m
        \mathbf{d}_m * \mathbf{x}_m - \mathbf{s} \|_2^2`.
        This is used for backtracking. Since the backtracking is
        computed in the DFT, it is important to preserve the
        DFT scaling.
        """
        if Xf is None:
            Xf = self.Xf

        Rf = self.Gf * Xf - self.Sf
        return 0.5 * np.linalg.norm(Rf.flatten(), 2)**2



    def rsdl(self):
        """Compute fixed point residual in Fourier domain."""

        diff = self.Xf - self.Yfprv
        return rfl2norm2(diff, self.X.shape, axis=self.axes)
