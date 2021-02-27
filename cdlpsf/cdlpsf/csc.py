# -*- coding: utf-8 -*-
# Author: Brendt Wohlberg <brendt@ieee.org>
# Please see the 'README.rst' and 'LICENSE' files in the SPORCO Extra
# repository for details of the copyright and user license

"""Classes for solving CSC problems."""

import copy

import numpy as np

from sporco.fft import rfftn, rfl2norm2
from sporco.linalg import inner
import sporco.linalg as sl
from sporco.admm import cbpdn
from sporco.util import u
from sporco.prox import norm_dl1l2, prox_dl1l2


__author__ = """Brendt Wohlberg <brendt@ieee.org>"""


class CSCl1l2(cbpdn.ConvBPDN):
    r"""Convolutional sparse coding with difference of :math:`\ell_1`
    and :math:`\ell_2` norm regularization.
    """

    class Options(cbpdn.ConvBPDN.Options):
        r"""CSCl1l2 algorithm options

        Options include all of those defined in
        :class:`.cbpdn.ConvBPDN.Options`, together with additional options:

          ``DatFidNoDC`` : Flag indicating whether the frequency domain
          weighting should be applied so that the value of the data
          fidelity term is independent of the DC offset of the signal.

          ``beta`` : Scaling factor in the regularization term
          :math:`\|\mathbf{x}\|_1 - \beta \|\mathbf{x}\|_2`. The default
          value is 1.0.
        """

        defaults = copy.deepcopy(cbpdn.ConvBPDN.Options.defaults)
        defaults.update({'DatFidNoDC': False, 'beta': 1.0})


    itstat_fields_objfn = ('ObjFun', 'DFid', 'RegL1L2')
    hdrtxt_objfn = ('Fnc', 'DFid', u('Regℓ1-ℓ2'))
    hdrval_objfun = {'Fnc': 'ObjFun', 'DFid': 'DFid',
                     u('Regℓ1-ℓ2'): 'RegL1L2'}


    def __init__(self, D, S, lmbda, opt=None, dimN=2):
        """
        Initalize the CSC solver object.

        Parameters
        ----------
        D : array_like
          Dictionary array
        S : array_like
          Signal array
        opt : :class:`CSCl1l2.Options` object
          Algorithm options
        dimN : int, optional (default 2)
          Number of spatial/temporal dimensions
        """

        self.beta = opt['beta']
        super(CSCl1l2, self).__init__(D, S, lmbda=lmbda, opt=opt, dimN=dimN)



    def setdict(self, D=None):
        """Set dictionary array."""

        if D is not None:
            self.D = np.asarray(D, dtype=self.dtype)
        self.Df = rfftn(self.D, self.cri.Nv, self.cri.axisN)

        if self.opt['DatFidNoDC']:
            if self.cri.dimN == 1:
                self.Df[0] = 0.0
            else:
                self.Df[0, 0] = 0.0

        self.DSf = np.conj(self.Df) * self.Sf
        if self.cri.Cd > 1:  # NB: Not tested
            self.DSf = np.sum(self.DSf, axis=self.cri.axisC, keepdims=True)
        if self.opt['HighMemSolve'] and self.cri.Cd == 1:
            self.c = sl.solvedbi_sm_c(self.Df, np.conj(self.Df), self.rho,
                                      self.cri.axisM)
        else:
            self.c = None



    def setS(self, S):
        """Set signal array."""

        self.S = np.asarray(S.reshape(self.cri.shpS), dtype=self.dtype)
        self.Sf = rfftn(self.S, None, self.cri.axisN)



    def ystep(self):
        r"""Minimise Augmented Lagrangian with respect to
        :math:`\mathbf{y}`.
        """

        self.Y = np.asarray(prox_dl1l2(self.AX + self.U,
                            (self.lmbda / self.rho) * self.wl1, self.beta),
                            dtype=self.dtype)
        cbpdn.GenericConvBPDN.ystep(self)



    def obfn_dfd(self):
        """Compute data fidelity term."""

        if self.opt['DatFidNoDC']:
            Sf = self.Sf.copy()
            if self.cri.dimN == 1:
                Sf[0] = 0
            else:
                Sf[0, 0] = 0
        else:
            Sf = self.Sf
        Ef = inner(self.Df, self.obfn_fvarf(), axis=self.cri.axisM) - Sf
        return rfl2norm2(Ef, self.S.shape, axis=self.cri.axisN) / 2.0



    def obfn_reg(self):
        """Compute regularisation term."""

        rl1 = norm_dl1l2((self.wl1 * self.obfn_gvar()).ravel(), self.beta)
        return (self.lmbda*rl1, rl1)
