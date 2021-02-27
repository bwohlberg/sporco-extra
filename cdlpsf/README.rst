ICDL PSF Estimation
===================
:Author: Brendt Wohlberg <brendt@ieee.org>

This directory contains the scripts used to generate the results in the paper `PSF Estimation in Crowded Astronomical Imagery as a Convolutional Dictionary Learning Problem <https://arxiv.org/abs/2101.01268>`_ (doi:`10.1109/LSP.2021.3050706 <http://dx.doi.org/10.1109/LSP.2021.3050706>`_). The proposed method, referred to here as `Interpolated Convolutional Dictionary Learning (ICDL)`, is compared with `Resolved Component Analysis (RCA) <https://github.com/CosmoStat/rca>`_.

The implementation of the `ICDL` method currently requires the development version of the `SPORCO <https://github.com/bwohlberg/sporco>`_ package, which can be installed by

::

    pip install git+https://github.com/bwohlberg/sporco

Once SPORCO version 0.1.13 has been released, it can be installed by

::

   pip install sporco

or, within a `conda <https://docs.conda.io/en/latest/miniconda.html>`_ environment

::

   conda install sporco

See the file `INSTALL_RCA.rst <INSTALL_RCA.rst>`_ for instructions on installing `RCA <https://github.com/CosmoStat/rca>`_ and its dependencies.



RCA Results
-----------

`compute_rca_opt_param.py <compute_rca_opt_param.py>`_
   Find optimal parameters for each test case by optimizing over 9000 different parameter combinations. The RCA scripts below depend on the results of this script, but pre-computed results are included in this repository, so it is not essential to re-run this script.

`tabulate_rca_opt_param.py <tabulate_rca_opt_param.py>`_
   Tabulate the results computed by `compute_rca_opt_param.py <compute_rca_opt_param.py>`_.

`compute_rca_results.py <compute_rca_results.py>`_
   Compute RCA solutions for each test case using optimal parameters found by `compute_rca_opt_param.py <compute_rca_opt_param.py>`_.

`tabulate_rca_results.py <tabulate_rca_results.py>`_
   Tabulate the results computed by `compute_rca_results.py <compute_rca_results.py>`_.


ICDL Results
------------

`compute_icdl_results.py <compute_icdl_results.py>`_
   Compute ICDL solutions using parameter selection heuristics.

`tabulate_icdl_results.py <tabulate_icdl_results.py>`_
   Tabulate the results computed by `compute_icdl_results.py <compute_icdl_results.py>`_.

`compute_icdl_opt_param.py <compute_icdl_opt_param.py>`_
   Find optimal parameters for each test case by optimizing over 768 different parameter combinations. These results are not included in the above-mentioned paper. Pre-computed results are included in this repository, so it is not essential to re-run this script before running `tabulate_icdl_opt_param.py <tabulate_icdl_opt_param.py>`_.

`tabulate_icdl_opt_param.py <tabulate_icdl_opt_param.py>`_
  Tabulate the results computed by `compute_icdl_opt_param.py <compute_icdl_opt_param.py>`_.


Comparison
----------

`make_psf_plots.py <make_psf_plots.py>`_
   Construct PSF estimation performance comparisons included in the extended version of the paper
