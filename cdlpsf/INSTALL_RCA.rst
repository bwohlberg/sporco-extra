RCA Installation
================

These instructions are provided since we found `RCA <https://github.com/CosmoStat/rca>`_ to be rather complex to install. (Some of these instructions assume an Ubuntu Linux operating system.) The recommended approach makes use of a cloned `conda <https://docs.conda.io/en/latest/miniconda.html>`_ environment to avoid undesired side-effects, such as downgrading of the SciPy version, on the user's main conda environment.

Assuming the main `conda <https://docs.conda.io/en/latest/miniconda.html>`_ environment is `py38`, create a new environment for `RCA <https://github.com/CosmoStat/rca>`_:

::

  conda create --name py38rca --clone py38
  conda activate py38rca


Install some `RCA <https://github.com/CosmoStat/rca>`_ dependencies

::

  pip install modopt
  conda install pyqtgraph

Build and install `Sparse2D <https://github.com/CosmoStat/Sparse2D>`_, which is required by `PySAP <https://github.com/CEA-COSMIC/pysap>`_:

::

  sudo apt install cmake libcfitsio-dev
  git clone https://github.com/CosmoStat/Sparse2D.git
  cd Sparse2
  mkdir build && cd build
  cmake ..
  make
  make install
  sudo mv -i sparse2d /opt
  export PATH=$PATH:/opt/sparse2d/bin

Install `PySAP <https://github.com/CEA-COSMIC/pysap>`_:

::

  pip install python-pysap

Clone `RCA <https://github.com/CosmoStat/rca>`_ from github and install it:

::

  git clone https://github.com/CosmoStat/rca.git
  cd rca
  python setup.py install


If the shell in which these instructions are executed is closed, the RCA enviroment must be re-initialized by:

::

   export PATH=$PATH:/opt/sparse2d/bin
   conda activate py38rca

It is recommended that the command above that adds ``/opt/sparse2d/bin`` to the path should be added to the user's shell initialization file (e.g. ``.bashrc``).
