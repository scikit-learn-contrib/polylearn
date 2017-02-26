.. -*- mode: rst -*-

polylearn
=========

A library for **factorization machines** and **polynomial networks**
for classification and regression in Python.

`Github repository. <https://github.com/scikit-learn-contrib/polylearn/>`_

.. image:: https://travis-ci.org/scikit-learn-contrib/polylearn.svg?branch=master
    :target: https://travis-ci.org/scikit-learn-contrib/polylearn

.. image:: https://ci.appveyor.com/api/projects/status/g9xnar9081l3vsw7/branch/master?svg=true
    :target: https://ci.appveyor.com/project/vene/polylearn

.. image:: https://coveralls.io/repos/scikit-learn-contrib/polylearn/badge.svg?branch=master&service=github
    :target: https://coveralls.io/r/scikit-learn-contrib/polylearn

.. image:: https://circleci.com/gh/scikit-learn-contrib/polylearn/tree/master.svg?style=shield&circle-token=:circle-token
    :target: https://circleci.com/gh/scikit-learn-contrib/polylearn/

Factorization machines and polynomial networks are machine learning models
that can capture **feature interaction** (co-occurrence) through polynomial terms.
Because feature interactions can be very sparse, it's common to use **low rank,
factorized representations**; this way, we can learn weights even for feature
co-occurrences that haven't been observed at training time.

Factorization machines are popular for recommender systems, as they are a
generalization of matrix completion models.

This package provides:

- coordinate descent algorithm for fitting factorization machines of degree 2 or 3,
- coordinate descent algorithm for fitting polynomial networks of arbitrary degree,
- `scikit-learn <http://scikit-learn.org>`_-compatible API,
- `Cython <http://cython.org>`_ implementations for computationally intensive parts.

Installation
------------

Binary packages are not yet available.

The development version of polylearn can be installed from its git repository. In
this case it is assumed that you have a working
C++ compiler.

1. Obtain the sources by::

    git clone https://github.com/scikit-learn-contrib/polylearn.git
 
or, if `git` is unavailable, `download as a ZIP from GitHub <https://github.com/scikit-learn-contrib/polylearn/archive/master.zip>`_.
 
 
2. Install the dependencies::
 
    # via pip

    pip install numpy scipy scikit-learn nose
    pip install sklearn-contrib-lightning

 
    # via conda

    conda install numpy scipy scikit-learn nose
    conda install -c conda-forge sklearn-contrib-lightning


3. Build and install polylearn::

    cd polylearn
    python setup.py build
    sudo python setup.py install


References
----------

The solvers implemented are introduced in [1]_. Factorization machines are introduced
in [2]_ and polynomial networks in [3]_.

.. [1] Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
       *Polynomial Networks and Factorization Machines: New Insights and
       Efficient Training Algorithms.*  In: Proc. of ICML 2016.
       [`PDF <http://mblondel.org/publications/mblondel-icml2016.pdf>`_]

.. [2] Steffen Rendle. *Factorization machines.* In: Proc. of IEEE ICDM 2010.
       [`PDF <https://www.ismll.uni-hildesheim.de/pub/pdfs/Rendle2010FM.pdf>`_]

.. [3] Roi Livni, Shai Shalev-Shwartz, Ohad Shamir.
       *On the computational efficiency of training neural networks.*
       In: Proc. of NIPS 2014.
       [`arXiv <http://arxiv.org/abs/1410.1141>`_]

Authors
-------

- Vlad Niculae, 2016-present
