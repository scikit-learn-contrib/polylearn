# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.utils.extmath import safe_sparse_dot
from scipy.sparse import issparse

import numpy as np


def safe_power(X, degree=2):
    """Element-wise power supporting both sparse and dense data.

    Parameters
    ----------
    X : ndarray or sparse
        The array whose entries to raise to the power.

    degree : int, default: 2
        The power to which to raise the elements.

    Returns
    -------

    X_ret : ndarray or sparse
        Same shape as X, but (x_ret)_ij = (x)_ij ^ degree
    """
    if issparse(X):
        return X.power(degree)
    else:
        return X ** degree


def _D(X, P, degree=2):
    """The "replacement" part of the homogeneous polynomial kernel.

    D[i, j] = sum_k [(X_ik * P_jk) ** degree]
    """
    return safe_sparse_dot(safe_power(X, degree), P.T ** degree)


def homogeneous_kernel(X, P, degree=2):
    """Convenience alias for homogeneous polynomial kernel between X and P::

        K_P(x, p) = <x, p> ^ degree

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 2

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    return polynomial_kernel(X, P, degree=degree, gamma=1, coef0=0)


def anova_kernel(X, P, degree=2):
    """ANOVA kernel between X and P::

        K_A(x, p) = sum_i1>i2>...>id x_i1 p_i1 x_i2 p_i2 ... x_id p_id

    See John Shawe-Taylor and Nello Cristianini,
    Kernel Methods for Pattern Analysis section 9.2.

    Parameters
    ----------
    X : ndarray of shape (n_samples_1, n_features)

    Y : ndarray of shape (n_samples_2, n_features)

    degree : int, default 2

    Returns
    -------
    Gram matrix : array of shape (n_samples_1, n_samples_2)
    """
    if degree == 2:
        K = homogeneous_kernel(X, P, degree=2)
        K -= _D(X, P, degree=2)
        K /= 2
    elif degree == 3:
        K = homogeneous_kernel(X, P, degree=3)
        K -= 3 * _D(X, P, degree=2) * _D(X, P, degree=1)
        K += 2 * _D(X, P, degree=3)
        K /= 6
    else:
        raise NotImplementedError("ANOVA kernel for degree >= 4 not yet "
                                  "implemented efficiently.")
    return K


def _poly_predict(X, P, lams, kernel, degree=2):
    if kernel == "anova":
        K = anova_kernel(X, P, degree)
    elif kernel == "poly":
        K = homogeneous_kernel(X, P, degree)
    else:
        raise ValueError(("Unsuppported kernel: {}. Use one "
                          "of {{'anova'|'poly'}}").format(kernel))

    return np.dot(K, lams)
