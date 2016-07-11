# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Vlad Niculae
# License: BSD

from libc.stdlib cimport malloc, free
from libc.math cimport fabs

cimport numpy as np

from lightning.impl.dataset_fast cimport ColumnDataset

from .loss_fast cimport LossFunction
from .cd_linear_fast cimport _cd_linear_epoch


cdef void _precompute(ColumnDataset X,
                      np.ndarray[double, ndim=3, mode='c'] P,
                      int order,
                      double* out,
                      int s,
                      int degree):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = P.shape[2]
    
    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz
    
    cdef int i, j, ii

    for i in range(n_samples):
        out[i] = 0
    
    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            out[i] += (data[ii] * P[order, s, j]) ** degree


cdef double _update(int* indices,
                    double* data,
                    int n_nz,
                    double p_js,
                    np.ndarray[double, ndim=1] y,
                    np.ndarray[double, ndim=1] y_pred,
                    LossFunction loss,
                    double* d1,
                    double* d2,
                    int degree,
                    double lam,
                    double beta):

    cdef double l1_reg = 2 * beta * fabs(lam)
    
    cdef int i, ii

    cdef double inv_step_size

    cdef double grad_y
    cdef double update = 0

    for ii in range(n_nz):
        i = indices[ii]

        if degree == 2:
            grad_y = d1[i] - p_js * data[ii]
        elif degree == 3:
            grad_y = 0.5 * (d1[i] ** 2 - d2[i])
            grad_y -= p_js * data[ii] * d1[i]
            grad_y += p_js ** 2 * data[ii] ** 2

        grad_y *= lam * data[ii]

        update += loss.dloss(y_pred[i], y[i]) * grad_y
        inv_step_size += grad_y ** 2

    inv_step_size *= loss.mu
    inv_step_size += l1_reg

    update += l1_reg * p_js
    update /= inv_step_size
    return update


cdef double _total_loss(np.ndarray[double, ndim=1] y,
                        np.ndarray[double, ndim=1] y_pred,
                        np.ndarray[double, ndim=3, mode='c'] P,
                        np.ndarray[double, ndim=1] lams,
                        np.ndarray[double, ndim=1] w,
                        double alpha,
                        double beta,
                        LossFunction loss):

    cdef double result = 0
    cdef double linear = 0
    cdef double row_norm

    cdef int n_orders = P.shape[0]
    cdef int n_components = P.shape[1]
    cdef int n_features = P.shape[2]
    cdef int n_samples = y.shape[0]

    cdef int i, s, j

    # regularization
    for o in range(n_orders):
        for s in range(n_components):
            row_norm = 0
            for j in range(n_features):
                row_norm += P[o, s, j] ** 2
            result += row_norm * lams[s]
    result *= beta

    for j in range(n_features):
        linear += w[j] ** 2

    linear *= alpha

    result += linear

    for i in range(n_samples):
        result += loss.loss(y_pred[i], y[i])

    return result


cdef double _cd_direct_epoch(np.ndarray[double, ndim=3, mode='c'] P,
                             int order,
                             ColumnDataset X,
                             np.ndarray[double, ndim=1] y,
                             np.ndarray[double, ndim=1] y_pred,
                             np.ndarray[double, ndim=1] lams,
                             double* d1,
                             double* d2,
                             int degree,
                             double beta,
                             LossFunction loss):

    cdef int s, j
    cdef double p_old, update, offset
    cdef double sum_viol = 0
    cdef int n_components = P.shape[1]
    cdef int n_features = P.shape[2]

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for s in range(n_components):

        # initialize the cached ds for this s
        _precompute(X, P, order, d1, s, 1)
        if degree == 3:
            _precompute(X, P, order, d2, s, 2)

        for j in range(n_features):

            X.get_column_ptr(j, &indices, &data, &n_nz)

            # compute coordinate update
            p_old = P[order, s, j]
            update = _update(indices, data, n_nz, p_old, y, y_pred,
                             loss, d1, d2, degree, lams[s], beta)
            P[order, s, j] -= update
            sum_viol += fabs(update)

            # Synchronize predictions and ds
            for ii in range(n_nz):
                i = indices[ii]
                if degree == 2:
                    offset = d1[i] - p_old * data[ii]
                else:
                    offset = d1[i] ** 2 - d2[i]
                    offset *= 0.5
                    offset -= p_old * data[ii] * d1[i]
                    offset += p_old ** 2 * data[ii] ** 2

                    d2[i] -= (p_old ** 2 - P[order, s, j] ** 2) * data[ii] ** 2

                d1[i] -= update * data[ii]
                y_pred[i] -= offset * lams[s] * update * data[ii]
    return sum_viol


def _cd_direct_ho(np.ndarray[double, ndim=3, mode='c'] P,
                  np.ndarray[double, ndim=1] w,
                  ColumnDataset X,
                  np.ndarray[double, ndim=1] col_norm_sq,
                  np.ndarray[double, ndim=1] y,
                  np.ndarray[double, ndim=1] y_pred,
                  np.ndarray[double, ndim=1, mode='c'] lams,
                  int degree,
                  double alpha,
                  double beta,
                  bint fit_linear,
                  bint fit_lower,
                  LossFunction loss,
                  int max_iter,
                  double tol,
                  int verbose,
                  bint compute_loss):

    cdef int n_samples = X.get_n_samples()
    cdef int it

    cdef double viol
    cdef bint converged = False

    # precomputed values
    cdef double *d1 = <double *> malloc(n_samples * sizeof(double))
    cdef double *d2
    if degree == 3:
        d2 = <double *> malloc(n_samples * sizeof(double))

    for it in range(max_iter):
        viol = 0

        if fit_linear:
            viol += _cd_linear_epoch(w, X, y, y_pred, col_norm_sq, alpha, loss)

        if fit_lower and degree == 3:  # fit degree 2. Will be looped later.
            viol += _cd_direct_epoch(P, 1, X, y, y_pred, lams, d1, d2,
                                     2, beta, loss)

        viol += _cd_direct_epoch(P, 0, X, y, y_pred, lams, d1, d2,
                                 degree, beta, loss)

        if verbose:
            print("Iteration", it + 1, "violation sum", viol, end=" ")
            if compute_loss:
                print("loss", _total_loss(y_pred, y, P, lams, w, alpha, beta,
                                          loss), end="")
            print()

        if viol < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            converged = True
            break

    # Free up cache
    free(d1)
    if degree == 3:
        free(d2)

    return converged