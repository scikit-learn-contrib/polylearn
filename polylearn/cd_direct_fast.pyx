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

from lightning.impl.dataset_fast cimport ColumnDataset

from .loss_fast cimport LossFunction
from .cd_linear_fast cimport _cd_linear_epoch


cdef void _precompute(ColumnDataset X,
                      double[:, :, ::1] P,
                      Py_ssize_t order,
                      double* out,
                      Py_ssize_t s,
                      unsigned int degree):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = P.shape[2]
    
    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz
    
    cdef Py_ssize_t i, j, ii

    for i in range(n_samples):
        out[i] = 0
    
    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)
        for ii in range(n_nz):
            i = indices[ii]
            out[i] += (data[ii] * P[order, s, j]) ** degree


cdef inline double _update(int* indices,
                           double* data,
                           int n_nz,
                           double p_js,
                           double[:] y,
                           double[:] y_pred,
                           LossFunction loss,
                           double* d1,
                           double* d2,
                           unsigned int degree,
                           double lam,
                           double beta,
                           double* cache_kp):

    cdef double l1_reg = 2 * beta * fabs(lam)
    
    cdef Py_ssize_t i, ii

    cdef double inv_step_size = 0

    cdef double kp  # derivative of the ANOVA kernel
    cdef double update = 0

    for ii in range(n_nz):
        i = indices[ii]

        if degree == 2:
            kp = d1[i] - p_js * data[ii]
        elif degree == 3:
            kp = 0.5 * (d1[i] ** 2 - d2[i])
            kp -= p_js * data[ii] * d1[i]
            kp += p_js ** 2 * data[ii] ** 2

        kp *= lam * data[ii]
        cache_kp[ii] = kp

        update += loss.dloss(y_pred[i], y[i]) * kp
        inv_step_size += kp ** 2

    inv_step_size *= loss.mu
    inv_step_size += l1_reg

    update += l1_reg * p_js
    update /= inv_step_size

    return update


cdef inline double _cd_direct_epoch(double[:, :, ::1] P,
                                    Py_ssize_t order,
                                    ColumnDataset X,
                                    double[:] y,
                                    double[:] y_pred,
                                    double[:] lams,
                                    double* d1,
                                    double* d2,
                                    unsigned int degree,
                                    double beta,
                                    LossFunction loss,
                                    double* cache_kp):

    cdef Py_ssize_t s, j
    cdef double p_old, update, offset
    cdef double sum_viol = 0
    cdef Py_ssize_t n_components = P.shape[1]
    cdef Py_ssize_t n_features = P.shape[2]

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
                             loss, d1, d2, degree, lams[s], beta, cache_kp)
            P[order, s, j] -= update
            sum_viol += fabs(update)

            # Synchronize predictions and ds
            for ii in range(n_nz):
                i = indices[ii]

                if degree == 3:
                    d2[i] -= (p_old ** 2 - P[order, s, j] ** 2) * data[ii] ** 2

                d1[i] -= update * data[ii]
                y_pred[i] -= update * cache_kp[ii]
    return sum_viol


def _cd_direct_ho(double[:, :, ::1] P not None,
                  double[:] w not None,
                  ColumnDataset X,
                  double[:] col_norm_sq not None,
                  double[:] y not None,
                  double[:] y_pred not None,
                  double[:] lams not None,
                  unsigned int degree,
                  double alpha,
                  double beta,
                  bint fit_linear,
                  bint fit_lower,
                  LossFunction loss,
                  unsigned int max_iter,
                  double tol,
                  int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef unsigned int it

    cdef double viol
    cdef bint converged = False

    # precomputed values
    cdef double *d1 = <double *> malloc(n_samples * sizeof(double))
    cdef double *d2
    if degree == 3:
        d2 = <double *> malloc(n_samples * sizeof(double))
    cdef double *cache_kp = <double *> malloc(n_samples * sizeof(double))

    for it in range(max_iter):
        viol = 0

        if fit_linear:
            viol += _cd_linear_epoch(w, X, y, y_pred, col_norm_sq, alpha, loss)

        if fit_lower and degree == 3:  # fit degree 2. Will be looped later.
            viol += _cd_direct_epoch(P, 1, X, y, y_pred, lams, d1, d2,
                                     2, beta, loss, cache_kp)

        viol += _cd_direct_epoch(P, 0, X, y, y_pred, lams, d1, d2,
                                 degree, beta, loss, cache_kp)

        if verbose:
            print("Iteration", it + 1, "violation sum", viol)

        if viol < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            converged = True
            break

    # Free up cache
    free(d1)
    free(cache_kp)
    if degree == 3:
        free(d2)

    return converged
