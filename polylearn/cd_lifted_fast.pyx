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


cpdef void _fast_lifted_predict(double[:, :, ::1] U,
                                ColumnDataset X,
                                double[:] out):

    # np.product(safe_sparse_dot(U, X.T), axis=0).sum(axis=0)
    #
    # a bit of a misnomer, since at least for dense data it's a bit slower,
    # but it's more memory efficient.

    cdef int degree = U.shape[0]
    cdef int n_components = U.shape[1]

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef int i, j, ii

    cdef double *middle = <double *> malloc(n_samples * sizeof(double))
    cdef double *inner = <double *> malloc(n_samples * sizeof(double))


    for s in range(n_components):

        for i in range(n_samples):
            middle[i] = 1

        for t in range(degree):
            # inner = np.dot(U[t, s, :], X.T)

            for i in range(n_samples):
                inner[i] = 0

            for j in range(n_features):
                X.get_column_ptr(j, &indices, &data, &n_nz)
                for ii in range(n_nz):
                    i = indices[ii]
                    inner[i] += data[ii] * U[t, s, j]

            # middle *= inner
            for i in range(n_samples):
                middle[i] *= inner[i]

        for i in range(n_samples):
            out[i] += middle[i]

    free(inner)
    free(middle)


cdef void _precompute(double[:, :, ::1] U,
                      ColumnDataset X,
                      int s,
                      int t,
                      double* out):

    cdef int degree = U.shape[0]
    cdef int n_components = U.shape[1]

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef int i, j, ii

    cdef double *inner = <double *> malloc(n_samples * sizeof(double))

    for i in range(n_samples):
        out[i] = 1

    for t_prime in range(degree):

        if t == t_prime:
            continue

        for i in range(n_samples):
            inner[i] = 0

        for j in range(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)
            for ii in range(n_nz):
                i = indices[ii]
                inner[i] += data[ii] * U[t_prime, s, j]
        for i in range(n_samples):
            out[i] *= inner[i]
    free(inner)


# cdef double _total_loss(np.ndarray[double, ndim=1] y_pred,
#                         np.ndarray[double, ndim=1] y,
#                         np.ndarray[double, ndim=3, mode='c'] U,
#                         double beta,
#                         LossFunction loss):
#
#     cdef double result = 0
#     cdef int degree = U.shape[0]
#     cdef int n_components = U.shape[1]
#     cdef int n_features = U.shape[2]
#     cdef int n_samples = y.shape[0]
#
#     cdef int i, t, s, j
#
#     # regularization
#     for t in range(degree):
#         for s in range(n_components):
#             for j in range(n_features):
#                 result += U[t, s, j] ** 2
#     result *= beta
#
#     # loss
#     for i in range(n_samples):
#         result += loss.loss(y_pred[i], y[i])
#     return result


def _cd_lifted(double[:, :, ::1] U,
               ColumnDataset X,
               double[:] y,
               double[:] y_pred,
               double beta,
               LossFunction loss,
               int max_iter,
               double tol,
               int verbose,
               bint compute_loss):

    cdef int n_samples = X.get_n_samples()
    cdef int n_features = X.get_n_features()
    cdef int degree = U.shape[0]
    cdef int n_components = U.shape[1]
    cdef int it, t, s, j

    cdef double sum_viol
    cdef bint converged = False

    cdef double inv_step_size
    cdef double update
    cdef double u_old

    cdef double *xi = <double *> malloc(n_samples * sizeof(double))

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for it in range(max_iter):
        sum_viol = 0
        for t in range(degree):
            for s in range(n_components):
                _precompute(U, X, s, t, xi)
                for j in range(n_features):

                    u_old = U[t, s, j]
                    X.get_column_ptr(j, &indices, &data, &n_nz)

                    inv_step_size = 0
                    update = 0

                    for ii in range(n_nz):
                        i = indices[ii]
                        inv_step_size += xi[i] ** 2 * data[ii] ** 2
                        update += xi[i] * data[ii] * loss.dloss(y_pred[i],
                                                                y[i])

                    inv_step_size *= loss.mu
                    inv_step_size += beta

                    update += beta * u_old
                    update /= inv_step_size

                    U[t, s, j] -= update
                    sum_viol += fabs(update)

                    # synchronize predictions
                    for ii in range(n_nz):
                        i = indices[ii]
                        y_pred[i] -= data[ii] * xi[i] * update

        if verbose:
            print("Iteration", it + 1, "violation sum", sum_viol, end=" ")
            # if compute_loss:
            #     print("loss", _total_loss(y_pred, y, U, beta, loss), end="")
            print()

        if sum_viol < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            converged = True
            break

    free(xi)
    return converged
