# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Vlad Niculae
# License: BSD

from libc.math cimport fabs
from cython.view cimport array

from lightning.impl.dataset_fast cimport ColumnDataset

from .loss_fast cimport LossFunction


def _fast_lifted_predict(double[:, :, ::1] U,
                         ColumnDataset X,
                         double[:] out):

    # np.product(safe_sparse_dot(U, X.T), axis=0).sum(axis=0)
    #
    # a bit of a misnomer, since at least for dense data it's a bit slower,
    # but it's more memory efficient.

    cdef Py_ssize_t degree = U.shape[0]
    cdef Py_ssize_t n_components = U.shape[1]

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, j, ii

    cdef double[:] middle = array((n_samples,), sizeof(double), 'd')
    cdef double[:] inner = array((n_samples,), sizeof(double), 'd')

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


cdef void _precompute(double[:, :, ::1] U,
                      ColumnDataset X,
                      Py_ssize_t s,
                      Py_ssize_t t,
                      double[:] out,
                      double[:] tmp):

    cdef Py_ssize_t degree = U.shape[0]
    cdef Py_ssize_t n_components = U.shape[1]

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()

    cdef double* data
    cdef int* indices
    cdef int n_nz

    cdef Py_ssize_t i, j, ii

    for i in range(n_samples):
        out[i] = 1

    for t_prime in range(degree):

        if t == t_prime:
            continue

        for i in range(n_samples):
            tmp[i] = 0

        for j in range(n_features):
            X.get_column_ptr(j, &indices, &data, &n_nz)
            for ii in range(n_nz):
                i = indices[ii]
                tmp[i] += data[ii] * U[t_prime, s, j]
        for i in range(n_samples):
            out[i] *= tmp[i]


def _cd_lifted(double[:, :, ::1] U,
               ColumnDataset X,
               double[:] y,
               double[:] y_pred,
               double beta,
               LossFunction loss,
               int max_iter,
               double tol,
               int verbose):

    cdef Py_ssize_t n_samples = X.get_n_samples()
    cdef Py_ssize_t n_features = X.get_n_features()
    cdef Py_ssize_t degree = U.shape[0]
    cdef Py_ssize_t n_components = U.shape[1]
    cdef Py_ssize_t t, s, j
    cdef int it

    cdef double sum_viol
    cdef bint converged = False

    cdef double inv_step_size
    cdef double update
    cdef double u_old

    cdef double[:] xi = array((n_samples,), sizeof(double), 'd')
    cdef double[:] tmp = array((n_samples,), sizeof(double), 'd')

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for it in range(max_iter):
        sum_viol = 0
        for t in range(degree):
            for s in range(n_components):
                _precompute(U, X, s, t, xi, tmp)
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
            print("Iteration", it + 1, "violation sum", sum_viol)

        if sum_viol < tol:
            if verbose:
                print("Converged at iteration", it + 1)
            converged = True
            break

    return converged
