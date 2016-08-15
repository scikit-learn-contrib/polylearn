# encoding: utf-8
# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
#
# Author: Vlad Niculae
# License: BSD

from libc.math cimport fabs

from lightning.impl.dataset_fast cimport ColumnDataset

from .loss_fast cimport LossFunction


cpdef double _cd_linear_epoch(double[:] w,
                              ColumnDataset X,
                              double[:] y,
                              double[:] y_pred,
                              double[:] col_norm_sq,
                              double alpha,
                              LossFunction loss):

    cdef Py_ssize_t i, j, ii
    cdef double sum_viol = 0
    cdef Py_ssize_t n_features = w.shape[0]
    cdef double update
    cdef double inv_step_size

    # Data pointers
    cdef double* data
    cdef int* indices
    cdef int n_nz

    for j in range(n_features):
        X.get_column_ptr(j, &indices, &data, &n_nz)

        # compute gradient with respect to w_j
        update = alpha * w[j]
        for ii in range(n_nz):
            i = indices[ii]
            update += loss.dloss(y_pred[i], y[i]) * data[ii]

        # compute second derivative upper bound
        inv_step_size = loss.mu * col_norm_sq[j] + alpha
        update /= inv_step_size

        w[j] -= update
        sum_viol += fabs(update)

        # update predictions
        for ii in range(n_nz):
            i = indices[ii]
            y_pred[i] -= update * data[ii]

    return sum_viol
