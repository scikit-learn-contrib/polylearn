cimport numpy as np
from lightning.impl.dataset_fast cimport ColumnDataset
from .loss_fast cimport LossFunction

cpdef double _cd_linear_epoch(np.ndarray[double, ndim=1] w, ColumnDataset X,
                              np.ndarray[double, ndim=1] y,
                              np.ndarray[double, ndim=1] y_pred,
                              np.ndarray[double, ndim=1] col_norm_sq,
                              double alpha,
                              LossFunction loss)