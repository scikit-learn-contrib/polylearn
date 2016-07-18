cimport numpy as np
from lightning.impl.dataset_fast cimport ColumnDataset
from .loss_fast cimport LossFunction

cpdef double _cd_linear_epoch(double[:] w, ColumnDataset X,
                              double[:] y,
                              double[:] y_pred,
                              double[:] col_norm_sq,
                              double alpha,
                              LossFunction loss)