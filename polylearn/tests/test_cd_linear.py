from nose.tools import assert_less_equal, assert_greater_equal
from numpy.testing import assert_array_almost_equal

import numpy as np
from sklearn.utils.validation import assert_all_finite
from polylearn.cd_linear_fast import _cd_linear_epoch
from polylearn.loss_fast import Squared, SquaredHinge, Logistic
from lightning.impl.dataset_fast import get_dataset

rng = np.random.RandomState(0)
X = rng.randn(50, 10)
w_true = rng.randn(10)

y = np.dot(X, w_true)
X_ds = get_dataset(X, order='fortran')
X_col_norm_sq = (X ** 2).sum(axis=0)

n_iter = 100


def _fit_linear(X, y, alpha, n_iter, loss, callback=None):
    n_samples, n_features = X.shape
    X_col_norm_sq = (X ** 2).sum(axis=0)
    X_ds = get_dataset(X, order='fortran')
    w_init = np.zeros(n_features)
    y_pred = np.zeros(n_samples)

    for _ in range(n_iter):
        viol = _cd_linear_epoch(w_init, X_ds, y, y_pred, X_col_norm_sq,
                                alpha, loss)
        if callback is not None:
            callback(w_init, viol)
    return w_init


class Callback(object):
    def __init__(self, X, y, alpha):
        self.X = X
        self.y = y
        self.alpha = alpha

        self.losses_ = []

    def __call__(self, w, viol):
        y_pred = np.dot(self.X, w)
        lv = np.mean((y_pred - self.y) ** 2)
        lv += 2 * self.alpha * np.sum(w ** 2)
        self.losses_.append(lv)


def test_cd_linear_fit():
    loss = Squared()
    alpha = 1e-6
    cb = Callback(X, y, alpha)
    w = _fit_linear(X, y, alpha, n_iter, loss, cb)

    assert_array_almost_equal(w_true, w)
    assert_less_equal(cb.losses_[1], cb.losses_[0])
    assert_less_equal(cb.losses_[-1], cb.losses_[0])


def check_cd_linear_clf(loss):
    alpha = 1e-3
    y_bin = np.sign(y)

    w = _fit_linear(X, y_bin, alpha, n_iter, loss)
    y_pred = np.dot(X, w)
    accuracy = np.mean(np.sign(y_pred) == y_bin)

    assert_greater_equal(accuracy, 0.97,
                         msg="classification loss {}".format(loss))


def test_cd_linear_clf():
    for loss in (Squared(), SquaredHinge(), Logistic()):
        yield check_cd_linear_clf, loss


def test_cd_linear_offset():
    loss = Squared()
    alpha = 1e-3
    w_a = np.zeros_like(w_true)
    w_b = np.zeros_like(w_true)

    n_features = X.shape[0]
    y_pred_a = np.zeros(n_features)
    y_pred_b = np.zeros(n_features)
    y_offset = np.arange(n_features).astype(np.double)

    # one epoch with offset
    _cd_linear_epoch(w_a, X_ds, y, y_pred_a + y_offset, X_col_norm_sq, alpha,
                     loss)

    # one epoch with shifted target
    _cd_linear_epoch(w_b, X_ds, y - y_offset, y_pred_b, X_col_norm_sq, alpha,
                     loss)

    assert_array_almost_equal(w_a, w_b)


def test_cd_linear_trivial():
    # trivial example that failed due to gh#4
    loss = Squared()
    alpha = 1e-5
    n_features = 100
    x = np.zeros((1, n_features))
    x[0, 1] = 1
    y = np.ones(1)
    cb = Callback(x, y, alpha)
    w = _fit_linear(x, y, alpha, n_iter=20, loss=loss, callback=cb)

    assert_all_finite(w)
    assert_all_finite(cb.losses_)