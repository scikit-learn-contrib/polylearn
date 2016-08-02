""" Benchmarking CD solvers for factorization machines.

Compares polylearn with with fastFM [1].

[1] http://ibayer.github.io/fastFM/

Note: this benchmark uses the squared loss and a regression formulation, for
the fairest comparison.  The CD solvers in polylearn support logistic loss and
squared hinge loss as well.

"""

from time import time

import numpy as np
import scipy.sparse as sp

from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import fetch_20newsgroups_vectorized

from polylearn import FactorizationMachineRegressor
if __name__ == '__main__':
    data_train = fetch_20newsgroups_vectorized(subset="train")
    data_test = fetch_20newsgroups_vectorized(subset="test")
    X_train = sp.csc_matrix(data_train.data)
    X_test = sp.csc_matrix(data_test.data)

    y_train = data_train.target == 0  # atheism vs rest
    y_test = data_test.target == 0

    y_train = (2 * y_train - 1).astype(np.float)

    print(__doc__)
    print("20 newsgroups")
    print("=============")
    print("X_train.shape = {0}".format(X_train.shape))
    print("X_train.format = {0}".format(X_train.format))
    print("X_train.dtype = {0}".format(X_train.dtype))
    print("X_train density = {0}"
          "".format(X_train.nnz / np.product(X_train.shape)))
    print("y_train {0}".format(y_train.shape))
    print("X_test {0}".format(X_test.shape))
    print("X_test.format = {0}".format(X_test.format))
    print("X_test.dtype = {0}".format(X_test.dtype))
    print("y_test {0}".format(y_test.shape))
    print()

    print("Training regressors")
    print("===================")
    f1, accuracy, train_time, test_time = {}, {}, {}, {}

    print("Training our solver... ", end="")
    fm = FactorizationMachineRegressor(n_components=20,
                                       fit_linear=True,
                                       fit_lower=False,
                                       alpha=5,
                                       beta=5,
                                       degree=2,
                                       random_state=0,
                                       max_iter=100)
    t0 = time()
    fm.fit(X_train, y_train)
    train_time['polylearn'] = time() - t0
    t0 = time()
    y_pred = fm.predict(X_test) > 0
    test_time['polylearn'] = time() - t0
    accuracy['polylearn'] = accuracy_score(y_test, y_pred)
    f1['polylearn'] = f1_score(y_test, y_pred)
    print("done")

    try:
        from fastFM import als

        print("Training fastfm... ", end="")
        clf = als.FMRegression(n_iter=100, init_stdev=0.01, rank=20,
                               random_state=0, l2_reg=10.)
        clf.ignore_w_0 = True  # since polylearn has no fit_intercept yet
        t0 = time()

        clf.fit(X_train, y_train)
        train_time['fastfm'] = time() - t0

        t0 = time()
        y_pred = clf.predict(X_test)
        test_time['fastfm'] = time() - t0
        y_pred = y_pred > 0
        accuracy['fastfm'] = accuracy_score(y_test, y_pred)
        f1['fastfm'] = f1_score(y_test, y_pred)

        print("done")
    except ImportError:
        print("fastfm not found")

    print("Regression performance:")
    print("=======================")
    print()
    print("%s %s %s %s %s" % ("Model".ljust(16),
                              "train".rjust(10),
                              "test".rjust(10),
                              "f1".rjust(10),
                              "accuracy".rjust(10)))
    print("-" * (16 + 4 * 11))
    for name in sorted(f1, key=f1.get):
        print("%s %s %s %s %s" % (
            name.ljust(16),
            ("%.4fs" % train_time[name]).rjust(10),
            ("%.4fs" % test_time[name]).rjust(10),
            ("%.4f" % f1[name]).rjust(10),
            ("%.4f" % accuracy[name]).rjust(10)))

    print()
