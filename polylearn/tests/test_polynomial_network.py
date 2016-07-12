# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings

from nose.tools import assert_less_equal, assert_equal

import numpy as np
from numpy.testing import assert_array_almost_equal
from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_warns_message

from polylearn import PolynomialNetworkClassifier, PolynomialNetworkRegressor


max_degree = 5
n_components = 3
n_features = 7
n_samples = 10

rng = np.random.RandomState(1)
U = rng.randn(max_degree, n_components, n_features)
X = rng.randn(n_samples, n_features)


def cd_lifted_slow(X, y, degree=2, n_components=5, beta=1., n_iter=10000,
                   tol=1e-5, verbose=False, random_state=None):
    from sklearn.utils import check_random_state

    n_samples, n_features = X.shape
    rng = check_random_state(random_state)
    U = rng.randn(degree, n_components, n_features)

    # homogeneous kernel
    pred = np.product(np.dot(U, X.T), axis=0).sum(axis=0)

    mu = 1  # squared loss
    converged = False

    for i in range(n_iter):
        sum_viol = 0
        for t in range(degree):
            deg_idx = np.zeros(degree, dtype=np.bool)
            deg_idx[t] = True
            for s in range(n_components):
                xi = np.product(np.dot(U[~deg_idx, s, :], X.T), axis=0)
                for j in range(n_features):
                    x = X[:, j]

                    inv_step_size = mu * (xi ** 2 * x ** 2).sum()
                    inv_step_size += beta

                    dloss = pred - y  # squared loss
                    step = (xi * x * dloss).sum()
                    step += beta * U[t, s, j]
                    step /= inv_step_size

                    U[t, s, j] -= step
                    sum_viol += np.abs(step)

                    # dumb synchronize
                    pred = np.product(np.dot(U, X.T), axis=0).sum(axis=0)
                    xi = np.product(np.dot(U[~deg_idx, s, :], X.T), axis=0)
        nrm = np.sum(U.ravel() ** 2)
        if verbose:
            print("Epoch", i, "violations", sum_viol, "loss",
                  0.5 * (np.sum((y - pred) ** 2) + beta * nrm))

        if sum_viol < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return U


def check_fit(degree):
    y = np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0)

    est = PolynomialNetworkRegressor(degree=degree, n_components=n_components,
                                     beta=0.00001, tol=1e-4, random_state=0)
    y_pred = est.fit(X, y).predict(X)
    assert_less_equal(mean_squared_error(y, y_pred), 1e-5,
                      msg="Cannot learn degree {} function.".format(degree))


def test_fit():
    for degree in range(2, max_degree + 1):
        yield check_fit, degree


def check_improve(degree):
    y = np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0)

    common_settings = dict(degree=degree, n_components=n_components,
                           beta=1e-10, tol=0, random_state=0)

    est_5 = PolynomialNetworkRegressor(max_iter=5, **common_settings)
    est_10 = PolynomialNetworkRegressor(max_iter=10, **common_settings)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        est_5.fit(X, y)
        est_10.fit(X, y)

    y_pred_5 = est_5.predict(X)
    y_pred_10 = est_10.predict(X)

    assert_less_equal(mean_squared_error(y, y_pred_10),
                      mean_squared_error(y, y_pred_5),
                      msg="More iterations do not improve fit.")


def test_improve():
    for degree in range(2, max_degree + 1):
        yield check_improve, degree


def test_convergence_warning():
    degree = 4
    y = np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0)

    est = PolynomialNetworkRegressor(degree=degree, n_components=n_components,
                                     beta=1e-10, max_iter=1, tol=1e-5,
                                     random_state=0)
    assert_warns_message(UserWarning, "converge", est.fit, X, y)


def test_random_starts():
    # not as strong a test as the direct case!
    # using training error here, and a higher threshold.
    # We observe the lifted solver reaches rather diff. solutions.
    degree = 3
    noisy_y = np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0)
    noisy_y += 5. * rng.randn(noisy_y.shape[0])

    common_settings = dict(degree=degree, n_components=n_components,
                           beta=0.001, tol=0.01)
    scores = []
    for k in range(5):
        est = PolynomialNetworkRegressor(random_state=k, **common_settings)
        y_pred = est.fit(X, noisy_y).predict(X)
        scores.append(mean_squared_error(noisy_y, y_pred))

    assert_less_equal(np.std(scores), 1e-4)


def check_same_as_slow(degree):
    y = np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0)
    reg = PolynomialNetworkRegressor(degree=degree, n_components=n_components,
                                     fit_lower=None, fit_linear=None,
                                     beta=1e-3, tol=1e-2, random_state=0)
    reg.fit(X, y)
    U_fit_slow = cd_lifted_slow(X, y, degree=degree, n_components=n_components,
                                beta=1e-3, tol=1e-2, random_state=0)

    assert_array_almost_equal(reg.U_, U_fit_slow)


def test_same_as_slow():
    for degree in range(2, max_degree + 1):
        yield check_same_as_slow, degree


def check_classification_losses(loss, degree):
    y = np.sign(np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0))

    clf = PolynomialNetworkClassifier(degree=degree, n_components=n_components,
                                      loss=loss, beta=1e-3, tol=1e-2,
                                      random_state=0)
    clf.fit(X, y)
    assert_equal(1.0, clf.score(X, y))


def test_classification_losses():
    for loss in ('squared_hinge', 'logistic'):
        for degree in range(2, max_degree + 1):
            yield check_classification_losses, loss, degree


def check_warm_start(degree):
    y = np.sign(np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0))
    # Result should be the same if:
    # (a) running 10 iterations

    common_settings = dict(fit_lower=None, fit_linear=None, degree=degree,
                           n_components=2, random_state=0)
    clf_10 = PolynomialNetworkRegressor(max_iter=10, warm_start=False,
                                        **common_settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_10.fit(X, y)

    # (b) running 5 iterations and 5 more
    clf_5_5 = PolynomialNetworkRegressor(max_iter=5, warm_start=True,
                                         **common_settings)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5_5.fit(X, y)
        U_fit = clf_5_5.U_.copy()
        clf_5_5.fit(X, y)

    # (c) running 5 iterations when starting from previous point.
    clf_5 = PolynomialNetworkRegressor(max_iter=5, warm_start=True,
                                       **common_settings)
    clf_5.U_ = U_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5.fit(X, y)

    assert_array_almost_equal(clf_10.U_, clf_5_5.U_)
    assert_array_almost_equal(clf_10.U_, clf_5.U_)

    # Prediction results should also be the same if:
    # (note: could not get this test to work for the exact P_.)
    # This test is very flimsy!

    y = np.sign(np.product(np.dot(U[:degree], X.T), axis=0).sum(axis=0))

    beta_low = 0.51
    beta = 0.5
    beta_hi = 0.49

    common_settings = dict(degree=degree, n_components=n_components,
                           tol=1e-3, random_state=0)
    ref = PolynomialNetworkRegressor(beta=beta, **common_settings)
    ref.fit(X, y)
    y_pred_ref = ref.predict(X)

    # # (a) starting from lower beta, increasing and refitting
    from_low = PolynomialNetworkRegressor(beta=beta_low, warm_start=True,
                                          **common_settings)
    from_low.fit(X, y)
    from_low.set_params(beta=beta)
    from_low.fit(X, y)
    y_pred_low = from_low.predict(X)

    # (b) starting from higher beta, decreasing and refitting
    from_hi = PolynomialNetworkRegressor(beta=beta_hi, warm_start=True,
                                         **common_settings)
    from_hi.fit(X, y)
    from_hi.set_params(beta=beta)
    from_hi.fit(X, y)
    y_pred_hi = from_hi.predict(X)

    decimal = 3
    assert_array_almost_equal(y_pred_low, y_pred_ref, decimal=decimal)
    assert_array_almost_equal(y_pred_hi, y_pred_ref, decimal=decimal)


def test_warm_start():
    for degree in range(2, max_degree + 1):
        yield check_warm_start, degree
