# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings
import ctypes

from nose.tools import assert_less_equal, assert_equal
from nose import SkipTest

import numpy as np
from numpy.testing import assert_array_almost_equal

from sklearn.metrics import mean_squared_error
from sklearn.utils.testing import assert_warns_message

from polylearn.kernels import _poly_predict
from polylearn import FactorizationMachineRegressor
from polylearn import FactorizationMachineClassifier


def cd_direct_slow(X, y, lams=None, degree=2, n_components=5, beta=1.,
                   n_iter=10, tol=1e-5, verbose=False, random_state=None):
    from sklearn.utils import check_random_state
    from polylearn.kernels import anova_kernel

    n_samples, n_features = X.shape

    rng = check_random_state(random_state)
    P = 0.01 * rng.randn(n_components, n_features)
    if lams is None:
        lams = np.ones(n_components)

    K = anova_kernel(X, P, degree=degree)
    pred = np.dot(lams, K.T)

    mu = 1  # squared loss
    converged = False

    for i in range(n_iter):
        sum_viol = 0
        for s in range(n_components):
            ps = P[s]
            for j in range(n_features):

                # trivial approach:
                # multilinearity allows us to isolate the term with ps_j * x_j
                x = X[:, j]
                notj_mask = np.arange(n_features) != j
                X_notj = X[:, notj_mask]
                ps_notj = ps[notj_mask]

                if degree == 2:
                    grad_y = lams[s] * x * np.dot(X_notj, ps_notj)
                elif degree == 3:
                    grad_y = lams[s] * x * anova_kernel(np.atleast_2d(ps_notj),
                                                        X_notj, degree=2)
                else:
                    raise NotImplementedError("Degree > 3 not supported.")

                l1_reg = 2 * beta * np.abs(lams[s])
                inv_step_size = mu * (grad_y ** 2).sum() + l1_reg

                dloss = pred - y  # squared loss
                step = (dloss * grad_y).sum() + l1_reg * ps[j]
                step /= inv_step_size

                P[s, j] -= step
                sum_viol += np.abs(step)

                # stupidly recompute all predictions. No rush yet.
                K = anova_kernel(X, P, degree=degree)
                pred = np.dot(lams, K.T)

        reg_obj = beta * np.sum((P ** 2).sum(axis=1) * np.abs(lams))

        if verbose:
            print("Epoch", i, "violations", sum_viol, "obj",
                  0.5 * ((pred - y) ** 2).sum() + reg_obj)

        if sum_viol < tol:
            converged = True
            break

    if not converged:
        warnings.warn("Objective did not converge. Increase max_iter.")

    return P


n_components = 5
n_features = 4
n_samples = 20

rng = np.random.RandomState(1)

X = rng.randn(n_samples, n_features)
P = rng.randn(n_components, n_features)

lams = rng.randn(n_components)


def test_augment():
    """Test that augmenting the data increases the dimension as expected"""
    y = _poly_predict(X, P, lams, kernel="anova", degree=3)
    fm = FactorizationMachineRegressor(degree=3, fit_lower='augment',
                                       fit_linear=True, tol=0.1)
    fm.fit(X, y)
    assert_equal(n_features + 1, fm.P_.shape[2],
                 msg="Augmenting is wrong with explicit linear term.")

    fm.set_params(fit_linear=False)
    fm.fit(X, y)
    assert_equal(n_features + 2, fm.P_.shape[2],
                 msg="Augmenting is wrong with augmented linear term.")


def check_fit(degree):
    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    est = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_linear=None, fit_lower=None,
                                        max_iter=15000, beta=1e-6, tol=1e-3,
                                        random_state=0)
    est.fit(X, y)
    y_pred = est.predict(X)
    err = mean_squared_error(y, y_pred)

    assert_less_equal(
        err,
        1e-6,
        msg="Error {} too big for degree {}.".format(err, degree))


def test_fit():
    yield check_fit, 2
    yield check_fit, 3


def check_improve(degree):
    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    est = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_lower="explicit", beta=0.001,
                                        max_iter=5, tol=0, random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        y_pred_5 = est.fit(X, y).predict(X)
        est.set_params(max_iter=10)
        y_pred_10 = est.fit(X, y).predict(X)

    assert_less_equal(mean_squared_error(y, y_pred_10),
                      mean_squared_error(y, y_pred_5),
                      msg="More iterations do not improve fit.")


def test_improve():
    yield check_improve, 2
    yield check_improve, 3


def check_overfit(degree):
    noisy_y = _poly_predict(X, P, lams, kernel="anova", degree=degree)
    noisy_y += 5. * rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    # weak regularization, should overfit
    est = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_linear=False, fit_lower=None,
                                        beta=1e-4, tol=0.01, random_state=0)
    y_train_pred_weak = est.fit(X_train, y_train).predict(X_train)
    y_test_pred_weak = est.predict(X_test)

    est.set_params(beta=10)  # high value of beta -> strong regularization
    y_train_pred_strong = est.fit(X_train, y_train).predict(X_train)
    y_test_pred_strong = est.predict(X_test)

    assert_less_equal(mean_squared_error(y_train, y_train_pred_weak),
                      mean_squared_error(y_train, y_train_pred_strong),
                      msg="Training error does not get worse with regul.")

    assert_less_equal(mean_squared_error(y_test, y_test_pred_strong),
                      mean_squared_error(y_test, y_test_pred_weak),
                      msg="Test error does not get better with regul.")


def test_overfit():
    yield check_overfit, 2
    yield check_overfit, 3


def test_convergence_warning():
    y = _poly_predict(X, P, lams, kernel="anova", degree=3)

    est = FactorizationMachineRegressor(degree=3, beta=1e-8, max_iter=1,
                                        random_state=0)
    assert_warns_message(UserWarning, "converge", est.fit, X, y)


def test_random_starts():
    noisy_y = _poly_predict(X, P, lams, kernel="anova", degree=2)
    noisy_y += 5. * rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    scores = []
    # init_lambdas='ones' is important to reduce variance here
    reg = FactorizationMachineRegressor(degree=2, n_components=n_components,
                                        beta=5, fit_lower=None,
                                        fit_linear=False, max_iter=2000,
                                        init_lambdas='ones', tol=0.001)
    for k in range(10):
        reg.set_params(random_state=k)
        y_pred = reg.fit(X_train, y_train).predict(X_test)
        scores.append(mean_squared_error(y_test, y_pred))

    assert_less_equal(np.std(scores), 0.001)


def check_same_as_slow(degree):

    # XXX: test fails under windows 32bit, presumably numerical issues.
    if ctypes.sizeof(ctypes.c_voidp) < 8:
        raise SkipTest("Numerical inconsistencies on Win32")

    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)

    reg = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_lower=None, fit_linear=False,
                                        beta=1e-8, warm_start=False, tol=1e-3,
                                        max_iter=10, random_state=0)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        reg.fit(X, y)

        P_fit_slow = cd_direct_slow(X, y, lams=reg.lams_, degree=degree,
                                    n_components=5, beta=1e-8, n_iter=10,
                                    tol=1e-3, random_state=0)

    assert_array_almost_equal(reg.P_[0, :, :], P_fit_slow, decimal=4)


def test_same_as_slow():
    yield check_same_as_slow, 2
    yield check_same_as_slow, 3


def check_classification_losses(loss, degree):
    y = np.sign(_poly_predict(X, P, lams, kernel="anova", degree=degree))
    clf = FactorizationMachineClassifier(degree=degree, loss=loss, beta=1e-3,
                                         fit_lower=None, fit_linear=False,
                                         tol=1e-3, random_state=0)
    clf.fit(X, y)
    assert_equal(1.0, clf.score(X, y))


def test_classification_losses():
    for loss in ('squared_hinge', 'logistic'):
        for degree in (2, 3):
            yield check_classification_losses, loss, degree


def check_warm_start(degree):
    y = _poly_predict(X, P, lams, kernel="anova", degree=degree)
    # Result should be the same if:
    # (a) running 10 iterations
    clf_10 = FactorizationMachineRegressor(degree=degree, n_components=5,
                                           fit_lower=None, fit_linear=False,
                                           max_iter=10, warm_start=False,
                                           random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_10.fit(X, y)

    # (b) running 5 iterations and 5 more
    clf_5_5 = FactorizationMachineRegressor(degree=degree, n_components=5,
                                            fit_lower=None, fit_linear=False,
                                            max_iter=5, warm_start=True,
                                            random_state=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5_5.fit(X, y)
        P_fit = clf_5_5.P_.copy()
        lams_fit = clf_5_5.lams_.copy()
        clf_5_5.fit(X, y)

    # (c) running 5 iterations when starting from previous point.
    clf_5 = FactorizationMachineRegressor(degree=degree, n_components=5,
                                          fit_lower=None, fit_linear=False,
                                          max_iter=5, warm_start=True,
                                          random_state=0)
    clf_5.P_ = P_fit
    clf_5.lams_ = lams_fit
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        clf_5.fit(X, y)

    assert_array_almost_equal(clf_10.P_, clf_5_5.P_)
    assert_array_almost_equal(clf_10.P_, clf_5.P_)

    # Prediction results should also be the same if:
    # (note: could not get this test to work for the exact P_.)

    noisy_y = _poly_predict(X, P, lams, kernel="anova", degree=2)
    noisy_y += rng.randn(noisy_y.shape[0])
    X_train, X_test = X[:10], X[10:]
    y_train, y_test = noisy_y[:10], noisy_y[10:]

    beta_low = 0.5
    beta = 0.1
    beta_hi = 1
    ref = FactorizationMachineRegressor(degree=degree, n_components=5,
                                        fit_linear=False, fit_lower=None,
                                        beta=beta, max_iter=20000,
                                        random_state=0)
    ref.fit(X_train, y_train)
    y_pred_ref = ref.predict(X_test)

    # (a) starting from lower beta, increasing and refitting
    from_low = FactorizationMachineRegressor(degree=degree, n_components=5,
                                             fit_lower=None, fit_linear=False,
                                             beta=beta_low, warm_start=True,
                                             random_state=0)
    from_low.fit(X_train, y_train)
    from_low.set_params(beta=beta)
    from_low.fit(X_train, y_train)
    y_pred_low = from_low.predict(X_test)

    # (b) starting from higher beta, decreasing and refitting
    from_hi = FactorizationMachineRegressor(degree=degree, n_components=5,
                                            fit_lower=None, fit_linear=False,
                                            beta=beta_hi, warm_start=True,
                                            random_state=0)
    from_hi.fit(X_train, y_train)
    from_hi.set_params(beta=beta)
    from_hi.fit(X_train, y_train)
    y_pred_hi = from_hi.predict(X_test)

    assert_array_almost_equal(y_pred_low, y_pred_ref, decimal=4)
    assert_array_almost_equal(y_pred_hi, y_pred_ref, decimal=4)


def test_warm_start():
    yield check_warm_start, 2
    yield check_warm_start, 3
