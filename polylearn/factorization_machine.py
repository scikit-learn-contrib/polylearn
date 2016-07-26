# encoding: utf-8

# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.preprocessing import add_dummy_feature
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import safe_sparse_dot, row_norms
from sklearn.externals import six

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        pass

from lightning.impl.dataset_fast import get_dataset

from .base import _BasePoly, _PolyClassifierMixin, _PolyRegressorMixin
from .kernels import _poly_predict
from .cd_direct_fast import _cd_direct_ho


class _BaseFactorizationMachine(six.with_metaclass(ABCMeta, _BasePoly)):

    @abstractmethod
    def __init__(self, degree=2, loss='squared', n_components=2, alpha=1,
                 beta=1, tol=1e-6, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=10000,
                 verbose=False, random_state=None):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.alpha = alpha
        self.beta = beta
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.init_lambdas = init_lambdas
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = random_state

    def _augment(self, X):
        # for factorization machines, we add a dummy column for each order.

        if self.fit_lower == 'augment':
            k = 2 if self.fit_linear else 1
            for _ in range(self.degree - k):
                X = add_dummy_feature(X, value=1)
        return X

    def fit(self, X, y):
        """Fit factorization machine to training data.

        Parameters
        ----------
        X : array-like or sparse, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : Estimator
            Returns self.
        """
        if self.degree > 3:
            raise ValueError("FMs with degree >3 not yet supported.")

        X, y = self._check_X_y(X, y)
        X = self._augment(X)
        n_features = X.shape[1]  # augmented
        X_col_norms = row_norms(X.T, squared=True)
        dataset = get_dataset(X, order="fortran")
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)

        if not (self.warm_start and hasattr(self, 'w_')):
            self.w_ = np.zeros(n_features, dtype=np.double)

        if self.fit_lower == 'explicit':
            n_orders = self.degree - 1
        else:
            n_orders = 1

        if not (self.warm_start and hasattr(self, 'P_')):
            self.P_ = 0.01 * rng.randn(n_orders, self.n_components, n_features)

        if not (self.warm_start and hasattr(self, 'lams_')):
            if self.init_lambdas == 'ones':
                self.lams_ = np.ones(self.n_components)
            elif self.init_lambdas == 'random_signs':
                self.lams_ = np.sign(rng.randn(self.n_components))
            else:
                raise ValueError("Lambdas must be initialized as ones "
                                 "(init_lambdas='ones') or as random "
                                 "+/- 1 (init_lambdas='random_signs').")

        y_pred = self._get_output(X)

        converged = _cd_direct_ho(self.P_, self.w_, dataset, X_col_norms, y,
                                  y_pred, self.lams_, self.degree, self.alpha,
                                  self.beta, self.fit_linear,
                                  self.fit_lower == 'explicit', loss_obj,
                                  self.max_iter, self.tol, self.verbose)
        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _get_output(self, X):
        y_pred = _poly_predict(X, self.P_[0, :, :], self.lams_, kernel='anova',
                               degree=self.degree)

        if self.fit_linear:
            y_pred += safe_sparse_dot(X, self.w_)

        if self.fit_lower == 'explicit' and self.degree == 3:
            # degree cannot currently be > 3
            y_pred += _poly_predict(X, self.P_[1, :, :], self.lams_,
                                    kernel='anova', degree=2)

        return y_pred

    def _predict(self, X):
        if not hasattr(self, "P_"):
            raise NotFittedError("Estimator not fitted.")
        X = check_array(X, accept_sparse='csc', dtype=np.double)
        X = self._augment(X)
        return self._get_output(X)


class FactorizationMachineRegressor(_BaseFactorizationMachine,
                                    _PolyRegressorMixin):
    """Factorization machine for regression (with squared loss).

    Parameters
    ----------

    degree : int >= 2, default: 2
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model. Currently only supports
        degrees up to 3.

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    alpha : float, default: 1
        Regularization amount for linear term (if ``fit_linear=True``).

    beta : float, default: 1
        Regularization amount for higher-order weights.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'explicit': fits a separate P directly for each lower order.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.

        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

    fit_linear : {True|False}, default: True
        Whether to fit an explicit linear term <w, x> to the model, using
        coordinate descent. If False, the model can still capture linear
        effects if ``fit_lower == 'augment'``.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.

        ``self.P_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.

        ``self.P_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear model, completing the FM.

        Only present if ``self.fit_linear`` is true.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    Factorization machines.
    Steffen Rendle
    In: Proceedings of IEEE 2010.
    """
    def __init__(self, degree=2, n_components=2, alpha=1, beta=1, tol=1e-6,
                 fit_lower='explicit', fit_linear=True, warm_start=False,
                 init_lambdas='ones', max_iter=10000, verbose=False,
                 random_state=None):

        super(FactorizationMachineRegressor, self).__init__(
            degree, 'squared', n_components, alpha, beta, tol, fit_lower,
            fit_linear, warm_start, init_lambdas, max_iter, verbose,
            random_state)


class FactorizationMachineClassifier(_BaseFactorizationMachine,
                                     _PolyClassifierMixin):
    """Factorization machine for classification.

    Parameters
    ----------

    degree : int >= 2, default: 2
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model. Currently only supports
        degrees up to 3.

    loss : {'logistic'|'squared_hinge'|'squared'}, default: 'squared_hinge'
        Which loss function to use.

        - logistic: L(y, p) = log(1 + exp(-yp))

        - squared hinge: L(y, p) = max(1 - yp, 0)²

        - squared: L(y, p) = 0.5 * (y - p)²

    n_components : int, default: 2
        Number of basis vectors to learn, a.k.a. the dimension of the
        low-rank parametrization.

    alpha : float, default: 1
        Regularization amount for linear term (if ``fit_linear=True``).

    beta : float, default: 1
        Regularization amount for higher-order weights.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'explicit'|'augment'|None}, default: 'explicit'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'explicit': fits a separate P directly for each lower order.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.
           Adds ``degree - 2`` columns if ``fit_linear`` is true, or
           ``degree - 1`` columns otherwise, to account for the linear term.

        - None: only learns weights for the degree given.  If ``degree == 3``,
          for example, the model will only have weights for third-order
          feature interactions.

    fit_linear : {True|False}, default: True
        Whether to fit an explicit linear term <w, x> to the model, using
        coordinate descent. If False, the model can still capture linear
        effects if ``fit_lower == 'augment'``.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    init_lambdas : {'ones'|'random_signs'}, default: 'ones'
        How to initialize the predictive weights of each learned basis. The
        lambdas are not trained; using alternate signs can theoretically
        improve performance if the kernel degree is even.  The default value
        of 'ones' matches the original formulation of factorization machines
        (Rendle, 2010).

        To use custom values for the lambdas, ``warm_start`` may be used.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.P_ : array, shape [n_orders, n_components, n_features]
        The learned basis functions.

        ``self.P_[0, :, :]`` is always available, and corresponds to
        interactions of order ``self.degree``.

        ``self.P_[i, :, :]`` for i > 0 corresponds to interactions of order
        ``self.degree - i``, available only if ``self.fit_lower='explicit'``.

    self.w_ : array, shape [n_features]
        The learned linear model, completing the FM.

        Only present if ``self.fit_linear`` is true.

    self.lams_ : array, shape [n_components]
        The predictive weights.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    Factorization machines.
    Steffen Rendle
    In: Proceedings of IEEE 2010.
    """

    def __init__(self, degree=2, loss='squared_hinge', n_components=2, alpha=1,
                 beta=1, tol=1e-6, fit_lower='explicit', fit_linear=True,
                 warm_start=False, init_lambdas='ones', max_iter=10000,
                 verbose=False, random_state=None):

        super(FactorizationMachineClassifier, self).__init__(
            degree, loss, n_components, alpha, beta, tol, fit_lower,
            fit_linear, warm_start, init_lambdas, max_iter, verbose,
            random_state)
