"""Polynomial networks for regression and classification."""

# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

import warnings
from abc import ABCMeta, abstractmethod

import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_array
from sklearn.externals import six

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        pass

from lightning.impl.dataset_fast import get_dataset

from .base import _BasePoly, _PolyClassifierMixin, _PolyRegressorMixin
from .cd_lifted_fast import _cd_lifted


class _BasePolynomialNetwork(six.with_metaclass(ABCMeta, _BasePoly)):
    @abstractmethod
    def __init__(self, degree=2, loss='squared', n_components=5, beta=1,
                 tol=1e-6, fit_lower='augment', fit_linear='augment',
                 warm_start=False, max_iter=10000, verbose=False,
                 compute_loss=False, random_state=None):
        self.degree = degree
        self.loss = loss
        self.n_components = n_components
        self.beta = beta
        self.tol = tol
        self.fit_lower = fit_lower
        self.fit_linear = fit_linear
        self.warm_start = warm_start
        self.max_iter = max_iter
        self.verbose = verbose
        self.compute_loss = compute_loss
        self.random_state = random_state

    def fit(self, X, y):
        """Fit polynomial network to training data.

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
        if self.fit_lower == 'explicit':
            raise NotImplementedError('Explicit fitting of lower orders '
                                      'not yet implemented for polynomial'
                                      'network models.')

        if self.fit_linear == 'explicit':
            raise NotImplementedError('Explicit fitting of linear component '
                                      'not yet implemented for polynomial'
                                      'network models.')

        X, y = self._check_X_y(X, y)
        X = self._augment(X)
        n_features = X.shape[1]  # augmented
        dataset = get_dataset(X, order="fortran")
        rng = check_random_state(self.random_state)
        loss_obj = self._get_loss(self.loss)

        if not (self.warm_start and hasattr(self, 'U_')):
            self.U_ = rng.randn(self.degree, self.n_components, n_features)

        y_pred = np.product(safe_sparse_dot(self.U_, X.T), axis=0).sum(axis=0)

        converged = _cd_lifted(self.U_, dataset, y, y_pred, self.beta,
                               loss_obj, self.max_iter, self.tol, self.verbose,
                               self.compute_loss)

        if not converged:
            warnings.warn("Objective did not converge. Increase max_iter.")

        return self

    def _predict(self, X):
        if not hasattr(self, "U_"):
            raise NotFittedError("Estimator not fitted.")

        X = check_array(X, accept_sparse='csc', dtype=np.double)
        X = self._augment(X)

        return np.product(safe_sparse_dot(self.U_, X.T), axis=0).sum(axis=0)


class PolynomialNetworkRegressor(_BasePolynomialNetwork, _PolyRegressorMixin):
    """Polynomial network for regression (with squared loss).

    Parameters
    ----------

    degree : int >= 2, default: 2
        Degree of the polynomial. Corresponds to the order of feature
        interactions captured by the model. Currently only supports
        degrees up to 3.

    n_components : int, default: 2
        Dimension of the lifted tensor.

    beta : float, default: 1
        Regularization amount for higher-order weights.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'augment'|None}, default: 'augment'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.

        - None: only learns weights for the degree given.

    fit_lower : {'augment'|None}, default: 'augment'
        Whether and how to fit a linear term to the model.

        - 'augment': adds an extra dummy column to account for the linear
          term. This only works if ``fit_lower='augment'``.

        - None: does not learn a linear term at all.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    compute_loss : boolean, optional, default: False
        Whether to compute the training loss at every iteration. Slows down
        runtime but may be useful for debugging convergence issues.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.U_ : array, shape [n_components, n_features, degree]
        The learned weights in the lifted tensor parametrization.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    On the computational efficiency of training neural networks.
    Roi Livni, Shai Shalev-Shwartz, Ohad Shamir.
    In: Proceedings of NIPS 2014.
    """

    def __init__(self, degree=2, n_components=2, beta=1, tol=1e-6,
                 fit_lower='augment', fit_linear='augment', warm_start=False,
                 max_iter=10000, verbose=False, compute_loss=False,
                 random_state=None):

        super(PolynomialNetworkRegressor, self).__init__(
            degree, 'squared', n_components, beta, tol, fit_lower, fit_linear,
            warm_start, max_iter, verbose, compute_loss, random_state)


class PolynomialNetworkClassifier(_BasePolynomialNetwork,
                                  _PolyClassifierMixin):
    """Polynomial network for classification.

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
        Dimension of the lifted tensor.

    beta : float, default: 1
        Regularization amount for higher-order weights.

    tol : float, default: 1e-6
        Tolerance for the stopping condition.

    fit_lower : {'augment'|None}, default: 'augment'
        Whether and how to fit lower-order, non-homogeneous terms.

        - 'augment': adds the required number of dummy columns (columns
           that are 1 everywhere) in order to capture lower-order terms.

        - None: only learns weights for the degree given.

    fit_lower : {'augment'|None}, default: 'augment'
        Whether and how to fit a linear term to the model.

        - 'augment': adds an extra dummy column to account for the linear
          term. This only works if ``fit_lower='augment'``.

        - None: does not learn a linear term at all.

    warm_start : boolean, optional, default: False
        Whether to use the existing solution, if available. Useful for
        computing regularization paths or pre-initializing the model.

    max_iter : int, optional, default: 10000
        Maximum number of passes over the dataset to perform.

    verbose : boolean, optional, default: False
        Whether to print debugging information.

    compute_loss : boolean, optional, default: False
        Whether to compute the training loss at every iteration. Slows down
        runtime but may be useful for debugging convergence issues.

    random_state : int seed, RandomState instance, or None (default)
        The seed of the pseudo random number generator to use for
        initializing the parameters.

    Attributes
    ----------

    self.U_ : array, shape [n_components, n_features, degree]
        The learned weights in the lifted tensor parametrization.

    References
    ----------
    Polynomial Networks and Factorization Machines:
    New Insights and Efficient Training Algorithms.
    Mathieu Blondel, Masakazu Ishihata, Akinori Fujino, Naonori Ueda.
    In: Proceedings of ICML 2016.
    http://mblondel.org/publications/mblondel-icml2016.pdf

    On the computational efficiency of training neural networks.
    Roi Livni, Shai Shalev-Shwartz, Ohad Shamir.
    In: Proceedings of NIPS 2014.
    """

    def __init__(self, degree=2, loss='squared_hinge', n_components=2, beta=1,
                 tol=1e-6, fit_lower='augment', fit_linear='augment',
                 warm_start=False, max_iter=10000, verbose=False,
                 compute_loss=False, random_state=None):

        super(PolynomialNetworkClassifier, self).__init__(
            degree, loss, n_components, beta, tol, fit_lower, fit_linear,
            warm_start, max_iter, verbose, compute_loss, random_state)
