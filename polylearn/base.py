# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

from abc import ABCMeta
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.preprocessing import LabelBinarizer, add_dummy_feature
from sklearn.utils.validation import check_X_y
from sklearn.utils.multiclass import type_of_target
from sklearn.externals import six

try:
    from sklearn.exceptions import NotFittedError
except ImportError:
    class NotFittedError(ValueError, AttributeError):
        pass

from .loss import CLASSIFICATION_LOSSES, REGRESSION_LOSSES


class _BasePoly(six.with_metaclass(ABCMeta, BaseEstimator)):

    def _augment(self, X):
        if self.fit_lower == 'augment':
            k = 1 if self.fit_linear == 'augment' else 2
            for _ in range(self.degree - k):
                X = add_dummy_feature(X, value=1)
        return X

    def _get_loss(self, loss):
        # classification losses
        if loss not in self._LOSSES:
            raise ValueError(
                'Loss function "{}" not supported. The available options '
                'are: "{}".'.format(loss,
                                    '", "'.join(self._LOSSES)))
        return self._LOSSES[loss]


class _PolyRegressorMixin(RegressorMixin):

    _LOSSES = REGRESSION_LOSSES

    def _check_X_y(self, X, y):
        X, y = check_X_y(X, y, accept_sparse='csc', multi_output=False,
                         dtype=np.double, y_numeric=True)
        y = y.astype(np.double).ravel()
        return X, y

    def predict(self, X):
        """Predict regression output for the samples in X.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        return self._predict(X)


class _PolyClassifierMixin(ClassifierMixin):

    _LOSSES = CLASSIFICATION_LOSSES

    def decision_function(self, X):
        """Compute the output of the factorization machine before thresholding.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_scores : array, shape = [n_samples]
            Returns predicted values.
        """
        return self._predict(X)

    def predict(self, X):
        """Predict using the factorization machine

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_pred : array, shape = [n_samples]
            Returns predicted values.
        """
        y_pred = self.decision_function(X) > 0
        return self.label_binarizer_.inverse_transform(y_pred)

    def predict_proba(self, X):
        """Compute probability estimates for the test samples.

        Only available if `loss='logistic'`.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Samples.

        Returns
        -------
        y_scores : array, shape = [n_samples]
            Probability estimates that the samples are from the positive class.
        """
        if self.loss == 'logistic':
            return 1 / (1 + np.exp(-self.decision_function(X)))
        else:
            raise ValueError("Probability estimates only available for "
                             "loss='logistic'. You may use probability "
                             "calibration methods from scikit-learn instead.")

    def _check_X_y(self, X, y):
        if type_of_target(y) != 'binary':
            raise TypeError("Only binary targets supported. For training "
                            "multiclass or multilabel models, you may use the "
                            "OneVsRest or OneVsAll metaestimators in "
                            "scikit-learn.")

        X, Y = check_X_y(X, y, dtype=np.double, accept_sparse='csc',
                         multi_output=False)

        self.label_binarizer_ = LabelBinarizer(pos_label=1, neg_label=-1)
        y = self.label_binarizer_.fit_transform(Y).ravel().astype(np.double)
        return X, y
