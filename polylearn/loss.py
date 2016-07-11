# Author: Vlad Niculae <vlad@vene.ro>
# License: Simplified BSD

from .loss_fast import Squared, SquaredHinge, Logistic


REGRESSION_LOSSES = {
    'squared': Squared()
}

CLASSIFICATION_LOSSES = {
    'squared': Squared(),
    'squared_hinge': SquaredHinge(),
    'logistic': Logistic()
}
