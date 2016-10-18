from __future__ import division

import numpy as np


def rms_log(y_true, y_pred):
    """

    Root Mean Squared Logarithmic Error

    see https://www.kaggle.com/wiki/RootMeanSquaredLogarithmicError

    How to use as scoring method in a scikit learn GridSearchCV:

       scorer = sklearn.metrics.make_scorer(rms_log, greater_is_better=False)
       GridSearchCV(..., scoring=scorer, ...)

    """

    y_pred2 = y_pred.astype(float)    
    
    return np.sqrt( ((np.log(1 + y_true) - np.log(1 + y_pred2))**2 ).sum() / y_true.shape[0] )
    
