""" Define a class that calculates the prediction error against a set of linear models.
"""

import numpy as np

from types import SimpleNamespace
from typing import Sequence, Optional, Union, Tuple

from bioslds.monitor import AttributeMonitor


class PredictionError(object):
    """ Calculate the prediction error against a set of linear models.

    Consider a one-dimensional dependent variable `y[t]` that we are linearly
    approximating from a higher-dimensional predictor `x[t]` using a set of weights:
        y[t] = np.dot(w[k, :], x[t]) + eps[t, k] ,
    where `k` indexes the model used, and `w[k, :]` are the coefficients of the `k`th
    model.

    This class simply calculates and returns the prediction errors `eps[k, t]`.

    Attributes
    ==========
    n_models : int
        Number of models in mixture.
    n_features : int
        Number of predictor variables (features).
    n_components : int
        Number of output dimensions. This is an alias for `n_models`.
    weights_ : array, shape `(n_models, n_features)`
        Regression weights for each of the models.
    output_ : array of float, shape (n_models, )
        Last calculated prediction errors. This corresponds to the last value returned
        from `transform`.
    """

    def __init__(
        self, weights: Sequence,
    ):
        """ Initialize the prediction error calculator.

        Parameters
        ----------
        weights
            Regression coefficients for each model, shape `(n_models, n_features)`.
        """
        self.weights_ = np.array(weights)

        self.n_models, self.n_features = np.shape(self.weights_)
        self.n_components = self.n_models

        self.output_ = np.zeros(self.n_models)

    # noinspection PyUnusedLocal
    def transform(
        self,
        X: Sequence,
        y: Sequence,
        monitor: Optional[AttributeMonitor] = None,
        return_history: bool = False,
        chunk_hint: int = 1000,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SimpleNamespace]]:
        """ Find the prediction errors for a set of samples.

        Parameters
        ----------
        X
            The values of the predictor variables. Shape `(n_samples, n_features)`.
        y
            The values of the dependent variable. Shape `(n_samples,)`.
        monitor
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`).
        return_history
            If true, return the monitor's `history_` namespace as a second return value.
            If `monitor` is `None`, returns an empty namespace.
        chunk_hint
            A hint about how to chunk the learning. This may or may not be used.

        If `return_history` is false, returns an array `eps` with shape `(n_samples,
        n_models)` such  that `eps[t, k]` gives the prediction error from model `k` at
        time `t`.

        If `return_history` is true, returns a tuple `(eps, history)` where `eps` is the
        output from above, and `history` is the `history_` attribute of the monitor that
        was used.
        """
        # do the calculation
        res = np.asarray(y)[:, None] - np.dot(X, self.weights_.T)
        self.output_[:] = res[-1]

        # handle monitor
        if monitor is not None:
            monitor.setup(len(res))
            monitor.record_batch(SimpleNamespace(output_=res))
        else:
            return_history = False

        if not return_history:
            return res
        else:
            return res, monitor.history_
