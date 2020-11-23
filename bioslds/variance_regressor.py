""" Define a class that solves a time-series segmentation task using an oracle-like
method based on an online estimate of variance. """

import numpy as np

from typing import Sequence, Optional, Union

from bioslds.chain import ChainMixin
from bioslds.prediction_error import PredictionError
from bioslds.variance import OnlineVariance


class VarianceRegressor(ChainMixin):
    def __init__(
        self,
        weights: Optional[Sequence] = None,
        n_models: Optional[int] = None,
        n_features: Optional[int] = None,
        rng: Union[int, np.random.Generator] = 0,
        variance_kws: Optional[dict] = None,
    ):
        """ Initialize the variance-based regressor.

        This chains together a `PredictionError` instance with an `OnlineVariance`.

        Parameters
        ----------
        weights
           Regression coefficients for each model, shape `(n_models, n_features)`. If
           this is provided, `n_models` and `n_features` need not be provided. If
           `weights` is not provided, it will be initialized randomly, using a standard
           normal distribution. See `rng`.
        n_models
           Number of models. If `weights` is provided, `n_models` need not be. However,
           if both are provided, then `n_models` should be equal to the number of rows
           in `weights`.
        n_features
           Number of predictor variables (features). If `weights` is provided,
           `n_features` need not be. However, if both are provided, then `n_features`
           should be equal to the number of columns in `weights`.
        rng
           Random number generator or seed to use for generating initial weight
           values. If seed, a random number generator is created using
           `np.random.default_rng`. If not provided, a seed of 0 is used. The random
           number generator is not used at all if `weights` is provided.
        variance_kws
           Additional keywords to pass to `OnlineVariance`. By default `negative` is set
           to true.
        """
        if weights is None:
            if not hasattr(rng, "normal"):
                rng = np.random.default_rng(rng)
            weights = rng.normal(size=(n_models, n_features))
        else:
            if n_models is not None and n_models != np.shape(weights)[0]:
                raise ValueError("n_models does not match shape(weights)[0]")
            if n_features is not None and n_features != np.shape(weights)[1]:
                raise ValueError("n_features does not match shape(weights)[1]")

        self.n_models = len(weights)

        self.prediction_error = PredictionError(weights)

        if variance_kws is None:
            variance_kws = {}
        variance_kws.setdefault("negative", True)
        self.variance_norm = OnlineVariance(self.n_models, **variance_kws)

        super().__init__(["prediction_error", "variance_norm"])

    def __repr__(self) -> str:
        s = (
            f"VarianceRegressor("
            f"n_models={self.n_models}, "
            f"n_features={self.n_features}, "
            f"prediction_error={repr(self.prediction_error)}, "
            f"variance_norm={repr(self.variance_norm)}, "
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"VarianceRegressor("
            f"n_models={self.n_models}, "
            f"n_features={self.n_features}, "
            f"prediction_error={str(self.prediction_error)}, "
            f"variance_norm={str(self.variance_norm)}, "
            f")"
        )
        return s
