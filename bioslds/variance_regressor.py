""" Define a class that solves a time-series segmentation task using an oracle-like
method based on an online estimate of variance. """


from typing import Sequence, Optional

from bioslds.chain import ChainMixin
from bioslds.prediction_error import PredictionError
from bioslds.variance import OnlineVariance


class VarianceRegressor(ChainMixin):
    def __init__(
        self, weights: Sequence, variance_kws: Optional[dict] = None,
    ):
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
