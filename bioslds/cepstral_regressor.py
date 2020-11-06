""" Define a class that solves a time-series segmentation task using an oracle-like
method based on the cepstral norm. """

from typing import Sequence, Optional

from bioslds.prediction_error import PredictionError
from bioslds.cepstral import OnlineCepstralNorm
from bioslds.chain import ChainMixin


class CepstralRegressor(ChainMixin):
    def __init__(
        self,
        weights: Sequence,
        cepstral_order: int,
        cepstral_kws: Optional[dict] = None,
    ):
        self.n_models = len(weights)

        self.prediction_error = PredictionError(weights)

        if cepstral_kws is None:
            cepstral_kws = {}
        cepstral_kws.setdefault("negative", True)
        self.cepstral_norm = OnlineCepstralNorm(
            self.n_models, cepstral_order, **cepstral_kws
        )

        super().__init__(["prediction_error", "cepstral_norm"])

    def __repr__(self) -> str:
        s = (
            f"CepstralRegressor("
            f"n_models={self.n_models}, "
            f"n_features={self.n_features}, "
            f"prediction_error={repr(self.prediction_error)}, "
            f"cepstral_norm={repr(self.cepstral_norm)}, "
            f")"
        )
        return s

    def __str__(self) -> str:
        s = (
            f"CepstralRegressor("
            f"n_models={self.n_models}, "
            f"n_features={self.n_features}, "
            f"prediction_error={str(self.prediction_error)}, "
            f"cepstral_norm={str(self.cepstral_norm)}, "
            f")"
        )
        return s
