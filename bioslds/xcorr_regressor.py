""" Define a class for implementing a biologically plausible, online, correlation-based
algorithm for segmenting a time series based on the generating linear process.
"""

import copy

import numpy as np

from bioslds.xcorr import OnlineCrosscorrelation
from bioslds.nsm import NonRecurrent
from bioslds.chain import ChainMixin

from typing import Union, Optional, Sequence, Callable


class CrosscorrelationRegressor(ChainMixin):
    """ A class implementing a biologically plausible, correlation-based algorithm that
    segments a multi-modal time series in a streaming setting.

    Specifically, consider an output (dependent) signal `y` given by a time-dependent
    linear function of an input (predictor) signal `x`,
        y[t] = sum_k z[t][k] * np.dot(w[k, :], x[t]) ,
    where `z[t]` is a one-hot encoding of the model used at time `t` (`z[t][k] == 1` iff
    model `k` is used, 0 else).

    This class attempts to infer the state assignments `z[t]` by clustering the recent
    cross-correlation profile between `x` and `y`. This is done in an online fashion,
    and it uses an implementation that has a simple interpretation in terms of a
    rate-based biologically plausible neural network.

    Attributes
    ==========
    n_models : int
        Number of models in mixture.
    n_features : int
        Number of predictor variables (features).
    n_components : int
        Number of output dimensions. This is an alias for `n_models`.
    nsm : NonRecurrent
        Non-negative similarity matching component.
    xcorr : OnlineCrosscorrelation
        Component for estimating local cross-correlation.
    """

    def __init__(
        self,
        n_models: int,
        n_features: int,
        nsm_rate: Union[float, Sequence, Callable[[float], float]] = 1e-3,
        xcorr_rate: float = 0.05,
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        nsm_kws: Optional[dict] = None,
    ):
        """ Initialize the segmentation model.

        Parameters
        ----------
        n_models
            Number of models in mixture.
        n_features
            Number of predictor variables (features).
        nsm_rate
            Learning rate or learning schedule for the non-negative similarity matching
            (NSM) algorithm. See `bioslds.nsm.NonRecurrent`.
        xcorr_rate
            Learning rate for the cross-correlation calculator. See
            `bioslds.xcorr.OnlineCrosscorrelation`.
        rng
            Random number generator or seed to use for generating initial NSM weight
            values. This is simply passed to `bioslds.nsm.NonRecurrent`.
        nsm_kws
            Additional keyword arguments to pass to `bioslds.nsm.NonRecurrent.__init__`.
        """
        self.n_models = n_models
        self.n_features = n_features
        self.n_components = self.n_models

        self.xcorr = OnlineCrosscorrelation(self.n_features, rate=xcorr_rate)

        if nsm_kws is None:
            nsm_kws = {}
        else:
            nsm_kws = copy.copy(nsm_kws)
        nsm_kws.setdefault("rng", rng)
        nsm_kws.setdefault("non_negative", True)
        nsm_kws.setdefault("rate", nsm_rate)
        self.nsm = NonRecurrent(self.n_features, self.n_models, **nsm_kws)

        super().__init__(["xcorr", "nsm"])
