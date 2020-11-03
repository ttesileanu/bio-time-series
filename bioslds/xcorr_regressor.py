""" Define a class for implementing a biologically plausible, online, correlation-based
algorithm for segmenting a time series based on the generating linear process.
"""

import copy

import numpy as np

from bioslds.xcorr import OnlineCrosscorrelation
from bioslds.nsm import NonRecurrent
from bioslds.monitor import AttributeMonitor

from types import SimpleNamespace
from typing import Union, Optional, Sequence, Callable, Tuple


class CrosscorrelationRegressor(object):
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

        if nsm_kws is None:
            nsm_kws = {}
        else:
            nsm_kws = copy.copy(nsm_kws)
        nsm_kws.setdefault("rng", rng)
        nsm_kws.setdefault("non_negative", True)
        nsm_kws.setdefault("rate", nsm_rate)
        self.nsm = NonRecurrent(self.n_features, self.n_models, **nsm_kws)

        self.xcorr = OnlineCrosscorrelation(self.n_features, rate=xcorr_rate)

    def transform(
        self,
        X: Sequence,
        y: Sequence,
        progress: Optional[Callable] = None,
        monitor: Union[None, AttributeMonitor, Sequence] = None,
        return_history: bool = False,
        chunk_hint: int = 1000,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SimpleNamespace]]:
        """ Feed a set of samples through the model, inferring which model fits best at
        each time step.

        Parameters
        ----------
        X
            The values of the predictor variables. Shape `(n_samples, n_features)`.
        y
            The values of the dependent variable. Shape `(n_samples,)`.
        progress
            Progress function that can be used either as a wrapper or manually, like
            `tqdm.tqdm`. Specifically, this needs to support wrapping an iterable, so
            you can write, e.g., `for x in progress(X): ...`; and it needs to support
            calling with a `total` argument, an `update`, and a `close` method, e.g.:
                pbar = progress(total=100)
                for i in range(100):
                    pbar.update(1)  # note that arg is step (=1), not i!
                pbar.close()
        monitor
            This can be an object for monitoring the evolution of the parameters during
            learning (e.g., an instance of `AttributeMonitor`), or a sequence of
            attribute names indicating the parameters to be tracked. If the latter, the
            function generates an `AttributeMonitor`, and `return_history` is forced to
            be true so that its `history_` attribute can be accessed. Parameter values
            are stored before their updates.
        return_history
            If true, return the monitor's `history_` namespace as a second return value.
            If `monitor` is `None`, returns an empty namespace.
        chunk_hint
            A hint about how to chunk the learning. This may or may not be used. If it
            is, the progress function will only be called once per chunk.

        If `return_history` is not false, returns an array `r` with shape `(n_samples,
        n_models)` such  that `r[t, k]` shows, roughly, how likely it is that the sample
        at time `t` was generated by model `k`. Note that this is not a full
        probabilistic model, so this should not be taken literally  -- in particular,
        each row of `r` does not, in general, sum to 1.

        If `return_history` is true, returns a tuple `(r, history)` where `r` is the
        output from above, and `history` is the `history_` attribute of the monitor that
        was used.
        """
        if monitor is None and progress is None:
            xcorr = self.xcorr.transform(X, y, chunk_hint=chunk_hint)
            out = self.nsm.transform(xcorr, chunk_hint=chunk_hint)
            return out
        else:
            # handle progress function
            n = len(y)
            if progress is not None:
                pbar = progress(total=n)
            else:
                pbar = None

            if monitor is not None:
                if hasattr(monitor, "__len__") and not hasattr(monitor, "setup"):
                    monitor = AttributeMonitor(monitor)
                    # sequence implies return_history
                    return_history = True
                monitor.setup(n)
            else:
                return_history = False

            out = np.zeros((n, self.n_models))
            for chunk_start in range(0, n, chunk_hint):
                crt_range = slice(chunk_start, chunk_start + chunk_hint)
                crt_x = X[crt_range]
                crt_y = y[crt_range]
                crt_out = out[crt_range]

                if monitor is not None:
                    xcorr_monitor = AttributeMonitor(
                        [_[6:] for _ in monitor.names if _.startswith("xcorr.")]
                    )
                    nsm_monitor = AttributeMonitor(
                        [_[4:] for _ in monitor.names if _.startswith("nsm.")]
                    )
                else:
                    xcorr_monitor = None
                    nsm_monitor = None

                xcorr = self.xcorr.transform(
                    crt_x, crt_y, monitor=xcorr_monitor, chunk_hint=chunk_hint
                )
                crt_out[:] = self.nsm.transform(xcorr, monitor=nsm_monitor,
                                                chunk_hint=chunk_hint)

                if monitor is not None:
                    # combine
                    obj = SimpleNamespace(
                        xcorr=xcorr_monitor.history_, nsm=nsm_monitor.history_
                    )
                    monitor.record_batch(obj)

                if pbar is not None:
                    pbar.update(len(crt_x))

            if pbar is not None:
                pbar.close()

            if not return_history:
                return out
            else:
                return out, monitor.history_
