""" Define a transformer class that calculates the local cross-correlation between a
scalar and a vector signal in an online fashion. """

import numpy as np

from types import SimpleNamespace
from typing import Sequence, Callable, Optional
from numba import njit

from bioslds.monitor import AttributeMonitor


class OnlineCrosscorrelation(object):
    """ Keep track of local cross-correlation profile of a pair of signals.

    Attributes
    ----------
    n_components : int
        Number of components in the output. This is equal to the number of non-zero lags
        at which to estimate the cross-correlation.
    rate : float
        Learning rate for the cross-correlation estimate.
    var_ : float
        Running estimate of the variance of the 1d series.
    coef_ : np.ndarray
        Running estimate of the cross-correlation at a number of lags.
    """

    def __init__(
        self, n_components: int, rate: float = 0.1,
    ):
        """ Initialize the cross-correlation learner.

        Parameters
        ----------
        n_components
            Number of components in the output. This is equal to the number of non-zero
            lags at which the cross-correlation is estimated.
        rate
            Learning rate for the cross-correlation estimate.
        """
        self.n_components = n_components
        self.rate = rate

        # initialize state variables
        self.var_ = 1.0
        self.coef_ = np.zeros(self.n_components)

        self._mode = "numba"

    def transform(
        self,
        X: Sequence,
        y: Sequence,
        monitor: Optional[AttributeMonitor] = None,
        progress: Optional[Callable] = None,
        chunk_hint: int = 1000,
    ) -> np.ndarray:
        """ Calculate local cross-correlation for a pair of sequences.

        Parameters
        ----------
        X : Sequence
            Sequence of higher-dimensional inputs. This has shape `(n_samples,
            n_components)`.
        y : Sequence
            Sequence of one-dimensional inputs, of length `n_samples`.
        monitor
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`). Parameter values are stored and
            calculated before their updates.
        progress
            Progress function that can be used either as a wrapper or manually, like
            `tqdm.tqdm`. Specifically, this needs to support wrapping an iterable, so
            you can write, e.g., `for x in progress(X): ...`; and it needs to support
            calling with a `total` argument, an `update`, and a `close` method, e.g.:
                pbar = progress(total=100)
                for i in range(100):
                    pbar.update(1)  # note that arg is step (=1), not i!
                pbar.close()
        chunk_hint
            A hint about how to chunk the learning. This may or may not be used. If it
            is, the progress function will only be called once per chunk.

        Returns an array of local estimates of the autocorrelation, with shape
        `(n_samples, n_components)`.
        """
        # set up the attribute monitor, if any
        n = len(y)
        if monitor is not None:
            monitor.setup(n)

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)

        fct_mapping = {
            "naive": self._transform_naive,
            "numba": self._transform_numba,
        }
        fct = fct_mapping[self._mode]

        # noinspection PyArgumentList
        return fct(X, y, progress=progress, monitor=monitor, chunk_hint=chunk_hint)

    # noinspection PyUnusedLocal
    def _transform_naive(
        self,
        X: np.ndarray,
        y: np.ndarray,
        monitor: Optional[AttributeMonitor],
        progress: Optional[Callable],
        chunk_hint: int,
    ) -> np.ndarray:
        # set up the progress bar, if any
        if progress is not None:
            it_y = progress(y)
        else:
            it_y = y

        # run the circuit
        n = len(y)
        coef_history = np.zeros((n, self.n_components))
        for i, crt_y in enumerate(it_y):
            # record the current state using the attribute monitor, if it exists
            if monitor is not None:
                monitor.record(self)
            coef_history[i, :] = self.coef_

            crt_x = X[i]

            # update the cross-correlation
            self.coef_ += self.rate * ((crt_y / self.var_) * crt_x - self.coef_)

            # update the variance
            self.var_ += self.rate * (crt_y ** 2 - self.var_)

        return coef_history

    def _transform_numba(
        self,
        X: np.ndarray,
        y: np.ndarray,
        monitor: Optional[AttributeMonitor],
        progress: Optional[Callable],
        chunk_hint: int,
    ) -> np.ndarray:
        if chunk_hint < 1:
            chunk_hint = 1

        # handle progress function
        n = len(X)
        if progress is not None:
            pbar = progress(total=n)
        else:
            pbar = None

        # set up monitor, if any
        if monitor is not None:
            monitor.setup(n)

        coef_history = np.zeros((n, self.n_components))
        for chunk_start in range(0, n, chunk_hint):
            crt_range = slice(chunk_start, chunk_start + chunk_hint)
            crt_X = X[crt_range]
            crt_y = y[crt_range]
            crt_n = len(crt_X)

            crt_var_hist = np.zeros(crt_n)

            final_var = _perform_transform(
                crt_X,
                crt_y,
                self.rate,
                self.var_,
                self.coef_,
                crt_var_hist,
                coef_history[crt_range],
            )
            self.var_ = final_var

            if pbar is not None:
                pbar.update(crt_n)
            if monitor is not None:
                crt_history = SimpleNamespace(
                    var_=crt_var_hist, coef_=coef_history[crt_range],
                )
                monitor.record_batch(crt_history)

        if pbar is not None:
            pbar.close()

        return coef_history

    def __repr__(self):
        return (
            "OnlineCrosscorrelation(n_components={}, rate={}, " "var_={}, coef_={})"
        ).format(self.n_components, self.rate, self.var_, self.coef_)

    def __str__(self):
        return "OnlineCrosscorrelation(n_components={}, rate={})".format(
            self.n_components, self.rate
        )

    _available_modes = ["naive", "numba"]


@njit
def _perform_transform(
    X: np.ndarray,
    y: np.ndarray,
    rate: float,
    var: float,
    coef: np.ndarray,
    var_history: np.ndarray,
    coef_history: np.ndarray,
) -> float:
    # run the circuit
    n = len(y)
    for i in range(n):
        var_history[i] = var
        coef_history[i, :] = coef

        crt_x = X[i]
        crt_y = y[i]

        coef_history[i, :] = coef

        # update the cross-correlation
        coef += rate * ((crt_y / var) * crt_x - coef)

        # update the variance
        var += rate * (crt_y ** 2 - var)

    return var
