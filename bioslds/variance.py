""" Define a class for calculating running estimate of variance. """

import numpy as np

from types import SimpleNamespace
from typing import Sequence, Optional
from numba import njit

from bioslds.monitor import AttributeMonitor


class OnlineVariance(object):
    """ Calculate variance of a set of sequences using an online algorithm.

    This assumes that the input data has zero mean. It uses the following simple update
    rule:
        variance[t + 1] = variance[t] + rate * (y[t] ** 2 - variance[t]) ,
    where `y[t]` is the input signal.

    Attributes
    ==========
    n_features : int
        Number of input dimensions.
    n_components : int
        Number of output dimensions. This is always equal to `n_features`.
    rate : float
        Learning rate.
    negative : bool
        If true, the negative variance is returned.
    output_ : np.ndarray
        Current output, which is the estimate of the variance for each sequence.
    """

    def __init__(self, n_features: int, rate: float = 0.01, negative: bool = False):
        """ Initialize the variance calculator.

        The initial variance is set to 0.

        Parameters
        ----------
        n_features
            Number of input dimensions.
        rate
            Learning rate.
        negative
            If true, the negative of the variance is returned.
        """
        self.n_features = n_features
        self.n_components = n_features
        self.rate = rate
        self.negative = negative

        self.output_ = np.zeros(self.n_features)

        self._mode = "numba"

    # noinspection PyUnusedLocal
    def transform(
        self,
        X: Sequence,
        monitor: Optional[AttributeMonitor] = None,
        chunk_hint: int = None,
    ) -> np.ndarray:
        """ Calculate local variance for a set of sequences.

        Parameters
        ----------
        X : Sequence
            Sequence of input sequences. This has shape `(n_samples, n_features)`.
        monitor
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`).
        chunk_hint
            A hint about how to chunk the learning. This is not currently used.

        Returns an array of local estimates of the variance for each input sequence,
        with shape `(n_samples, n_features)`.
        """
        n = len(X)

        norms = np.zeros((n, self.n_components))

        fct_mapping = {"naive": self._transform_naive, "numba": self._transform_numba}
        fct = fct_mapping[self._mode]

        # noinspection PyArgumentList
        fct(X, norms)

        if self.negative:
            norms = -norms

        self.output_[:] = norms[-1]

        if monitor is not None:
            monitor.setup(n)
            obj = SimpleNamespace(output_=norms)
            monitor.record_batch(obj)

        return norms

    def _transform_naive(
        self, X: Sequence, norms: np.ndarray,
    ):
        """ Perform the transform in pure Python code. """
        n = len(X)
        for i in range(n):
            crt_x = X[i]

            self.output_ += self.rate * (crt_x ** 2 - self.output_)
            norms[i, :] = self.output_

    def _transform_numba(
        self, X: Sequence, norms: np.ndarray,
    ):
        """ Perform the transform using Numba. """
        X = np.asarray(X)

        _perform_numba_transform(X, norms, self.rate, self.output_)

    def __repr__(self) -> str:
        s = f"OnlineVariance(" f"n_features={self.n_features}, rate={self.rate})"
        return s

    _available_modes = ["naive", "numba"]


@njit
def _perform_numba_transform(
    X: np.ndarray, norms: np.ndarray, rate: float, output: np.ndarray
):
    n = len(X)
    for i in range(n):
        crt_x = X[i]

        output += rate * (crt_x ** 2 - output)
        norms[i, :] = output
