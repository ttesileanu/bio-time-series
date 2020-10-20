""" Define a transformer class that calculates the local cross-correlation between a
scalar and a vector signal in an online fashion. """

import numpy as np

from typing import Sequence, Callable

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

        self._mode = "naive"

    def transform(
        self,
        X: Sequence,
        y: Sequence,
        monitor: AttributeMonitor = None,
        progress: Callable = None,
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

        Returns an array of local estimates of the autocorrelation, with shape
        `(n_samples, n_components)`.
        """
        # set up the attribute monitor, if any
        n = len(y)
        if monitor is not None:
            monitor.setup(n)

        X = np.asarray(X)

        # set up the progress bar, if any
        if progress is not None:
            it_y = progress(y)
        else:
            it_y = y

        # run the circuit
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

    def __repr__(self):
        return (
            "OnlineCrosscorrelation(n_components={}, rate={}, " "var_={}, coef_={})"
        ).format(self.n_components, self.rate, self.var_, self.coef_)

    def __str__(self):
        return "OnlineCrosscorrelation(n_components={}, rate={})".format(
            self.n_components, self.rate
        )

    _available_modes = ["naive"]
