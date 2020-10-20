""" Define a class for implementing a biologically plausible, online, winner-
take-all algorithm for fitting a mixture of regression models to a dataset.
"""

import numpy as np

from bioslds.monitor import AttributeMonitor

from types import SimpleNamespace
from typing import Union, Optional, Sequence, Callable, Tuple
from numba import njit


class BioWTARegressor(object):
    """ A class implementing a biologically plausible winner-take-all algorithm
    that performs a multi-modal regression in a streaming setting.

    Specifically, this learns a set of regressions, with coefficients `w[k, :]`,
    such that we have
        y[t] = sum_k z[t][k] * np.dot(w[k, :], x[t]) ,
    where `z[t]` is a one-hot encoding of the model that should be used at time
    `t` (`z[t][k] == 1` if model `k` is used), `y[t]` is the output, and `x[t]`
    is the input, or predictor, variable.

    This class attempts to fit both the coefficients `w[k, i]` and the state
    assignments `z[t]`, and do so on-the-fly, as the data is being processed.
    Moreover, it uses an implementation that has a simple interpretation in
    terms of a rate-based biologically plausible neural network.

    Attributes
    ==========
    n_models : int
        Number of models in mixture.
    n_features : int
        Number of predictor variables (features).
    rate_weights : float, np.ndarray, callable
        Learning rate or learning schedule for the regression weights.
    weights_ : array, shape `(n_models, n_features)`
        Regression weights for each of the models.
    prediction_ : float
        Value of last best prediction from the model.
    start_prob_ : array of float, shape (n_models, )
        Distribution for the initial latent state. Currently this is fixed and cannot be
        learned.
    trans_mat_ : array of float, shape (n_models, n_models)
        Transition matrix for the latent state, with `trans_mat[i, j]` being the
        probability of transitioning from latent state (model) `i` to latent state
        (model) `j` at every time step. Currently this is fixed and cannot be learned.
    output_ : array of float, shape (n_models, )
        Last latent state assignment. This corresponds to the last value returned from
        `fit_infer`.
    """

    def __init__(
        self,
        n_models: int,
        n_features: int,
        rate_weights: Union[float, Sequence, Callable[[float], float]] = 1e-3,
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        weights: Optional[Sequence] = None,
        start_prob: Optional[Sequence] = None,
        trans_mat: Optional[Union[float, Sequence]] = None,
    ):
        """ Initialize the regression model.

        Parameters
        ----------
        n_models
            Number of models in mixture.
        n_features
            Number of predictor variables (features).
        rate_weights
            Learning rate or learning schedule for the regression weights. If this is a
            sequence, the `i`th element is used as the learning rate at the `i`th step.
            The last element is used for any steps beyond the length of the sequence.
            If this is a callable, the learning rate is obtained by calling it with the
            current step number as an argument.
        rng
            Random number generator or seed to use for generating initial weight
            values. If seed, a random number generator is created using
            `np.random.default_rng`. If not provided, a seed of 0 is used.
        weights
            Initial value for the weights. This overrides `rng`.
        start_prob
            Distribution for the initial latent state. If not provided, this is set to
            uniform. Currently this distribution is fixed and cannot be learned.
        trans_mat
            Transition matrix for the latent state, with `trans_mat[i, j]` being the
            probability of transitioning from latent state (model) `i` to latent state
            (model) `j` at every time step. If this is a scalar, it gives the diagonal
            element (i.e., the probability of staying in the same state), and all other
            elements are set equal (i.e., uniform probability to transition to each of
            the other states). Currently the transition matrix is fixed and cannot be
            learned.
        """
        self.n_models = n_models
        self.n_features = n_features

        if callable(rate_weights) or not hasattr(rate_weights, "__len__"):
            self.rate_weights = rate_weights
        else:
            self.rate_weights = np.array(rate_weights)
        self._rate_weights_vector = None

        # handle integer seeds
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)

        if weights is None:
            # use random initial weights
            self.weights_ = rng.normal(size=(self.n_models, self.n_features))
        else:
            self.weights_ = np.copy(weights)

        if start_prob is not None:
            self.start_prob_ = np.copy(start_prob)
        else:
            self.start_prob_ = np.ones(self.n_models) / self.n_models

        if trans_mat is not None:
            if np.size(trans_mat) > 1:
                self.trans_mat_ = np.copy(trans_mat)
            else:
                if self.n_models == 1:
                    self.trans_mat_ = np.ones((self.n_models, self.n_models))
                else:
                    p_diag = trans_mat
                    p_offdiag = (1 - trans_mat) / (self.n_models - 1)
                    self.trans_mat_ = p_offdiag * np.ones(
                        (self.n_models, self.n_models)
                    ) + (p_diag - p_offdiag) * np.eye(self.n_models)
        else:
            self.trans_mat_ = np.ones((self.n_models, self.n_models)) / self.n_models

        self.output_ = np.zeros(self.n_models)

        self._mode = "naive"

    def fit_infer(
        self,
        X: Sequence,
        y: Sequence,
        progress: Optional[Callable] = None,
        monitor: Union[None, AttributeMonitor, Sequence] = None,
        chunk_hint: int = 1000,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SimpleNamespace]]:
        """ Feed a set of samples through the model, inferring which model fits best at
        each time step, and updating the model weights.

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
            function generates an `AttributeMonitor`. In either case, the monitor's
            `history_` is returned as a second return value. Parameter values are stored
            and calculated before their updates.
        chunk_hint
            A hint about how to chunk the learning. This may or may not be used. If it
            is, the progress function will only be called once per chunk.

        If `monitor` is not provided: returns an array `r` with shape `(n_samples,
        n_models)` such  that `r[t, k]` shows, roughly, how likely it is that the sample
        at time `t` was generated by model `k`. Note that this is not a full
        probabilistic model, so this should not be taken literally.

        If a `monitor` is used: returns a tuple `(r, history)` where `r` is the output
        from above, and `history` is the `history_` attribute of the monitor that was
        used. When an `AttributeMonitor` is used (which is what happens when `monitor`
        is a sequence), the `history` output behaves like a `SimpleNamespace`, where
        each tracked attribute is a member.
        """
        n_samples, n_features = np.shape(X)
        if n_features != self.n_features:
            raise ValueError("Number of columns in X should be equal to n_features.")

        # make sure we're using numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        r = np.zeros((n_samples, self.n_models))

        fct_mapping = {
            "naive": self._fit_infer_naive,
            "numba": self._fit_infer_numba,
        }
        fct = fct_mapping[self._mode]

        if monitor is not None:
            if hasattr(monitor, "__len__") and not hasattr(monitor, "setup"):
                monitor = AttributeMonitor(monitor)

        # figure out per-step rates
        n = len(y)
        if callable(self.rate_weights):
            self._rate_weights_vector = np.array(
                [self.rate_weights(_) for _ in range(n)]
            )
        elif hasattr(self.rate_weights, "__len__"):
            if n <= len(self.rate_weights):
                self._rate_weights_vector = self.rate_weights[:n]
            else:
                n_extra = n - len(self.rate_weights)
                last_rate = self.rate_weights[-1]
                self._rate_weights_vector = np.hstack(
                    (self.rate_weights, n_extra * [last_rate])
                )
        else:
            self._rate_weights_vector = np.repeat(self.rate_weights, n)

        # noinspection PyArgumentList
        fct(
            X, y, r, progress=progress, monitor=monitor, chunk_hint=chunk_hint,
        )

        if monitor is None:
            return r
        else:
            return r, monitor.history_

    # noinspection PyUnusedLocal
    def _fit_infer_naive(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
        progress: Optional[Callable],
        monitor: Optional[AttributeMonitor],
        chunk_hint: int,
    ):
        # handle progress function
        if progress is None:
            itX = X
        else:
            itX = progress(X)

        if monitor is not None:
            monitor.setup(len(y))

        log_trans_mat = _log_safe_zero(self.trans_mat_)

        last_k = None
        for i, (crt_x, crt_y) in enumerate(zip(itX, y)):
            crt_pred = np.dot(self.weights_, crt_x)
            crt_eps = crt_y - crt_pred

            # find best-fitting model:
            # start with prior on latent states
            if i == 0:
                crt_obj = _log_safe_zero(self.start_prob_)
            else:
                crt_obj = np.copy(log_trans_mat[last_k])

            crt_obj -= 0.5 * crt_eps ** 2
            max_obj = np.max(crt_obj)
            r0 = np.exp(crt_obj - max_obj)
            r[i] = r0 / np.sum(r0)

            k = r0.argmax()

            self.prediction_ = crt_pred[k]
            self.output_[:] = r[i]
            if monitor is not None:
                monitor.record(self)

            crt_rate_weights = self._rate_weights_vector[i]
            dw = (crt_rate_weights * crt_eps[k]) * crt_x
            self.weights_[k] += dw

            last_k = k

    def _fit_infer_numba(
        self,
        X: np.ndarray,
        y: np.ndarray,
        r: np.ndarray,
        progress: Optional[Callable],
        monitor: Optional[AttributeMonitor],
        chunk_hint: int,
    ):
        if chunk_hint < 1:
            chunk_hint = 1

        # handle progress function
        if progress is not None:
            pbar = progress(total=len(X))
        else:
            pbar = None

        if monitor is not None:
            monitor.setup(len(y))

        log_start_prob = _log_safe_zero(self.start_prob_)
        log_trans_mat = _log_safe_zero(self.trans_mat_)

        crt_last_r = None
        for chunk_start in range(0, len(X), chunk_hint):
            crt_range = slice(chunk_start, chunk_start + chunk_hint)
            crt_X = X[crt_range]
            crt_y = y[crt_range]
            crt_r = r[crt_range]

            crt_n = len(crt_y)

            crt_weights = np.zeros((crt_n, self.n_models, self.n_features))
            crt_predictions = np.zeros(crt_n)

            self.weights_ = _perform_fit_infer(
                crt_X,
                crt_y,
                crt_r,
                np.copy(self.weights_),
                self._rate_weights_vector,
                crt_last_r,
                log_start_prob,
                log_trans_mat,
                crt_weights,
                crt_predictions,
            )

            if pbar is not None:
                pbar.update(crt_n)
            if monitor is not None:
                monitor.record_batch(
                    SimpleNamespace(
                        weights_=crt_weights, prediction_=crt_predictions, output_=crt_r
                    )
                )

            crt_last_r = crt_r[-1]
            self.output_[:] = crt_last_r

        if pbar is not None:
            pbar.close()

    def __repr__(self) -> str:
        r = (
            f"BioWTARegressor(n_models={self.n_models}, n_features"
            f"={self.n_features}, rate_weights={self.rate_weights}, weights_="
            f"{repr(self.weights_)})"
        )

        return r

    def __str__(self) -> str:
        s = (
            f"BioWTARegressor(n_models={self.n_models}, n_features"
            f"={self.n_features})"
        )

        return s

    _available_modes = ["naive", "numba"]


@njit
def _perform_fit_infer(
    X: np.ndarray,
    y: np.ndarray,
    r: np.ndarray,
    crt_weights: np.ndarray,
    rate: np.ndarray,
    last_r: Optional[np.ndarray],
    log_start_prob: np.ndarray,
    log_trans_mat: np.ndarray,
    weights: np.ndarray,
    predictions: np.ndarray,
) -> np.ndarray:
    n = len(y)
    last_k = 0
    crt_obj = np.zeros(len(log_start_prob))
    for i in range(n):
        crt_x = X[i]
        crt_pred = np.dot(crt_weights, crt_x)
        crt_eps = y[i] - crt_pred

        # find best-fitting model:
        # start with prior on latent states
        if i == 0:
            if last_r is None:
                crt_obj[:] = log_start_prob
            else:
                last_k = np.argmax(last_r)
                crt_obj[:] = log_trans_mat[last_k]
        else:
            crt_obj[:] = log_trans_mat[last_k]

        crt_obj -= 0.5 * crt_eps ** 2
        max_obj = np.max(crt_obj)
        r0 = np.exp(crt_obj - max_obj)
        r[i, :] = r0 / np.sum(r0)

        k = r0.argmax()

        if weights is not None:
            weights[i, :, :] = crt_weights
            predictions[i] = crt_pred[k]

        crt_weights[k, :] += (rate[i] * crt_eps[k]) * crt_x

        last_k = k

    return crt_weights


def _log_safe_zero(m: np.ndarray) -> np.ndarray:
    """ Take elementwise log of an array while avoiding warnings with elements that are
    zero.

    Parameters
    ----------
    m
        Array whose elementwise log is to be taken.

    Returns an array of elementwise log values, in which zeros are mapped to `-np.inf`.
    Unlike `np.log`, this is done without issuing a warning.
    """
    lm = np.tile(-np.inf, np.shape(m))
    mask = m != 0
    lm[mask] = np.log(m[mask])

    return lm
