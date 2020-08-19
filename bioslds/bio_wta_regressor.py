""" Define a class for implementing a biologically plausible, online, winner-
take-all algorithm for fitting a mixture of regression models to a dataset.
"""

import numpy as np

from types import SimpleNamespace
from typing import Union, Optional, Sequence, Callable, Tuple


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
    rate_weights : float
        Learning rate for the regression weights.
    weights_ : array, shape `(n_models, n_features)`
        Regression weights for each of the models.
    start_prob_ : array of float, shape (n_models, )
        Distribution for the initial latent state. Currently this is fixed and cannot be
        learned.
    trans_mat_ : array of float, shape (n_models, n_models)
        Transition matrix for the latent state, with `trans_mat[i, j]` being the
        probability of transitioning from latent state (model) `i` to latent state
        (model) `j` at every time step. Currently this is fixed and cannot be learned.
    """

    def __init__(
        self,
        n_models: int,
        n_features: int,
        rate_weights: float = 1e-3,
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
            Learning rate for the regression weights.
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
        self.rate_weights = rate_weights

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
            self.trans_mat_ = (
                np.ones((self.n_models, self.n_models)) / self.n_models
            )

    def fit_infer(
        self,
        X: Sequence,
        y: Sequence,
        progress: Optional[Callable] = None,
        return_history: bool = False,
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
            Progress function that can be used as a wrapper, like `tqdm.tqdm`.
        return_history
            If true, the function returns a second output -- a namespace containing the
            time evolution for the weights and for the predictions at each step. These
            are stored and calculated before the weight update.

        If `return_history` is false: returns an array `r` with shape `(n_samples,
        n_models)` such  that `r[t, k]` shows, roughly, how likely it is that the sample
        at time `t` was generated by model `k`. Note that this is not a full
        probabilistic model, so this should not be taken literally.

        If `return_history` is true: returns a tuple `(r, history)` where `r` is the
        output from above and `history` is a `SimpleNamespace` with members `weights`
        and `predictions`. These have shapes `(n_samples, n_models, n_features)` and
        `(n_samples,)`, respectively.
        """
        n_samples, n_features = np.shape(X)
        if n_features != self.n_features:
            raise ValueError(
                "Number of columns in X should be equal to n_features."
            )

        # make sure we're using numpy arrays
        X = np.asarray(X)
        y = np.asarray(y)

        r = np.zeros((n_samples, self.n_models))

        # handle progress function
        if progress is None:
            itX = X
        else:
            itX = progress(X)

        if return_history:
            weights = np.zeros((n_samples, self.n_models, self.n_features))
            predictions = np.zeros(n_samples)
        else:
            # this is just to assuage silly PyCharm lint
            weights = None
            predictions = None

        log_trans_mat = _log_safe_zero(self.trans_mat_)
        for i, (crt_x, crt_y) in enumerate(zip(itX, y)):
            crt_pred = np.dot(self.weights_, crt_x)
            crt_eps = crt_y - crt_pred

            # find best-fitting model:
            # start with prior on latent states
            if i == 0:
                crt_obj = _log_safe_zero(self.start_prob_)
            else:
                crt_obj = r[i - 1] @ log_trans_mat

            crt_obj -= 0.5 * crt_eps ** 2
            max_obj = np.max(crt_obj)
            r0 = np.exp(crt_obj - max_obj)
            r[i] = r0 / np.sum(r0)

            k = r0.argmax()

            if return_history:
                weights[i, :, :] = self.weights_
                predictions[i] = crt_pred[k]

            dw = (self.rate_weights * crt_eps[k]) * crt_x
            self.weights_[k] += dw

        if not return_history:
            return r
        else:
            history = SimpleNamespace(weights=weights, predictions=predictions)
            return r, history

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
