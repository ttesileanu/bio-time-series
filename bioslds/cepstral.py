""" Define functions and classes for implementing online cepstral norm calculation. """

import numpy as np

from types import SimpleNamespace
from typing import Tuple, Sequence, Optional
from numba import njit

from bioslds.monitor import AttributeMonitor
from bioslds.utils import to_hankel


@njit
def givens(a: float, b: float) -> Tuple[float, float]:
    """ Calculate the cosine and sine for a Givens rotation.

    The Givens rotation generated here is

    G = [[ c, s],
         [-s, c]]

    and has the property that

    [a, b] @ G = [d, 0]

    where `abs(d) = sqrt(a * a + b * b)`.

    Parameters
    ----------
    a
        First element of vector to rotate.
    b
        Second element of vector to rotate.

    Returns a tuple, `(c, s)`, where `c` is the cosine of the rotation angle that zeros
    the second component, and `s` is its sine.
    """
    # XXX this will have trouble if a == b == 0!
    if abs(b) >= abs(a):
        # the branching is meant to maximize numerical stability
        t = -a / b
        s = -np.sign(b) / np.sqrt(1 + t * t)
        c = s * t
    else:
        t = -b / a
        c = np.sign(a) / np.sqrt(1 + t * t)
        s = c * t

    return c, s


@njit
def add_lq_column(l_factor: np.ndarray, v: np.ndarray):
    """ Update the non-zero part of the L term in an LQ decomposition upon appending a
    column to `L * Q`.

    The calculation is done in place. This is only for the case in which the number of
    columns of `L * Q` is at least as large as the number of rows.

    This avoids recalculating the full LQ decomposition, and avoids updating the Q
    matrix altogether. The `l_factor` matrix must be the square, non-zero portion of the
    L term.

    This uses algorithms 2.6 and 2.7 from
    http://eprints.ma.man.ac.uk/1192/1/qrupdating_12nov08.pdf
    (after transposing to get LQ instead of QR).

    Parameters
    ----------
    l_factor
        The L term in the LQ decomposition of some matrix `A`. This must be a square
        matrix, and will be changed by the function.
    v
        The elements of the row to add.

    The function does not output anything, but updates the `l_factor` matrix.
    """
    # make sure `v` isn't changed
    v = np.copy(v)

    for i in range(l_factor.shape[0]):
        c, s = givens(l_factor[i, i], v[i])
        l_factor[i, i] = c * l_factor[i, i] - s * v[i]

        t1, t2 = l_factor[i + 1 :, i], v[i + 1 :]
        l_factor[i + 1 :, i], v[i + 1 :] = c * t1 - s * t2, s * t1 + c * t2


class OnlineCepstralNorm(object):
    """ Calculate cepstral norm of a set of sequences using an online algorithm.

    This assumes that the input data has zero mean.

    This works by updating the L part of the LQ decomposition of the late and total
    Hankel matrices (e.g., `H[i, j](t) = y[t - i - j]`). The update first discounts all
    old samples by multiplying the (implicit) Hankel matrices by a discount factor
    (`gamma = 1 - rate`), and then adds a column corresponding to the new samples. The
    cepstral norm is calculated from the diagonal elements of the L matrices for the
    full and late Hankel matrices, using a version of the method from De Cock, 2002. The
    update of the L factors is done efficiently (see `add_lq_column`).

    Attributes
    ==========
    n_features : int
        Number of input dimensions.
    n_components : int
        Number of output dimensions. This is always equal to `n_features`.
    order : int
        Number of cepstral coefficients to use. This is equal to the number of rows in
        the late Hankel matrix, or half the number of rows in the total Hankel matrix.
    rate : float
        Learning rate.
    output_ : np.ndarray
        Current output, which is the estimate of the cepstral norm for each sequence.
    history_ : np.ndarray
        Vector containing the past `2 * self.order` samples in each sequence. This is
        shape `(2 * self.order, self.n_features)`.
    l_total_ : np.ndarray
        Current estimate of the L factor for the total Hankel matrix.
    l_late_ : np.ndarray
        Current estimate of the L factor for the later half of the Hankel matrix.
    """

    def __init__(self, n_features: int, order: int, rate: float = 0.01):
        """ Initialize the cepstral-norm calculator.

        The initial L factors (`self.l_total_` and `self.l_late_`) are set to the
        identity.

        Parameters
        ----------
        n_features
            Number of input dimensions.
        order
            Number of cepstral coefficients to use.
        rate
            Learning rate.
        """
        self.n_features = n_features
        self.n_components = n_features

        self.order = order
        self.rate = rate

        self.output_ = np.zeros(self.n_features)
        self.history_ = np.zeros((2 * self.order, self.n_features))

        self.l_total_ = np.asarray(
            [np.eye(2 * self.order) for _ in range(self.n_features)]
        )
        self.l_late_ = np.asarray([np.eye(self.order) for _ in range(self.n_features)])

    # noinspection PyUnusedLocal
    def transform(
        self,
        X: Sequence,
        monitor: Optional[AttributeMonitor] = None,
        chunk_hint: int = None,
    ) -> np.ndarray:
        """ Calculate local cross-correlation for a pair of sequences.

        Parameters
        ----------
        X : Sequence
            Sequence of input sequences. This has shape `(n_samples, n_features)`.
        monitor
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`). Parameter values are stored and
            calculated before their updates.
        chunk_hint
            A hint about how to chunk the learning. This is not currently used.

        Returns an array of local estimates of the cepstral norm for each input
        sequence, with shape `(n_samples, n_features)`.
        """
        n = len(X)
        gamma = 1 - self.rate
        alpha = np.sqrt(1 - gamma ** 2)

        norms = np.zeros((n, self.n_components))
        for i in range(n):
            crt_x = X[i]

            # add samples to history
            self.history_[:-1] = self.history_[1:]
            self.history_[-1] = crt_x

            # apply the discount factor
            self.l_total_ *= gamma
            self.l_late_ *= gamma

            # ...then add columns to the Hankel matrices
            v = alpha * self.history_
            v_late = v[self.order :]

            for k in range(self.n_features):
                add_lq_column(self.l_total_[k], v[:, k])
                add_lq_column(self.l_late_[k], v_late[:, k])

                # calculate cepstral norm
                neg_terms = np.log(np.abs(np.diag(self.l_total_[k])[self.order :]))
                pos_terms = np.log(np.abs(np.diag(self.l_late_[k])))

                norms[i, k] = 2 * np.sum(pos_terms - neg_terms)

        self.output_[:] = norms[-1]

        if monitor is not None:
            monitor.setup(n)
            obj = SimpleNamespace(output_=norms)
            monitor.record_batch(obj)

        return norms

    def __repr__(self) -> str:
        s = (
            f"OnlineCesptralNorm("
            f"n_features={self.n_features}, "
            f"order={self.order}, "
            f"rate={self.rate})"
        )
        return s
