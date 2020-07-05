""" Define a class to generate samples from ARMA processes. """

import numpy as np

from typing import Sequence, Tuple, Callable, Optional, Union


def _get_len(x: Union[float, Sequence]) -> int:
    """ Get length of single object or sequence.

    This returns 1 if the input does not have a `__len__` attribute, otherwise
    returns the length.

    Parameters
    ----------
    x
        The object whose length to return.

    Returns the length.
    """
    if hasattr(x, "__len__"):
        return len(x)
    else:
        return 1


class Arma(object):
    """ Transform input into output samples using autoregressive moving-average
    (ARMA) processes.

    Given an input signal `u[t]`, the output `y[t]` of an ARMA process is
    described by

        y[t] = bias + a[0]*y[t-1] + a[1]*y[t-2] + ... + a[p-1]*y[t-p] +
                      u[t] + b[0]*u[t-1] + ... + b[q-1]*u[t-q] ,

    where `a[i]`, `b[i]` are the coefficients of the process. This class
    implements both scalar processes, where `y[t]`, `u[t]`, `a[i]`, and `b[i]`
    are all scalars, and vector (VARMA) processes, where `y[t]` and `u[t]` are
    vectors and `a[i]` and `b[i]` are matrices.

    Attributes
    ==========
    n_components : int
        Number of dimensions in the output. This is the size of `y[t]`, and is
        read-only.
    n_features : int
        Number of dimensions in the input. This is the size of `u[t]`, and is
        read-only. In the parametrization used here, `n_features` is always
        equal to `n_components`.
    a : array of float or matrix
        AR parameters. These are read-only.
    b : array of float or matrix
        MA parameters. These are read-only.
    p : int
        AR order. This is equal to `len(a)`, and is read-only.
    q : int
        MA order. This is equal to `len(b)`, and is read-only.
    bias : float or vector
        Constant term. This is read-only.
    default_source : callable
        Callable used to generate source data if not explicitly provided. This
        should take a keyword argument `size` that is a tuple of the form
        `(n_samples, n_features)` and return a correspondingly sized array.
        If `default_source` is not provided, the `transform` method needs to be
        called with an input sequence.
    history_ : tuple of arrays
        A tuple, `(history_y, history_u)`, of recent samples of the output and
        input sequences. The number of samples kept depends on the order: `p`
        samples are kept in `history_y`, and `q` samples are kept in
        `history_u`.
    """

    def __init__(
        self,
        a: Sequence,
        b: Sequence,
        bias: Union[float, Sequence] = 0,
        default_source: Optional[Callable] = None,
        initial_conditions: Optional[Tuple[Sequence, Sequence]] = None,
    ):
        """ Initialize the process.

        Parameters
        ----------
        a
            AR parameters.
        b
            MA parameters.
        bias
            Constant term.
        default_source
            Callable used to generate source data if not explicitly provided.
            This should take a keyword argument `size` that is a tuple of the
            form `(n_samples, n_features)` and return a correspondingly sized
            array. If `default_source` is not provided, the `transform` method
            needs to be called with an input sequence.
        initial_conditions
            A tuple, `(initial_y, initial_u)`, of recent samples of the output
            and input sequences used to seed the simulation. If these are not
            provided, they are assumed equal to zero.
        """
        # making sure to make copies
        self.a = np.array(a)
        self.b = np.array(b)
        if hasattr(bias, "__len__"):
            # make sure to make copy
            self.bias = np.array(bias)
        else:
            self.bias = bias

        # inferred quantities
        self.p = len(self.a)
        self.q = len(self.b)

        # work out number of input and output dimensions, make sure all matches
        na = _get_len(self.a[0]) if len(self.a) > 0 else None
        nb = _get_len(self.b[0]) if len(self.b) > 0 else None
        nbias = _get_len(self.bias)
        if na is None and nb is not None:
            na = nb
        elif na is not None and nb is None:
            nb = na
        elif na is None and nb is None:
            na = nb = nbias
        if na != nb:
            raise ValueError("Mismatching a and b dimensions.")
        if na != nbias and nbias != 1:
            raise ValueError("Mismatching coefficient and bias dimensions.")
        self.n_components = na
        self.n_features = self.n_components

        if initial_conditions is not None:
            self.history_ = (
                np.copy(initial_conditions[0]),
                np.copy(initial_conditions[1]),
            )
        else:
            self.history_ = (np.zeros(self.p), np.zeros(self.q))

        # _mode allows us to choose legacy implementations
        # useful for testing
        self._mode = "naive"

    def transform(
        self,
        n_samples: Optional[int] = None,
        U: Union[None, Sequence, Callable] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ Process or generate input samples.

        The function uses exactly `n_samples` input samples.

        The recent history of the input and output sequences is stored so that
        consecutive calls to `transform` are equivalent to a single call with
        a larger number of samples,
            `transform(n1 + n2) == concat(transform(n1), transform(n2)`) .

        If no input source is explicitly provided, the `self.default_source` is
        used. An exception is raised if a default source was not provided in
        this case.

        Parameters
        ----------
        n_samples
            Number of samples to generate. If not provided, `U` must be provided
            and it must be a sequence.
        U
            Input samples or input generator. If not provided,
            `self.default_source` is used to generate the sample (and if the
            latter wasn't provided, an exception is raised). If `U` is given and
            is a sequence, it must be that `len(U) == n_samples`. If `U` is a
            callable, it must take a keyword argument `size` in the form of a
            tuple `(n_samples, n_features)`.

        Returns a tuple `(Y, U)` of generated `y` and `u` samples. If the `U`
        parameter was used and was a sequence, the output `U` simply mirrors the
        input.
        """
        if U is None:
            if n_samples == 0:
                return np.array([]), np.array([])
            raise NotImplementedError("default_source not yet implemented.")

        # output vectors including pre-history
        n = len(U)
        y_out_full = np.zeros(n + self.p)
        u_out_full = np.zeros(n + self.q)

        y_out_full[: self.p] = self.history_[0]
        u_out_full[: self.q] = self.history_[1]
        u_out_full[self.q :] = U

        a_flip = np.flip(self.a)
        b_flip_big = np.hstack((np.flip(self.b), [1]))

        for i in range(n):
            ar_part = np.dot(a_flip, y_out_full[i : i + self.p])
            ma_part = np.dot(b_flip_big, u_out_full[i : i + self.q + 1])
            y_out_full[i + self.p] = ar_part + ma_part

        # update history
        self.history_[0][:] = y_out_full[-self.p:]
        self.history_[1][:] = u_out_full[-self.q:]

        y_out = y_out_full[self.p :]
        u_out = u_out_full[self.q :]
        return y_out, u_out
