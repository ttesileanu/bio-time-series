""" Define a class to generate samples from ARMA processes. """

from __future__ import annotations

import numpy as np

from typing import Sequence, Tuple, Callable, Optional, Union


class Arma(object):
    """ Transform input into output samples using autoregressive moving-average
    (ARMA) processes.

    Given an input signal `u[t]`, the output `y[t]` of an ARMA process is
    described by

        y[t] = bias + a[0]*y[t-1] + a[1]*y[t-2] + ... + a[p-1]*y[t-p] +
                      u[t] + b[0]*u[t-1] + ... + b[q-1]*u[t-q] ,

    where `a[i]`, `b[i]` are the coefficients of the process. This class
    focuses on scalar processes, where `y[t]`, `u[t]`, `a[i]`, and `b[i]` are
    all scalars.

    Attributes
    ==========
    a : array of float
        AR parameters. This shouldn't be changed after `__init__`.
    b : array of float
        MA parameters. This shouldn't be changed after `__init__`.
    p : int
        AR order. This shouldn't be changed.
    q : int
        MA order. This shouldn't be changed.
    bias : float
        Constant term.
    default_source : callable
        Callable used to generate source data if not explicitly provided. This
        should take an integer keyword argument `size` and return an array of
        that length. If `default_source` is not provided, the `transform` method
        needs to be called with an input sequence.
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
        bias: float = 0,
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
            This should take an integer keyword argument `size` and return an
            array of that length. If `default_source` is not provided, the
            `transform` method needs to be called with an input sequence.
        initial_conditions
            A tuple, `(initial_y, initial_u)`, of recent samples of the output
            and input sequences used to seed the simulation. If these are not
            provided, they are assumed equal to zero.
        """
        # making sure to make copies
        self.a = np.array(a)
        self.b = np.array(b)
        self.bias = bias

        self.default_source = default_source

        # inferred quantities
        self.p = len(self.a)
        self.q = len(self.b)

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
            if n_samples is None:
                raise ValueError("Need either U or n_samples.")
            if n_samples == 0:
                return np.array([]), np.array([])
            if self.default_source is None:
                raise ValueError("Need default_source if there's no U.")

            U = self.default_source

        if callable(U):
            if n_samples is None:
                raise ValueError("If source is callable n_samples is needed.")
            U = U(size=n_samples)

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
        self.history_[0][:] = y_out_full[-self.p :]
        self.history_[1][:] = u_out_full[-self.q :]

        y_out = y_out_full[self.p :]
        u_out = u_out_full[self.q :]
        return y_out, u_out

    def __str__(self) -> str:
        s = f"Arma(a={str(self.a)}, b={str(self.b)}, bias={str(self.bias)})"
        return s

    def __repr__(self) -> str:
        r = (
            f"Arma(a={repr(self.a)}, b={repr(self.b)}, "
            + f"bias={repr(self.bias)}, "
            + f"default_source={repr(self.default_source)})"
        )
        return r

    def is_stable(self) -> bool:
        """ Check whether the system is stable.

        An ARMA system is stable if all the roots of the polynomial
            z**p - a[0] * z**(p-1) - ... - a[p-1]
        lie within the unit circle.

        Returns true if the system is stable.
        """
        a_coeffs = np.ones(self.p + 1)
        a_coeffs[1:] = -self.a
        roots = np.roots(a_coeffs)
        return all(np.abs(roots) < 1)

    def is_invertible(self) -> bool:
        """ Check whether the system is invertible.

        An ARMA system is invertible if all the roots of the polynomial
            z**q + b[0] * z**(q-1) + ... + b[q-1]
        lie within the unit circle.

        Returns true if the system is invertible.
        """
        b_coeffs = np.ones(self.q + 1)
        b_coeffs[1:] = self.b
        roots = np.roots(b_coeffs)
        return all(np.abs(roots) < 1)

    def inverse(self, **kwargs) -> Arma:
        """ Return the inverse process.

        The inverse is defined by the condition that, when applied to the output
        of the current process, it returns its input.

        All keyword arguments are passed to `Arma.__init__`.

        Returns the inverse process.
        """
        inv_a = -self.b
        inv_b = -self.a
        inv_bias = 0

        return Arma(
            inv_a,
            inv_b,
            bias=inv_bias,
            **kwargs,
        )
