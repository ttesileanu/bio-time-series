""" Define a class to generate samples from ARMA processes. """

import numpy as np
import copy

from bioslds.monitor import AttributeMonitor

from numba import njit
from typing import Sequence, Tuple, Callable, Optional, Union
from types import SimpleNamespace


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
    n_features : int
        Number of input dimensions. This is always equal to 1.
    n_components : int
        Number of output dimensions. This is always equal to 1.
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
    source_scaling : float
        Amount by which the source data is scaled before using.
    input_ : float
        Last value of input (source) signal.
    output_ : float
        Last value of output signal.
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
        source_scaling: float = 1,
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
        source_scaling
            Amount by which the source data is scaled before using.
        """
        self.n_features = 1
        self.n_components = 1

        # making sure to make copies
        self.a = np.array(a)
        self.b = np.array(b)
        self.bias = bias

        self.default_source = default_source
        self.source_scaling = source_scaling

        # inferred quantities
        self.p = len(self.a)
        self.q = len(self.b)

        if initial_conditions is not None:
            self.history_ = (
                np.copy(initial_conditions[0]),
                np.copy(initial_conditions[1]),
            )
            if len(self.history_[0]) > 0:
                self.output_ = self.history_[0][-1]
            else:
                self.output_ = 0

            if len(self.history_[1]) > 0:
                self.input_ = self.history_[1][-1]
            else:
                self.input_ = 0
        else:
            self.history_ = (np.zeros(self.p), np.zeros(self.q))
            self.input_ = 0
            self.output_ = 0

        # _mode allows us to choose legacy implementations
        # useful for testing
        self._mode = "ma_conv_ar_numba"

    def transform(
        self,
        n_samples: Optional[int] = None,
        U: Union[None, Sequence, Callable] = None,
        monitor: Optional[AttributeMonitor] = None,
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
        monitor
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`). Parameter values are stored and
            calculated before their updates.

        Returns a tuple `(Y, U)` of generated `y` and `u` samples. If the `U`
        parameter was used and was a sequence, the output `U` simply mirrors the
        input.
        """
        if U is None:
            if n_samples is None:
                raise ValueError("Need either U or n_samples.")
            if n_samples == 0:
                if monitor is not None:
                    monitor.setup(0)
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
        if monitor is not None:
            monitor.setup(n)
        if n == 0:
            return np.array([]), np.array([])

        u_out_full = np.zeros(n + self.q)
        u_out_full[: self.q] = self.history_[1]
        u_out_full[self.q :] = U

        y_out_full = np.zeros(n + self.p)
        y_out_full[: self.p] = self.history_[0]

        transform_dict = {
            "naive": self._transform_naive,
            "ma_conv": self._transform_ma_conv,
            "ma_conv_ar_numba": self._transform_ma_conv_ar_numba,
            "numba": self._transform_numba,
        }
        transform_fct = transform_dict[self._mode]
        # noinspection PyArgumentList
        transform_fct(y_out_full, u_out_full, monitor=monitor)

        # update history
        if self.p > 0:
            self.history_[0][:] = y_out_full[-self.p :]
        if self.q > 0:
            self.history_[1][:] = u_out_full[-self.q :]

        y_out = y_out_full[self.p :]
        u_out = u_out_full[self.q :]

        self.input_ = u_out[-1]
        self.output_ = y_out[-1]
        return y_out, u_out

    def _transform_naive(
        self,
        y_out_full: np.ndarray,
        u_out_full: np.ndarray,
        monitor: Optional[AttributeMonitor],
    ):
        """ Perform the transformation using a naive, slow algorithm. """
        n = len(y_out_full) - self.p
        a_flip = np.flip(self.a)
        b_flip_big = self.source_scaling * np.hstack((np.flip(self.b), [1]))

        for i in range(n):
            ar_part = np.dot(a_flip, y_out_full[i : i + self.p])
            ma_part = np.dot(b_flip_big, u_out_full[i : i + self.q + 1])
            y_out_full[i + self.p] = ar_part + ma_part + self.bias

            self.input_ = u_out_full[i + self.q]
            self.output_ = y_out_full[i + self.p]
            if monitor is not None:
                monitor.record(self)

    def _transform_ma_conv(
        self,
        y_out_full: np.ndarray,
        u_out_full: np.ndarray,
        monitor: Optional[AttributeMonitor],
    ):
        """ Perform the transformation using `np.convolve` for the MA part. """
        n = len(y_out_full) - self.p
        a_flip = np.flip(self.a)

        b_ext = self.source_scaling * np.hstack(([1], self.b))
        u = np.convolve(u_out_full, b_ext, mode="valid")

        for i in range(n):
            ar_part = np.dot(a_flip, y_out_full[i : i + self.p])
            y_out_full[i + self.p] = ar_part + u[i] + self.bias

            self.input_ = u_out_full[i + self.q]
            self.output_ = y_out_full[i + self.p]
            if monitor is not None:
                monitor.record(self)

    def _transform_ma_conv_ar_numba(
        self,
        y_out_full: np.ndarray,
        u_out_full: np.ndarray,
        monitor: Optional[AttributeMonitor],
    ):
        """ Perform the transformation using `np.convolve` for MA part and
        Numba-accelerated code for AR. """
        b_ext = self.source_scaling * np.hstack(([1], self.b))
        u = np.convolve(u_out_full, b_ext, mode="valid")

        if self.p > 0:
            # flip doesn't create a copy by default, which can slow Numba!
            _perform_ar(y_out_full, u, np.copy(self.a[::-1]), self.p, self.bias)
        else:
            y_out_full[self.p :] = u + self.bias

        if monitor is not None:
            monitor.record_batch(
                SimpleNamespace(
                    input_=u_out_full[self.q :], output_=y_out_full[self.p :],
                )
            )

    def _transform_numba(
        self,
        y_out_full: np.ndarray,
        u_out_full: np.ndarray,
        monitor: Optional[AttributeMonitor],
    ):
        """ Perform the transformation using Numba-accelerated version of naive
        algorithm. """
        if self.q > 0:
            b_flip_big = self.source_scaling * np.hstack((np.flip(self.b), [1]))
            u = _perform_ma(u_out_full, b_flip_big)
        else:
            u = u_out_full

        if self.p > 0:
            _perform_ar(y_out_full, u, np.copy(self.a[::-1]), self.p, self.bias)
        else:
            y_out_full[self.p :] = u + self.bias

        if monitor is not None:
            monitor.record_batch(
                SimpleNamespace(
                    input_=u_out_full[self.q :], output_=y_out_full[self.p :],
                )
            )

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

    def calculate_poles(self) -> np.ndarray:
        """ Find the poles for the ARMA process.

        Returns the poles -- the complex roots of the polynomial
            z**p - a[0] * z**(p-1) - ... - a[p-1] .
        """
        a_coeffs = np.ones(self.p + 1)
        a_coeffs[1:] = -self.a
        roots = np.roots(a_coeffs)
        return roots

    def calculate_zeros(self) -> np.ndarray:
        """ Find the zeros for the ARMA process.

        Returns the zeros -- the complex roots of the polynomial
            z**q + b[0] * z**(q-1) + ... + b[q-1]
        """
        b_coeffs = np.ones(self.q + 1)
        b_coeffs[1:] = self.b
        roots = np.roots(b_coeffs)
        return roots

    def is_stable(self) -> bool:
        """ Check whether the system is stable.

        An ARMA system is stable if all the roots of the polynomial
            z**p - a[0] * z**(p-1) - ... - a[p-1]
        lie within the unit circle.

        Returns true if the system is stable.
        """
        poles = self.calculate_poles()
        return all(np.abs(poles) < 1)

    def is_invertible(self) -> bool:
        """ Check whether the system is invertible.

        An ARMA system is invertible if all the roots of the polynomial
            z**q + b[0] * z**(q-1) + ... + b[q-1]
        lie within the unit circle.

        Returns true if the system is invertible.
        """
        zeros = self.calculate_zeros()
        return all(np.abs(zeros) < 1)

    def inverse(self, **kwargs) -> "Arma":
        """ Return the inverse process.

        The inverse is defined by the condition that, when applied to the output
        of the current process, it returns its input.

        All keyword arguments are passed to `Arma.__init__`.

        Returns the inverse process.
        """
        inv_a = -self.b
        inv_b = -self.a
        inv_bias = -self.bias

        return Arma(inv_a, inv_b, bias=inv_bias, **kwargs,)

    def copy(self) -> "Arma":
        """ Make a deep copy of the current process.

        This ensures that the history and coefficients are copied by value so
        that running or editing the original does not affect the copy. The
        `default_source`, if provided, is also deep-copied.

        This is equivalent to using `copy.deepcopy()`.

        Returns a copy of the current process.
        """
        return copy.deepcopy(self)

    _available_modes = ["naive", "ma_conv", "ma_conv_ar_numba", "numba"]


@njit
def _perform_ar(y: np.ndarray, u: np.ndarray, a_flip: np.ndarray, p: int, bias: float):
    n = len(y) - p
    for i in range(n):
        crt_past = y[i : i + p]
        ar_part = np.dot(a_flip, crt_past)
        y[i + p] = ar_part + u[i] + bias


@njit
def _perform_ma(u_out_full: np.ndarray, b_flip_big: np.ndarray) -> np.ndarray:
    q_big = len(b_flip_big)
    q = q_big - 1
    n = len(u_out_full) - q
    u = np.empty(n)

    for i in range(n):
        u[i] = np.dot(b_flip_big, u_out_full[i : i + q_big])

    return u


def make_random_arma(
    p: int,
    q: int,
    rng: Union[np.random.Generator, np.random.RandomState],
    max_pole_radius: float = 1,
    max_zero_radius: float = 1,
    bias_range: Optional[tuple] = None,
    **kwargs,
) -> Arma:
    """ Generate a random ARMA process.

    This works by randomly drawing values for the process' poles and zeros, then
    computing the corresponding ARMA coefficients. The poles and zeros are drawn
    uniformly within a disc of a given radius in the complex plane.

    If requested, a random value can also be chosen for the bias (see
    `bias_range` below); otherwise a zero-mean process is generated.

    Parameters
    ----------
    p
        AR order.
    q
        MA order.
    rng
        Random number generator to use for generating the ARMA coefficients.
    max_pole_radius
        Ensure that all the system's poles lie within the given radius. When
        this is set to 1 (the default), the process is guaranteed stable.
    max_zero_radius
        Ensure that all the system's zeros lie within the given radius. When
        this is set to 1 (the default), the process is guaranteed invertible.
    bias_range
        A tuple `(min_bias, max_bias)` giving the range in which to sample
        (uniformly) the bias. If not provided, the bias is fixed at 0.
    All other keyword arguments are passed directly to `Arma.__init__`.

    Returns the randomly generated ARMA process.
    """
    a_full = _generate_random_poly(p, max_pole_radius, rng)
    b_full = _generate_random_poly(q, max_zero_radius, rng)

    bias_args = {}
    if bias_range is not None:
        bias_args["bias"] = rng.uniform(*bias_range)

    # a_full[0] and b_full[0] are just 1
    return Arma(-a_full[1:], b_full[1:], **bias_args, **kwargs)


def _generate_random_poly(
    n: int, radius: float, rng: Union[np.random.Generator, np.random.RandomState],
) -> np.ndarray:
    """ Generate a random real polynomial with roots constrained to lie within a
    disk of given radius.

    The ordering of the resulting coefficients is as in `numpy.roots`, i.e.,
    such that the polynomial is given by
        coeffs[0] * z**(n+1) + coeffs[1] * z**n + ... + coeffs[n]

    Parameters
    ----------
    n
        Number of roots.
    radius
        Radius within which the roots must lie.
    rng
        Random number generator.

    Returns the coefficients of the random polynomial.
    """
    roots = np.zeros(n, dtype=complex)

    # will need at least one real root if n is odd
    n_complex_pairs = n // 2
    n_complex = 2 * n_complex_pairs

    # generate pairs of random complex roots
    roots[:n_complex_pairs] = radius * _random_unit_circle(rng, size=n_complex_pairs)
    roots[n_complex_pairs:n_complex] = np.conjugate(roots[:n_complex_pairs])

    # generate one random real root, if necessary
    if n % 2 != 0:
        # ensure root is strictly smaller than `radius` in absolute value
        roots[n_complex] = rng.uniform(-radius, radius)

    # build the polynomial
    coeffs = np.polynomial.polynomial.polyfromroots(roots)

    # ensure the dtype is real (our choice of roots guarantees that the
    # actual polynomial is real)
    coeffs = coeffs.real

    # ensure that the coefficients are ordered in the proper way
    coeffs = np.copy(coeffs[::-1])

    return coeffs


def _random_unit_circle(
    rng: Union[np.random.Generator, np.random.RandomState], size: int = 1
):
    """ Generate random complex numbers within the unit circle.

    Parameters
    ----------
    size
        Number of random values to generate.
    """
    r = np.sqrt(rng.uniform(size=size))
    # !!! unlike RandomState.randint / Generator.integers, passing only one
    # !!! positional argument to `uniform` *does not* mean drawing numbers
    # !!! between 0 and that value. It instead samples number between *1* (the
    # !!! default `high`) and that value !!!
    theta = rng.uniform(0, 2 * np.pi, size=size)
    return r * np.exp(1j * theta)
