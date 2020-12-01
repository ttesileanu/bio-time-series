""" Define convenient objects to use as sources for Arma processes. """

import numpy as np
import copy

from typing import Sequence, Union, Callable
from scipy import optimize


class Constant(object):
    """ A source that always returns the same value.

    Attributes
    ----------
    value
        Value returned by the source.
    """

    def __init__(self, value: float):
        self.value = value

    def __call__(self, size: int) -> np.ndarray:
        """ Generate constant values.

        Parameter
        ---------
        n
            Number of values to generate.
        """
        return np.repeat(self.value, size)

    def __repr__(self) -> str:
        return f"Constant({self.value})"


class Stream(object):
    """ A source that streams data from an array.

    Attributes
    ----------
    data_store
        Array from which values are returned.
    ptr_
        Current location in array.
    """

    def __init__(self, data: Sequence):
        self.data_store = data
        self.ptr_ = 0

    def __call__(self, size: int) -> np.ndarray:
        """ Return values from the data store.

        Raises `IndexError` if not enough data is available.

        Parameter
        ---------
        n
            Number of values to return.
        """
        if self.ptr_ + size > len(self.data_store):
            raise IndexError("Ran out of data.")

        data = self.data_store[self.ptr_ : self.ptr_ + size]
        self.ptr_ += size
        return data

    def __repr__(self) -> str:
        return f"Stream(data_store={self.data_store}, ptr_={self.ptr_})"


class GaussianNoise(object):
    """ A source that generates random Gaussian noise.

    Attributes
    ----------
    rng
        Random number generator.
    loc
        Location (mean) of normal distribution.
    scale
        Scale (standard deviation) of normal distribution.
    """

    def __init__(
        self,
        rng: Union[int, np.random.Generator, np.random.RandomState] = 0,
        loc: float = 0,
        scale: float = 1,
    ):
        """ Initialize the Gaussian noise source.

        Parameters
        ----------
        rng
            Random number generator or seed. If seed, a random number generator
            is created using `np.random.default_rng`.
        loc
            Location (mean) of distribution.
        scale
            Scale (standard deviation) of distribution.
        """
        if isinstance(rng, int):
            rng = np.random.default_rng(rng)

        self.rng = rng
        self.loc = loc
        self.scale = scale

    def __call__(self, size: int) -> np.ndarray:
        """ Return Gaussian random values.

        Parameter
        ---------
        n
            Number of values to return.
        """
        return self.rng.normal(self.loc, self.scale, size=size)

    def __str__(self) -> str:
        return f"GaussianNoise(loc={self.loc}, scale={self.scale})"

    def __repr__(self) -> str:
        r = f"GaussianNoise(loc={self.loc}, scale={self.scale}, " + f"rng={self.rng})"
        return r


def fix_source_scale(
    transformer, output_std: float = 1, n_samples: int = 1000, use_copy: bool = True,
) -> float:
    """ Adjust the scale for a data source to fix the output variance of a
    transformer.

    The transformer's data source must have a `scale` parameter.

    Parameters
    ----------
    transformer
        Transformer whose output variance is optimized. This should behave like
        `Arma`: it needs to have a `transform` method that can be called like
        `transformer.transform(U=source)`; and it needs an attribute called
        `default_source`.
    output_std
        Value to which to fix the transformer's output standard deviation.
    n_samples
        Number of samples to generate for each optimization iteration.
    use_copy
        If true, a deep copy of the data source is made for the optimization, so
        that the source's random generator is unaffected by this procedure.

    Returns the final value for the scale.
    """
    output_var = output_std ** 2

    source = transformer.default_source
    if use_copy:
        source_copy = copy.deepcopy(source)
    else:
        source_copy = source

    def objective(scale: float):
        source_copy.scale = np.abs(scale)
        samples = transformer.transform(n_samples, X=source_copy)
        return np.var(samples) / output_var - 1

    soln = optimize.root_scalar(
        objective, x0=np.sqrt(output_var / 2), x1=np.sqrt(2 * output_var), maxiter=100,
    )

    source.scale = np.abs(soln.root)
    return source.scale


def fix_transformer_scale(
    transformer,
    output_std: float = 1,
    n_samples: int = 1000,
    source_constructor: Callable = GaussianNoise,
) -> float:
    """ Adjust the source scaling for a transformer in order to fix its output variance.

    Parameters
    ----------
    transformer
        Transformer whose output variance is optimized. This should behave like
        `Arma`: it needs to have a `transform` method that can be called like
        `transformer.transform(U=source)`; it needs an attribute called
        `default_source`; and it needs an attribute called `source_scaling`.
    output_std
        Value to which to fix the transformer's output standard deviation.
    n_samples
        Number of samples to generate for each optimization iteration.
    source_constructor
        Callable to use to create a source for adjusting the scaling factor.
        The created object needs to have a `scale` attribute.

    Returns the final value for the scale.
    """
    # don't mess with the transformer's initial default_source
    old_source = transformer.default_source

    # make a new source, and use fix_source_scale
    source = source_constructor()
    transformer.default_source = source

    # XXX this isn't very robust: the meaning of `scale` to a particular source may well
    # be very different from simply scaling the source values after they're generated
    scale = fix_source_scale(
        transformer, output_std=output_std, n_samples=n_samples, use_copy=False
    )

    # revert to original source
    transformer.default_source = old_source

    # set scaling factor
    transformer.source_scaling = scale

    return scale
