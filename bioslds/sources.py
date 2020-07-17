""" Define convenient objects to use as sources for Arma processes. """

import numpy as np

from typing import Sequence, Union


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
        rng: Union[int, np.random.Generator, np.random.RandomState],
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
