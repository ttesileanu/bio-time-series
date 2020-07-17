""" Define convenient objects to use as sources for Arma processes. """

import numpy as np

from typing import Sequence


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
