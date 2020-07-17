""" Define convenient objects to use as sources for Arma processes. """

import numpy as np


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
