""" Define a class for monitoring the progress of an object's attributes during a
simulation run. """

import copy

import numpy as np

from typing import Sequence
from types import SimpleNamespace


class AttributeMonitor(object):
    """ Keep track of attributes of an object during a simulation run.

    The `setup` function must be called to inform the tracker of the number of time
    steps to expect. This is used to allocate memory more efficiently.

    Attributes
    ==========
    names : sequence
        The names of the attributes to be tracked. Attributes of sub-objects can be
        accessed by using the dot ('.').
    step : int
        How often to store state information.
    n_ : int
        Total number of simulation steps expected.
    t_ : int
        Number of simulation time steps that occurred.
    i_ : int
        Number of simulation time steps that were recorded. This is, roughly,
        `self.t // self.step`.
    history_ : SimpleNamespace
        Namespace storing the history for each attribute. The type of each attribute is
        inferred from the first recorded entry. Subsequent entries will be converted to
        this type (if possible). This means that it's important that the initial type of
        a `float` entry be of `float` type as opposed to, e.g., `int`.
    """

    def __init__(self, names: Sequence, step: int = 1):
        """ Initialize the monitor with a list of attributes to follow.

        Parameters
        ----------
        names : sequence
            The names of the attributes to be tracked.
        step : int
            How often to store state information.
        """
        self.names = list(names)
        self.step = step

        # these fields will be initialized when calling `setup`
        self.n_ = None
        self.t_ = None
        self.i_ = None
        self.history_ = None

    def setup(self, n: int):
        """ Instruct the tracker how many steps will be in the simulation.

        Parameters
        ----------
        n
            Number of steps for which the simulation will run. This is used to optimize
            memory allocation.
        """
        self.n_ = n
        self.t_ = 0
        self.i_ = 0
        self.history_ = SimpleNamespace(**{name: None for name in self.names})

    def record(self, obj: object):
        """ Store the current state of the attributes that are being followed, assuming
        the current step number is divisible by `self.step`.

        Parameters
        ----------
        obj
            The object whose attributes should be copied.
        """
        if self.t_ % self.step == 0:
            for name in self.names:
                value = getattr(obj, name)

                if getattr(self.history_, name) is None:
                    # figure out the data type
                    dtype = np.asarray(value).dtype
                    # convert things like strings to dtype object
                    if np.issubdtype(dtype, np.flexible):
                        dtype = np.object

                    # allocate space
                    shape = np.shape(value)
                    n_elements = (self.n_ - 1) // self.step + 1
                    setattr(
                        self.history_,
                        name,
                        np.zeros((n_elements,) + shape, dtype=dtype),
                    )

                # make sure to make copies of object values!
                if getattr(self.history_, name).dtype == np.object:
                    value = copy.copy(value)
                getattr(self.history_, name)[self.i_] = value

            self.i_ += 1

        self.t_ += 1

    def __str__(self) -> str:
        s = (
            f"AttributeMonitor(names={str(self.names)}, n_={self.n_}, "
            + f"step={self.step}, t_={self.t_})"
        )

        return s

    def __repr__(self) -> str:
        r = (
            f"AttributeMonitor(names={repr(self.names)}, n_={self.n_}, "
            + f"step={self.step}, t_={self.t_}, i_={self.i_}, "
            + f"history_={repr(self.history_)})"
        )

        return r
