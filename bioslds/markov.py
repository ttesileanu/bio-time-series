""" Define a class for simulating a type of semi-Markov model. """

import numpy as np

from typing import Sequence, Union, Optional


class SemiMarkov(object):
    """ A semi-Markov model.

     One way to describe a pure Markov model is to say that the system stays in
     its current state for a time that follows a geometric distribution; and
     then switches to a different state with probability given by the transition
     matrix. This makes it a special case of a Markov renewal process.

     A semi-Markov model is the same, except the dwell times (the time spent in
     each state) need not follow a geometric distribution.

     This class implements such a semi-Markov model. The following distributions
     are currently available:
       * geometric, i.e., pure Markov model
       * truncated geometric -- a minimum and/or maximum dwell time is given

    Attributes
    ==========
    n_components : int
        Number of states in the model.
    start_prob : array of float, shape (n_components, )
        Initial state distribution.
    trans_mat : array of float, shape (n_components, n_components)
        Transition matrix, with `trans_mat[i, j]` giving the probability of
        transitioning from state `i` to state `j` in the pure Markov case. In
        all cases, the matrix needs to be stochastic, i.e.,
            sum(trans_mat[i, :]) = 1 .
        Also in all cases, `1 / (1 - trans_mat[i, i])` gives the expected dwell
        time in the `i`th state, before switching to a different state. The
        probability of switching to state `j != i` after the dwell time passed
        is in all cases given by `trans_mat[i, j] / (1 - trans_mat[i, i])`. Note
        that this is properly normalized given the stochastic nature of
        `trans_mat`, and the fact that `j` cannot be the same as `i`. The only
        difference between different semi-Markov models is the distribution used
        to draw the dwell time (see below).
    min_dwell : array of float, shape (n_components, )
        Minimum dwell times per state.
    max_dwell : array of float, shape (n_components, )
        Maximum dwell times per state (inclusive).
    """

    def __init__(
        self,
        n_components: int,
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        start_prob: Optional[Sequence] = None,
        trans_mat: Optional[Sequence] = None,
        min_dwell: Union[float, Sequence] = 0,
        max_dwell: Union[float, Sequence] = np.inf,
    ):
        """ Initialize the semi-Markov model.

        Parameters
        ----------
        n_components
            Number of states.
        rng
            Random number generator or seed. If seed, a random number generator
            is created using `np.random.default_rng`.
        start_prob
            Initial state distribution. Default is uniform across all the
            states.
        trans_mat
            Transition matrix. See class docstring for details on how this is
            used in the semi-Markovian setting. Default is all transitions are
            equally probable.
        min_dwell
            Minimum dwell time for each state.
        max_dwell
            Maximum dwell time for each state (inclusive).
        """
        self.n_components = n_components

        if start_prob is not None:
            self.start_prob = np.array(start_prob)
        else:
            self.start_prob = np.ones(self.n_components) / self.n_components

        if trans_mat is not None:
            self.trans_mat = np.array(trans_mat)
        else:
            self.trans_mat = (
                np.ones((self.n_components, self.n_components))
                / self.n_components
            )

        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        if hasattr(min_dwell, "__len__"):
            self.min_dwell = np.asarray(min_dwell)
        else:
            self.min_dwell = np.repeat(min_dwell, self.n_components)

        if hasattr(max_dwell, "__len__"):
            self.max_dwell = np.asarray(max_dwell)
        else:
            self.max_dwell = np.repeat(max_dwell, self.n_components)

        # _mode allows us to choose legacy implementations
        # useful for testing
        self._mode = "naive"

    def sample(self, n: int) -> np.ndarray:
        """ Generate samples from the model.

        Parameters
        ----------
        n
            Number of samples to generate.

        Returns an array of integers representing the states that the model goes
        through.
        """
        if n <= 0:
            return np.array([])

        seq = np.zeros(n, dtype=int)
        seq[0] = self.rng.choice(self.n_components, p=self.start_prob)
        dwell_time = 1
        for i in range(1, len(seq)):
            crt_min_dwell = self.min_dwell[seq[i - 1]]
            crt_max_dwell = self.max_dwell[seq[i - 1]]
            if crt_min_dwell <= dwell_time < crt_max_dwell:
                seq[i] = self.rng.choice(
                    self.n_components, p=self.trans_mat[seq[i - 1]]
                )
            elif dwell_time < crt_min_dwell:
                # we need to stay in this state longer
                seq[i] = seq[i - 1]
            else:
                # we need to move away from this state
                crt_p = np.copy(self.trans_mat[seq[i - 1]])
                crt_p[seq[i - 1]] = 0
                crt_p = crt_p / np.sum(crt_p)
                seq[i] = self.rng.choice(self.n_components, p=crt_p)

            if seq[i] == seq[i - 1]:
                dwell_time += 1
            else:
                dwell_time = 1

        return seq
