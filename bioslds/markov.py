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
        Transition matrix, with `trans_mat[i, j]` being proportional to the
        probability of transitioning from state `i` to state `j` when a state
        switch occurs. The diagonal elements of this matrix are ignored, and
        the other elements in each row are normalized to sum to 1 to obtain
        transition probabilities.
    dwell_times : array of float, shape (n_components, )
        Expected dwell times in each state.
    min_dwell : array of float, shape (n_components, )
        Minimum dwell times per state.
    max_dwell : array of float, shape (n_components, )
        Maximum dwell times per state (inclusive).
    """

    def __init__(
        self,
        n_components: int,
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        dwell_times: Union[float, Sequence] = 1.0,
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
        dwell_times
            Expected dwell times in each state. These should be between
            `min_dwell` and `max_dwell`. Note that an average dwell_time of 1
            (the default) implies that a switch happens after every time step.
        start_prob
            Initial state distribution. Default is uniform across all the
            states.
        trans_mat
            Transition matrix. The diagonal elements are not used; see class
            docstring for details. The default is all transitions are equally
            probable.
        min_dwell
            Minimum dwell time for each state.
        max_dwell
            Maximum dwell time for each state (inclusive).
        """
        if n_components < 1:
            raise ValueError("Number of components should be at least 1.")
        self.n_components = n_components

        # copy parameters and handle defaults
        if start_prob is not None:
            self.start_prob = np.array(start_prob)
        else:
            self.start_prob = np.ones(self.n_components) / self.n_components

        if trans_mat is not None:
            self.trans_mat = np.array(trans_mat)
        else:
            self.trans_mat = np.ones((self.n_components, self.n_components))
        # this removes the diagonal and normalizes elements
        # also performs very basic error checking
        self._adjust_trans_mat()

        # initialize / take ownership of random number generator
        if isinstance(rng, np.random.RandomState):
            self.rng = rng
        else:
            self.rng = np.random.default_rng(rng)

        # copy / set default average dwell time
        if hasattr(dwell_times, "__len__"):
            self.dwell_times = np.array(dwell_times)
        else:
            self.dwell_times = np.repeat(dwell_times, self.n_components)
        # set dwell probabilities
        self._dwell_p = 1 - 1 / self.dwell_times

        # set minimum and maximum dwell time
        if hasattr(min_dwell, "__len__"):
            self.min_dwell = np.array(min_dwell)
        else:
            self.min_dwell = np.repeat(min_dwell, self.n_components)
        # ensure min_dwell isn't below 1
        self.min_dwell = np.clip(self.min_dwell, 1, None)

        if hasattr(max_dwell, "__len__"):
            self.max_dwell = np.array(max_dwell)
        else:
            self.max_dwell = np.repeat(max_dwell, self.n_components)
        # raise if any max_dwell is lower than the corresponding min_dwell
        if np.any(self.max_dwell < self.min_dwell):
            raise ValueError("max_dwell lower than min_dwell.")

        # check that averge dwell times make sense
        if np.any(self.dwell_times < self.min_dwell) or np.any(
            self.dwell_times > self.max_dwell
        ):
            raise ValueError("dwell_times outside [min_dwell, max_dwell].")

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
            # nothing to do
            return np.array([])

        seq = np.zeros(n, dtype=int)
        if self.n_components == 1:
            # only one thing to do
            return seq

        # start by drawing initial state
        seq[0] = self.rng.choice(self.n_components, p=self.start_prob)

        # sample the rest of the chain
        dwell_time = 1
        for i in range(1, len(seq)):
            # first decide whether to switch or not
            state = seq[i - 1]
            crt_min_dwell = self.min_dwell[state]
            crt_max_dwell = self.max_dwell[state]
            crt_dwell_p = self._dwell_p[state]

            do_switch = False
            if crt_min_dwell <= dwell_time < crt_max_dwell:
                # switch away with fixed probability
                do_switch = self.rng.uniform() >= crt_dwell_p
            elif dwell_time >= crt_max_dwell:
                # we need to move away from this state
                do_switch = True

            if do_switch:
                # switch and reset dwell counter
                # note that this switches because self._adjust_trans_mat made
                # sure to zero the diagonal of self.trans_mat
                seq[i] = self.rng.choice(
                    self.n_components, p=self.trans_mat[state]
                )
                dwell_time = 1
            else:
                seq[i] = seq[i - 1]
                dwell_time += 1

        return seq

    def _adjust_trans_mat(self):
        """ Adjust trans_mat by zero-ing the diagonal and normalizing each
        resulting row to sum to 1.

        This raises an error if any off-diagonal entries are negative.
        """
        self.trans_mat -= np.diag(np.diag(self.trans_mat))
        if np.min(self.trans_mat) < 0:
            raise ValueError("Negative transition probabilities.")

        sums = np.sum(self.trans_mat, axis=1)
        self.trans_mat = self.trans_mat / sums[:, None]
