""" Define classes for generating test datasets based on alternating signal snippets.
"""

import sys
import copy

import numpy as np

from typing import Sequence, Union

from bioslds.markov import SemiMarkov


class SwitchingSnippetSignal:
    """ A signal generated from switching snippets.

    Attributes
    ==========
    y : np.ndarray
        The signal.
    usage_seq : np.ndarray
        A vector of integers indicating which snippet was used at each time step.
    scale : float
        Amount by which the whole signal was scaled (if `normalize` was enabled in
        `RandomSnippetDataset`).
    """

    def __init__(self, y: np.ndarray, usage_seq: np.ndarray, scale: float = 1):
        self.y = y
        self.usage_seq = usage_seq
        self.scale = scale

    def __repr__(self) -> str:
        return (
            f"SwitchingSnippetSignal(y={self.y}, usage_seq={self.usage_seq}, "
            f"scale={self.scale})"
        )


class RandomSnippetDataset(Sequence[SwitchingSnippetSignal]):
    """ Generate random datasets of switching snippets on demand.

        This acts like an iterable, allowing sequential or random access to randomly-
        generated signals. Instead of being stored in memory, like in a list, the
        signals are generated on-demand, every time an element is accessed.

        The switching between snippets follows a semi-Markov chain, with the maximum
        dwell time equal to the length of the snippet. If the dwell time is shorter than
        the length, a portion of the snippet is used, chosen uniformly at random from
        the entire length.

        Attributes
        ==========
        n_signals : int
            Number of signals in the dataset. This is the same as `len(self)`.
        n_samples : int
            Number of samples in each signal.
        snippets : Sequence
            Library of snippets to choose from. Each snippet should be a numeric
            sequence.
        signal_seeds : Sequence
            Random seeds to use for generating the signals.
        normalize : bool
            If true, every signal is normalized to have unit variance.
        semi_markov_kws : dict
            Additional arguments to be passed to `SemiMarkov` constructor.
        """

    def __init__(
        self,
        n_signals: int,
        n_samples: int,
        snippets: Sequence[Sequence],
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        normalize: bool = False,
        **kwargs,
    ):
        """ Initialize the dataset.

        Parameters
        ----------
        n_signals
            Number of signals in the dataset. This will be the dataset's length.
        n_samples
            Number of samples in each signal.
        snippets
            Library of snippets to choose from. Each snippet should be a numeric
            sequence.
        rng
            Random number generator or seed to use to generate the datasets. The precise
            way in which this is used is not specified, but it is guaranteed that
            starting with the same state (or seed) will give the same results when using
            the same version of Numpy (similar to the guarantees from
            `np.random.Generator`).
        normalize
            If true, every signal is normalized to have unit variance.
        All other keyword arguments will be passed to `SemiMarkov` constructor.
        """
        self.n_signals = n_signals
        self.n_samples = n_samples
        self.snippets = snippets

        # handle seed as int case
        if not hasattr(rng, "normal"):
            rng = np.random.default_rng(rng)

        # generate the seeds for the signals
        seeds = []
        try:
            rand_int = rng.integers
        except AttributeError:
            rand_int = rng.randint
        for i in range(self.n_signals):
            seeds.append(rand_int(0, sys.maxsize))
        self.signal_seeds = seeds
        self.normalize = normalize

        # keep track of SemiMarkov keywords
        self.semi_markov_kws = copy.copy(kwargs)
        # ...but ensure that the maximum dwell time does not exceed snippet length
        max_dwell = self.semi_markov_kws.get("max_dwell", np.inf)
        if np.size(max_dwell) == 1:
            max_dwell = np.repeat(max_dwell, len(self.snippets))
        else:
            max_dwell = np.array(max_dwell)
        for crt_max, crt_snippet in zip(max_dwell, self.snippets):
            max_dwell = min(crt_max, len(crt_snippet))
        self.semi_markov_kws["max_dwell"] = max_dwell

        self.hdf_skip_contents = True

    def __len__(self) -> int:
        return self.n_signals

    def __iter__(self) -> "RandomSnippetDatasetIterator":
        return RandomSnippetDatasetIterator(self)

    def __getitem__(self, idx: int) -> SwitchingSnippetSignal:
        """ Generate a signal.

        Parameters
        ----------
        idx
            The index of the signal to generate.

        Returns a `SwitchingSnippetSignal` instance. Note that the signal is generated
        anew every time it is accessed, making this a potentially time-consuming
        operation.
        """
        if idx < 0:
            idx = self.n_signals + idx
        if idx < 0 or idx >= self.n_signals:
            raise IndexError("index out of range")

        seed = self.signal_seeds[idx]
        rng = np.random.default_rng(seed)

        # create an ArmaHSMM instance
        semi_markov = SemiMarkov(len(self.snippets), rng=rng, **self.semi_markov_kws)
        usage_seq = semi_markov.sample(self.n_samples)

        y = rng.normal(size=self.n_samples)
        if self.normalize:
            scale = 1.0 / np.std(y)
            y *= scale
        else:
            scale = 1.0

        return SwitchingSnippetSignal(y=y, usage_seq=usage_seq, scale=scale)

    def __str__(self) -> str:
        s = (
            f"RandomSnippetDataset(n_signals={self.n_signals}, "
            f"n_samples={self.n_samples})"
        )
        return s

    def __repr__(self) -> str:
        r = (
            f"RandomSnippetDataset(n_signals={self.n_signals}, "
            + f"n_samples={self.n_samples}, "
            + f"signal_seeds={repr(self.signal_seeds)}, "
            + f"semi_markov_kws={repr(self.semi_markov_kws)})"
        )
        return r


class RandomSnippetDatasetIterator(object):
    """ Iterator for RandomSnippetDataset. """

    def __init__(self, dataset: RandomSnippetDataset):
        self._dataset = dataset
        self._idx = 0

    def __next__(self) -> SwitchingSnippetSignal:
        if self._idx < len(self._dataset):
            res = self._dataset[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration
