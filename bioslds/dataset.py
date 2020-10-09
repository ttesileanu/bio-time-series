""" Define classes for generating test datasets. """

import sys

import copy
import numpy as np

from bioslds.arma import make_random_arma
from bioslds.arma_hsmm import ArmaHSMM
from bioslds import sources

from typing import Sequence, Optional, Union, Tuple, Callable


class SwitchingSignal:
    # noinspection PyUnresolvedReferences
    """ A signal generated from switching dynamical processes.

    Attributes
    ==========
    y : np.ndarray
        The signal.
    u : np.ndarray
        The source signal.
    usage_seq : np.ndarray
        A vector of integers indicating which model was used at each time step.
    """

    def __init__(self, y: np.ndarray, u: np.ndarray, usage_seq: np.ndarray):
        self.y = y
        self.u = u
        self.usage_seq = usage_seq

    def __repr__(self) -> str:
        return f"SwitchingSignal(y={self.y}, u={self.u}, usage_seq={self.usage_seq})"


class RandomArmaDataset(Sequence[SwitchingSignal]):
    """ Generate random Arma datasets on demand.

    This acts like an iterable, allowing sequential or random access to randomly-
    generated signals. Instead of being stored in memory, like in a list, the signals
    are generated on-demand, every time an element is accessed.

    Attributes
    ==========
    n_signals : int
        Number of signals in the dataset. This is the same as `len(self)`.
    n_samples : int
        Number of samples in each signal.
    armas : Sequence
        Sequence of sequences of ARMA processes used to generate each signal.
    signal_seeds : Sequence
        Random seeds to use for generating the signals.
    source : callable
        Callable used to generate source data. When this is `None`, a Gaussian source is
        created for each signal.
    initial_conditions : Optional[Tuple]
        Tuple of initial conditions, to be passed to `ArmaHSMM`.
    arma_hsmm_kws : dict
        Additional arguments to be passed to `ArmaHSMM` constructor.
    """

    def __init__(
        self,
        n_signals: int,
        n_samples: int,
        arma_orders: Optional[Sequence] = None,
        armas: Optional[Sequence] = None,
        arma_kws: Optional[dict] = None,
        rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
        source: Optional[Callable] = None,
        fix_scale: Optional[float] = None,
        initial_conditions: Optional[Tuple[Sequence, Sequence]] = None,
        **kwargs,
    ):
        """ Initialize the dataset.

        Parameters
        ----------
        n_signals
            Number of signals in the dataset. This will be the dataset's length.
        n_samples
            Number of samples in each signal.
        arma_orders
            Sequence of pairs `(p, q)` indicating the order of each ARMA process to use.
            The number of such pairs gives the number of models that will be used to
            generate each signal. Note that new random processes are generated for each
            signal in the dataset, using `make_random_arma`. Additional keywords to
            `make_random_arma` are passed according to `arma_kws`. Either `arma_orders`
            or `armas` must be provided.
        armas
            This is a sequence of ARMA process, giving the models that are to be used to
            generate each signal. This is directly used with `ArmaHSMM` instead of
            generating random processes when using `arma_orders`. Either `arma_orders`
            or `armas` must be provided.
        arma_kws
            Keyword arguments to pass to `make_random_arma` when `arma_orders` is used.
        rng
            Random number generator or seed to use to generate the datasets. The precise
            way in which this is used is not specified, but it is guaranteed that
            starting with the same state (or seed) will give the same results when using
            the same version of Numpy (similar to the guarantees from
            `np.random.Generator`).
        source
            Callable used to generate source data. This should take an integer keyword
            argument `size` and return an array of that length. If not provided,
            Gaussian noise with zero mean and standard deviation equal to 1 is used,
            with random seed chosen such that the source for every signal in the dataset
            is different, but reproducible every time it is accessed. Note that if a
            custom `source`, `copy.deepcopy` is used before every use, in an attempt to
            keep the signal reproducible. However, the reproducibility guarantee cannot
            always be enforced (e.g., if `source` uses Numpy's global pseudorandom
            generator, every call will yield a different output).
        fix_scale
            If given, use different sources for each ARMA and adjust their scales such
            that the output standard deviation is equal to `fix_scale`. This is used
            only if `source` is not used.
        initial_conditions
            A tuple, `(initial_y, initial_u)`, of recent samples of the output and input
            sequences used to seed the simulation. This is directly passed to
            `ArmaHSMM`.
        All other keyword arguments will be passed to `ArmaHSMM` constructor.
        """
        self.n_signals = n_signals
        self.n_samples = n_samples

        # handle seed as int case
        if not hasattr(rng, "normal"):
            rng = np.random.default_rng(rng)

        # generate ARMA processes
        if armas is not None:
            self.armas = self.n_signals * [armas]
        elif arma_orders is not None:
            self._generate_armas(rng, arma_orders, arma_kws)
        else:
            raise ValueError("Need either armas or arma_orders.")

        # generate the seeds for the signals
        seeds = []
        try:
            rand_int = rng.integers
        except AttributeError:
            rand_int = rng.randint
        for i in range(self.n_signals):
            seeds.append(rand_int(0, sys.maxsize))
        self.signal_seeds = seeds

        # if asked to, figure out correct scales
        self.noise_scales = None
        if fix_scale is not None:
            self._figure_out_scales(fix_scale)

        self.source = source
        self.initial_conditions = initial_conditions

        self.arma_hsmm_kws = copy.copy(kwargs)

    def _figure_out_scales(self, fix_scale: float):
        """ Use fix_source_scale to normalize ARMA outputs. """
        all_scales = []
        for sig_arma in self.armas:
            sig_scales = []
            for arma in sig_arma:
                scale = sources.fix_transformer_scale(arma, output_std=fix_scale)

                # # switch to a new source for scale calculation (but only temporarily)
                # old_source = arma.default_source
                # arma.default_source = sources.GaussianNoise()
                # scale = sources.fix_source_scale(
                #     arma, output_std=fix_scale, use_copy=False
                # )

                sig_scales.append(scale)

                # return to the Arma process's original source
                # arma.default_source = old_source

            all_scales.append(sig_scales)

        self.noise_scales = np.asarray(all_scales)

    def _generate_armas(
        self,
        rng: Union[np.random.RandomState, np.random.Generator],
        arma_orders: Optional[Sequence] = None,
        arma_kws: Optional[dict] = None,
    ):
        """ Generate the ARMA processes to use for each signal. """
        # handle None arma_kws
        if arma_kws is None:
            arma_kws = {}

        armas = []
        for i in range(self.n_signals):
            crt_armas = []
            for crt_order in arma_orders:
                crt_arma = make_random_arma(*crt_order, rng=rng, **arma_kws)
                crt_armas.append(crt_arma)

            armas.append(crt_armas)

        self.armas = armas

    def __len__(self) -> int:
        return self.n_signals

    def __iter__(self) -> "RandomArmaDatasetIterator":
        return RandomArmaDatasetIterator(self)

    def __getitem__(self, idx: int) -> SwitchingSignal:
        """ Generate a signal.

        Parameters
        ----------
        idx
            The index of the signal to generate.

        Returns a `SwitchingSignal` instance. Note that the signal is generated anew
        every time it is accessed, making this a potentially time-consuming operation.
        """
        if idx < 0:
            idx = self.n_signals + idx
        if idx < 0 or idx >= self.n_signals:
            raise IndexError("index out of range")

        # create an ArmaHSMM instance
        seed = self.signal_seeds[idx]
        rng = np.random.default_rng(seed)
        arma_hsmm = ArmaHSMM(self.armas[idx], rng=rng, **self.arma_hsmm_kws)

        if self.source is not None:
            source = copy.deepcopy(self.source)
        else:
            source_seed = rng.integers(0, sys.maxsize)
            source_rng = np.random.default_rng(source_seed)
            source = sources.GaussianNoise(rng=source_rng)

        y, u, usage_seq = arma_hsmm.transform(
            self.n_samples, U=source, initial_conditions=self.initial_conditions
        )

        return SwitchingSignal(y, u, usage_seq)

    def __str__(self) -> str:
        s = f"RandomArmaDataset(n_signals={self.n_signals}, n_samples={self.n_samples})"
        return s

    def __repr__(self) -> str:
        r = (
            f"RandomArmaDataset(n_signals={self.n_signals}, "
            + f"n_samples={self.n_samples}, "
            + f"signal_seeds={repr(self.signal_seeds)}, "
            + f"arma_hsmm_kws={repr(self.arma_hsmm_kws)})"
        )
        return r


class RandomArmaDatasetIterator(object):
    """ Iterator for RandomArmaDataset. """

    def __init__(self, dataset: RandomArmaDataset):
        self._dataset = dataset
        self._idx = 0

    def __next__(self) -> SwitchingSignal:
        if self._idx < len(self._dataset):
            res = self._dataset[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration
