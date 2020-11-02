""" Define a mixture ARMA model. """

import numpy as np

from bioslds import sources
from bioslds.utils import rle_encode
from bioslds.markov import SemiMarkov

from typing import Sequence, Tuple, Union, Callable, Optional


def sample_switching_models(
    models: Sequence,
    usage_seq: Sequence,
    U: Union[None, Sequence, Callable] = None,
    initial_conditions: Optional[Tuple[Sequence, Sequence]] = None,
    return_input: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """ Sample from a non-stationary stochastic processes that switches between
    different ARMA models at given times.

    This functions sets the models' `history_` attribute appropriately to ensure
    consistency across time.

    Parameters
    ----------
    models
        Sequence of models to use.
    usage_seq
        Sequence identifying the model to use at each time steps. Models are
        labeled from `0` to `len(models) - 1`.
    U
        If given, this overrides the input source for the models. If it is a
        sequence, it should be at least as long as `len(usage_seq)`.
    initial_conditions
        A tuple, `(initial_y, initial_u)`, of recent samples of the output and
        input sequences used to seed the simulation. If these are not provided,
        they are assumed equal to zero.
    return_input
        If true, returns both output and input. If false (the default), returns only
        the output.

    Returns a sequence `Y` of generated samples. If `return_input` is true,
    returns a  tuple `(Y, U)` of generated output samples and input samples. If
    the `U` parameter was used and was a sequence, the output `U` simply mirrors
    the input.
    """
    # check the inputs
    if len(models) == 0:
        raise ValueError("No models given.")

    if np.min(usage_seq) < 0 or np.max(usage_seq) >= len(models):
        raise ValueError("Invalid entry in usage_seq vector.")

    # handle vector U
    if U is not None and not callable(U):
        if len(U) < len(usage_seq):
            raise ValueError("Not enough input values in U.")

        U_ret = U
        U = sources.Stream(U)
        have_U_ret = True
    else:
        U_ret = np.zeros(len(usage_seq))
        have_U_ret = False

    # handle default initial conditions
    if initial_conditions is None:
        initial_conditions = ([], [])

    # generate the samples
    Y_ret = np.zeros(len(usage_seq))
    usage_rle = rle_encode(usage_seq)
    ptr = 0
    for model_id, n_samples in usage_rle:
        model = models[model_id]

        # ensure proper history
        if ptr >= model.p:
            history_y = np.copy(Y_ret[ptr - model.p : ptr])
        else:
            n_left = model.p - ptr
            if len(initial_conditions[0]) >= n_left:
                history_y = np.hstack((initial_conditions[0][-n_left:], Y_ret[:ptr]))
            else:
                history_y = np.hstack(
                    (
                        np.zeros(n_left - len(initial_conditions[0])),
                        initial_conditions[0],
                        Y_ret[:ptr],
                    )
                )
        if ptr >= model.q:
            history_u = np.copy(U_ret[ptr - model.q : ptr])
        else:
            n_left = model.q - ptr
            if len(initial_conditions[1]) >= n_left:
                history_u = np.hstack((initial_conditions[1][-n_left:], U_ret[:ptr]))
            else:
                history_u = np.hstack(
                    (
                        np.zeros(n_left - len(initial_conditions[1])),
                        initial_conditions[1],
                        U_ret[:ptr],
                    )
                )

        model.history_ = (history_y, history_u)

        # generate and store the samples from this model
        crt_y, crt_u = model.transform(n_samples, U=U, return_input=True)

        Y_ret[ptr : ptr + n_samples] = crt_y

        if not have_U_ret:
            U_ret[ptr : ptr + n_samples] = crt_u

        ptr += n_samples

    if return_input:
        return Y_ret, U_ret
    else:
        return Y_ret


class ArmaHSMM(object):
    """ A hidden semi-Markov model with ARMA emissions.

    This class can be used to generate samples from a non-stationary stochastic
    process that stochastically switches between several ARMA processes based on
    a hidden semi-Markov model.

    Attributes
    ==========
    models
        Sequence of models to use.
    smm
        Semi-Markov model used to generate ARMA states.
    """

    def __init__(self, models: Sequence, **kwargs):
        """ Initialize the ARMA HSMM.

        Parameters
        ----------
        models
            Sequence of models to use. This sets the number of states in the
            semi-Markov model.
        All other keyword arguments are passed to the semi-Markov model
        constructor.
        """
        self.models = models
        self.smm = SemiMarkov(len(models), **kwargs)

    def transform(
        self,
        n_samples: Optional[int] = None,
        U: Union[None, Sequence, Callable] = None,
        initial_conditions: Optional[Tuple[Sequence, Sequence]] = None,
        return_input: bool = False,
        return_usage_seq: bool = False,
    ) -> Union[
        np.ndarray,
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray],
    ]:
        """ Process input samples.

        The function uses exactly `n_samples` input samples.

        If no input source is explicitly provided, the default source for each
        of the ARMAs is used. An exception is raised if a process needs to be
        used that does not have a default source.

        Parameters
        ----------
        n_samples
            Number of samples to generate. If not provided, `U` must be provided
            and it must be a sequence.
        U
            Input samples or input generator. See `Arma.transform`.
        initial_conditions
            A tuple, `(initial_y, initial_u)`, of recent samples of the output
            and input sequences used to seed the simulation. If these are not
            provided, they are assumed equal to zero.
        return_input
            If true, returns both output and input. If false (the default), returns only
            the output.
        return_usage_seq
            If true, returns the `usage_seq` in addition to output (and potentially
            input).

        Returns either a single array (`Y`) if `return_input` and `return_usage_seq` are
        both false; or a tuple `(Y, U)` or `(Y, usage_sea)` if only `return_input` or
        only `return_usage_seq` is true, respectively; or a tuple `(Y, U, usage_seq)` if
        both are true. Here `Y` is an array of generated `y`; `U` contains the input `u`
        samples; and `usage_seq` is an integer array indicating which model was used at
        each time step. If the `U` parameter was used and was a sequence, the output `U`
        simply mirrors the input `U`.
        """
        # check inputs
        if n_samples is None:
            if U is None or not hasattr(U, "__len__"):
                raise ValueError("Need either n_samples or sequence U.")
            n_samples = len(U)

        # generate usage sequence, then use sample_switching_models
        usage_seq = self.smm.sample(n_samples)
        y, u = sample_switching_models(
            self.models,
            usage_seq,
            U=U,
            initial_conditions=initial_conditions,
            return_input=True,
        )

        res = (y,)
        if return_input:
            res = res + (u,)
        if return_usage_seq:
            res = res + (usage_seq,)

        if len(res) == 1:
            return res[0]
        else:
            return res

    def __repr__(self) -> str:
        r = f"ArmaHSMM(models={repr(self.models)}, smm={repr(self.smm)})"
        return r

    def __str__(self) -> str:
        s = f"ArmaHSMM(models={str(self.models)}, smm={str(self.smm)})"
        return s
