""" Define a mixture ARMA model. """

import numpy as np

from bioslds import sources
from bioslds.utils import rle_encode

from typing import Sequence, Tuple, Union, Callable, Optional


def sample_switching_models(
    models: Sequence,
    usage_seq: Sequence,
    U: Union[None, Sequence, Callable] = None,
    initial_conditions: Optional[Tuple[Sequence, Sequence]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
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

    Returns a tuple `(Y, U)` of generated output samples and input samples. If
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
                history_y = np.hstack(
                    (initial_conditions[0][-n_left:], Y_ret[:ptr])
                )
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
                history_u = np.hstack(
                    (initial_conditions[1][-n_left:], U_ret[:ptr])
                )
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
        crt_y, crt_u = model.transform(n_samples, U=U)

        Y_ret[ptr : ptr + n_samples] = crt_y

        if not have_U_ret:
            U_ret[ptr : ptr + n_samples] = crt_u

        ptr += n_samples

    return Y_ret, U_ret
