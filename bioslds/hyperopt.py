""" Define functions for basic hyperparameter optimization. """

import numpy as np

from typing import Callable, Optional, Union, Tuple, Sequence


def random_maximize(
    fct: Callable,
    param_ranges: dict,
    n_iter: int,
    rng: Union[int, np.random.Generator, np.random.RandomState] = 0,
    log_scale: Optional[Sequence] = None,
    progress: Optional[Callable] = None,
) -> Tuple[float, dict, dict]:
    """ Evaluate a function at a set of randomly generated parameter values, finding the
    parameters yielding the maximum value.

    Parameters
    ----------
    fct
        Function whose output needs to be maximized. If the function outputs a tuple,
        the first return value is used for maximization, while the others are stored in
        the detailed output.
    param_ranges
        Parameter names and ranges for the optimization. All parameters will be passed
        to `fct` by name. Currently all values are drawn uniformly at random from the
        given ranges (including lower edge and excluding the upper one for both `float`s
        and `int`s). Variable types are inferred from the ranges. Only two types are
        supported currently: `int` and `float`. `int` is chosen if *both* ends of the
        range are `int`s, `float` if at least one is not an `int`.
    n_iter
        Number of random parameter sets to generate.
    rng
        Random number generator to use, or integer seed. If a seed, a generator is
        created using `np.random.default_rng`.
    log_scale
        List of parameter names that should be sampled uniformly on a log scale instead
        of the usual linear one. This assumes that both ends of the range for that
        parameter are strictly positive. For integer parameters, values are first
        sampled on a continuous log scale and then rounded to the nearest integer.
    progress
        Callable to use for progress tracking.

    Returns a tuple, `(max, argmax, details)` where `max` is the maximum score that was
    reached, `argmax` is a `dict` containing the parameters leading to the maximum
    score, and `details` is a dictionary containing more information about all values
    that were tried. Specifically, it contains the following keys:
        "params":       sequence of dictionaries containing the used parameter sets
        "fct_scalar":   sequence of the scalar outputs of the function `fct`
        "fct_full":     sequence of full outputs of the function `fct`
    """
    # handle seed form of rng
    if not hasattr(rng, "uniform"):
        rng = np.random.default_rng(rng)

    # handle optional log_scale
    if log_scale is None:
        log_scale = []

    # figure out which random function to use for each parameter
    param_rng = {}
    for name, (low, high) in param_ranges.items():
        if isinstance(low, int) and isinstance(high, int):
            if name in log_scale:
                param_rng[name] = lambda llo=np.log(low), lhi=np.log(high): int(
                    np.round(np.exp(rng.uniform(llo, lhi)))
                )
            elif low == high:
                param_rng[name] = lambda out=low: out
            elif hasattr(rng, "integers"):
                param_rng[name] = lambda lo=low, hi=high: rng.integers(lo, hi)
            else:
                param_rng[name] = lambda lo=low, hi=high: rng.randint(lo, hi)
        else:
            if name not in log_scale:
                param_rng[name] = lambda lo=low, hi=high: rng.uniform(lo, hi)
            else:
                param_rng[name] = lambda llo=np.log(low), lhi=np.log(high): np.exp(
                    rng.uniform(llo, lhi)
                )

    # generate random parameters
    params = []
    for i in range(n_iter):
        crt_params = {}
        for name in param_rng:
            crt_params[name] = param_rng[name]()

        params.append(crt_params)

    # evaluate the function
    fct_scalar = np.zeros(n_iter)
    fct_full = []
    if progress is not None:
        it = progress(params)
    else:
        it = params

    for i, crt_params in enumerate(it):
        crt_full_output = fct(**crt_params)
        if hasattr(crt_full_output, "__len__"):
            crt_scalar = crt_full_output[0]
        else:
            crt_scalar = crt_full_output

        fct_scalar[i] = crt_scalar
        fct_full.append(crt_full_output)

    # find the best
    # noinspection PyTypeChecker
    best_idx: int = np.argmax(fct_scalar)
    best_fct = fct_scalar[best_idx]
    best_params = params[best_idx]

    # build the output
    details = {
        "params": params,
        "fct_scalar": fct_scalar,
        "fct_full": fct_full,
    }

    return best_fct, best_params, details
