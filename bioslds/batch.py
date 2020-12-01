""" Define functions for running batches of simulations. """

import copy
import sys

import numpy as np

from types import SimpleNamespace
from typing import Sequence, Optional, Callable, Union, Tuple

from bioslds.regressor_utils import transform_ar
from bioslds.monitor import AttributeMonitor


def hyper_score_ar(
    regressor_class: Callable,
    dataset: Sequence,
    metric: Callable[[np.ndarray, np.ndarray], float],
    rng: Union[int, np.random.RandomState, np.random.Generator] = 0,
    fit_kws: Optional[dict] = None,
    initial_weights: str = "default",
    monitor: Optional[Sequence] = None,
    monitor_step: int = 1,
    progress: Optional[Callable] = None,
    progress_trial: Optional[Callable] = None,
    test_fraction: float = 0.2,
    test_samples: Optional[int] = None,
    **kwargs,
) -> Tuple[float, SimpleNamespace]:
    """ Score hyperparameter choice for AR clustering.

    This runs a time series clustering algorithm on a set of signals and assesses the
    accuracy of the output using a given metric. This is done using a given set of
    hyperparameter values for the algorithm. The function then returns a summary score.

    Parameters
    ----------
    regressor_class
        Class to use to create regressors. One regressor is created for each signal in
        the dataset, and `transform_ar` is used to fit the regressor on each signal.
        Additional keyword arguments passed to `hyper_score_ar` are transferred to the
        `regressor_class` constructor -- you'll probably need at least `n_models` and
        `n_features`. A random number generator is chosen automatically by
        `hyper_score_ar` and passed as the `rng` argument to the constructor -- note
        that this will not pass `hyper_score_ar`s `rng` directly!
    dataset
        Sequence of signals to use for testing the regressors. Each entry in `dataset`
        should be an object containing at least fields `y` and `usage_seq`, which should
        be arrays of equal length. `y` is the signal, `usage_seq` gives the ground truth
        of which generating model was used at each time step.
    metric
        The function to use to assess the accuracy of the clustering. This should take a
        vector `labels_true` of ground-truth latent states at each time step and a
        vector `labels_pred` of predicted (inferred) latent states, and return a scalar
        accuracy score.
    rng
        Random number generator or seed to use for generating initial weight values. If
        seed, a random number generator is created using `np.random.default_rng`.
    fit_kws
        Additional argument to pass to `transform_ar`.
    initial_weights
        How to set the initial weights (for regressors that have that option). This can
        be
          "default":    do the default, i.e., no `weights` attribute passed to __init__
          "oracle_ar":  pass the `a` coefficients from the signal's `armas` member as
                        `weights`.
    monitor
        Sequence of strings denoting values to monitor during the inference procedure
        for each regressor and signal pair. The special value "r" can be used to also
        store continuous latent-state assignments that are returned as the first output
        of `transform` / `transform_ar`.
    monitor_step
        How often to record the values from `monitor`.
    progress
        Progress callable for monitoring how far the function is in running the
        regressor on each signal from the `dataset`. This is used to wrap an iterator,
        `progress(iterator) -> iterator`.
    progress_trial
        Callable for monitoring the progress during each trial. This is directly passed
        to `transform_ar` (and thus to `regressor_class().transform`).
    test_fraction
        Fraction of samples to use when estimating the clustering score.
    test_samples
        Number of samples to use when estimating the clustering score. If this is
        provided, it overrides `test_fraction`
    Additional keyword arguments are directly passed to the `regressor_class`
    constructor.

    Returns a tuple `(summary_score, details)`, where `summary_score` is the median of
    the scores assigned for the regressor output on each signal in the `dataset`, and
    `details` is a `SimpleNamespace` containing more details from the run:
        trial_scores (np.ndarray)       -- The scores for each trial.
        regressor_seeds (np.ndarray)    -- Random seeds used to initialize regressors.
        regressors (Sequence)           -- Regressors used for each trial.
        history (Sequence)              -- Monitoring data for each trial.
    """
    # handle seed form of rng
    if not hasattr(rng, "normal"):
        rng = np.random.default_rng(rng)

    # handle optional fit_kws
    if fit_kws is None:
        fit_kws = {}
    else:
        fit_kws = copy.copy(fit_kws)
    fit_kws.setdefault("progress", progress_trial)

    # handle optional monitor
    if monitor is None:
        monitor = []

    # handle monitoring of transform output
    if "r" in monitor:
        monitor = [_ for _ in monitor if _ != "r"]
        store_r = True
    else:
        store_r = False

    # ensure monitor_step makes sense
    monitor_step = max(monitor_step, 1)

    # set up trial scores
    n_trials = len(dataset)
    trial_scores = np.zeros(n_trials)

    # set up regressor random seeds
    if hasattr(rng, "randint"):
        gen_integers = rng.randint
    else:
        gen_integers = rng.integers
    regressor_seeds = gen_integers(0, sys.maxsize, size=n_trials)

    # run the simulations
    it = dataset
    if progress is not None:
        it = progress(it)
    history = []
    regressors = []
    for i, signal in enumerate(it):
        # create the regressor
        crt_args = copy.copy(kwargs)

        crt_seed = regressor_seeds[i]
        crt_rng = np.random.default_rng(crt_seed)
        crt_args["rng"] = crt_rng

        if initial_weights == "oracle_ar" and not hasattr(crt_args, "weights"):
            crt_args["weights"] = np.asarray([_.a for _ in signal.armas])

        regressor = regressor_class(**crt_args)

        # run transform_ar with this regressor
        crt_monitor = monitor
        if len(crt_monitor) > 0 and monitor_step > 1:
            crt_monitor = AttributeMonitor(monitor, step=monitor_step)

        crt_r, crt_history = transform_ar(
            regressor, signal.y, monitor=crt_monitor, **fit_kws
        )
        if store_r:
            crt_history.r = crt_r[::monitor_step]
        history.append(crt_history)

        crt_inferred_usage = np.argmax(crt_r, axis=1)

        # score only the last test_fraction / test_samples samples
        if test_samples is None:
            crt_n = int(test_fraction * len(crt_r))
        else:
            crt_n = test_samples
        crt_score = metric(signal.usage_seq[-crt_n:], crt_inferred_usage[-crt_n:])

        regressors.append(regressor)

        trial_scores[i] = crt_score

    # noinspection PyTypeChecker
    summary_score: float = np.median(trial_scores)
    details = SimpleNamespace(
        trial_scores=trial_scores,
        regressor_seeds=regressor_seeds,
        regressors=regressors,
        history=history,
    )

    return summary_score, details
