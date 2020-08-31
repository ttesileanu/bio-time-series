""" Define functions for assessing the quality of time series clustering results. """

import numpy as np

from typing import Callable, Optional, Sequence, Tuple


def calculate_sliding_score(
    metric: Callable,
    labels_true: Sequence,
    labels_pred: Sequence,
    window_size: int = 1000,
    overlap_fraction: float = 0.5,
    step: Optional[int] = None,
    progress: Optional[Callable] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Calculate clustering metric in a rolling window.

    Parameters
    ----------
    metric
        Metric to evaluate. Will be called with `metric(true, pred)` and should return a
        scalar. Many metrics from `sklearn.metrics` obey these requirements.
    labels_true
        Ground-truth class labels.
    labels_pred
        Cluster labels to evaluate.
    window_size
        Size of the rolling window.
    overlap_fraction
        Fraction of rolling window to overlap for consecutive metric evaluations. Put
        differently, the `step` of the rolling window (if not specified directly; see
        below) is set to `max(window_size - int(overlap_fraction * window_size), 1)`.
    step
        Step for the rolling window. If specified, this overrides `overlap_fraction`.
    progress
        Callable to act as an iterator wrapper for displaying a progress bar (e.g.,
        tqdm).

    Returns a tuple of arrays `(window_loc, score)`, with `score[i]` representing the
    value of the `metric` evaluated for the window starting at `window_loc[i]`. Only
    windows that fully fit within the length of `labels_...` are considered.
    """
    # ensure labels are the same length
    n = len(labels_true)
    if n != len(labels_pred):
        raise ValueError("Lengths of labels_true and labels_pred do not match.")

    if step is None:
        step = max(window_size - int(overlap_fraction * window_size), 1)

    window_loc = np.arange(0, n - window_size + 1, step)
    score = np.zeros(len(window_loc))

    it_loc = window_loc
    if progress is not None:
        it_loc = progress(it_loc)
    for i, loc in enumerate(it_loc):
        crt_range = slice(loc, loc + window_size)
        crt_true = labels_true[crt_range]
        crt_pred = labels_pred[crt_range]
        score[i] = metric(crt_true, crt_pred)

    return window_loc, score
