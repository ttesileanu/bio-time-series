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
    length_mismatch_policy: str = "align_end",
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
    length_mismatch_policy
        This sets how to handle length mismatches between `labels_true` and
        `labels_pred`. It can be
            "raise":        Exception (`ValueError`) is raised.
            "align_start":  Align sequences at start, ignoring end of longer sequence.
            "align_end":    Align sequences at end, ignoring beginning of longer one.
        In either of the last two cases, the indices from `window_loc` match positions
        in `labels_true`.

    Returns a tuple of arrays `(window_loc, score)`, with `score[i]` representing the
    value of the `metric` evaluated for the window starting at `window_loc[i]` (these
    indices refer to positions in `labels_true`; see `length_mismatch_policy`). Only
    windows that fully fit within the length of `labels_...` are considered.
    """
    # ensure labels are the same length
    n_pred = len(labels_pred)
    n_true = len(labels_true)
    n = min(n_pred, n_true)
    idx_shift = 0
    if n_pred != n_true:
        if length_mismatch_policy == "raise":
            raise ValueError("Lengths of labels_true and labels_pred do not match.")
        elif length_mismatch_policy == "align_start":
            labels_true = labels_true[:n]
            labels_pred = labels_pred[:n]
        elif length_mismatch_policy == "align_end":
            idx_shift = len(labels_true) - n

            labels_true = labels_true[-n:]
            labels_pred = labels_pred[-n:]
        else:
            raise ValueError("Unknown length_mismatch_policy.")

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

    # adjust indices so they match original labels_true positions
    window_loc += idx_shift

    return window_loc, score
