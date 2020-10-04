""" Define functions for assessing the quality of time series clustering results. """

import numpy as np
import scipy.optimize as sciopt

from typing import Callable, Optional, Sequence, Tuple, Union


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


def unordered_accuracy_score(
    y_true: Sequence,
    y_pred: Sequence,
    normalize: bool = True,
    return_assignment: bool = False,
) -> Union[Union[int, float], Tuple[float, np.ndarray]]:
    """ Calculate permutation-invariant clustering accuracy score.

    Returns the highest accuracy score across all possible assignments of prediction
    labels to ground-truth ones. The score is either the number of matching pairs (after
    permutation), or the fraction, depending on the value of `normalize`.

    Parameters
    ----------
    y_true
        Ground-truth class labels.
    y_pred
        Cluster labels to evaluate.
    normalize
        If true, the fraction of matching ground-truth and predicted labels is returned,
        after performing the best-matching permutation. If false, the number of matches
        is returned.
    return_assignment
        In addition to accuracy score, also return the best assignment of predicted to
        ground-truth labels.

    Returns the number or fraction of matching ground-truth and predicted labels, after
    performing the best permutation. If `return_assignment` is true, also returns the
    best assignment, as an array such that `best_assignment[i]` is the ground-truth
    label that best matches predicted label `i`.
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths.")
    if len(y_true) == 0:
        if not return_assignment:
            return 1
        else:
            return 1, np.asarray([])

    # ensure we have arrays
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # guess number of clusters
    n_clusters = max(np.max(y_true), np.max(y_pred)) + 1

    # calculate accuracy matrix
    accuracy_matrix = np.empty((n_clusters, n_clusters))
    for i in range(n_clusters):
        mask_pred = y_pred == i
        for j in range(n_clusters):
            mask_true = y_true == j
            accuracy_matrix[i, j] = np.mean(mask_true & mask_pred)

    # find optimal assignment
    # negative sign because scipy function finds minimum
    # first return value, row_ind, is trivial when argument is square
    _, best_assignment = sciopt.linear_sum_assignment(-accuracy_matrix)

    # calculate best accuracy score
    y_pred_shuf = best_assignment[y_pred]

    if normalize:
        # noinspection PyTypeChecker
        score: float = np.mean(y_true == y_pred_shuf)
    else:
        # noinspection PyTypeChecker
        score: int = np.sum(y_true == y_pred_shuf)

    if not return_assignment:
        return score
    else:
        return score, best_assignment
