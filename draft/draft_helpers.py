""" Define some functions to help run analyses and make figures for the draft. """


import copy

from typing import Callable, Union, Optional, Sequence, List, Tuple
from types import SimpleNamespace

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import scipy.optimize as sciopt
import pickle

from tqdm.notebook import tqdm

from bioslds.dataset import RandomArmaDataset
from bioslds.cluster_quality import unordered_accuracy_score, calculate_sliding_score


paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]


def make_multi_trajectory_plot(
    results: SimpleNamespace,
    dataset: RandomArmaDataset,
    metric: Callable = unordered_accuracy_score,
    n_traces: int = 10,
    highlight_idx: Union[None, int, Sequence] = None,
    convergence_threshold: float = 0.9,
    sliding_kws: Optional[dict] = None,
    trace_kws: Optional[dict] = None,
    highlight_kws: Optional[dict] = None,
    fig_kws: Optional[dict] = None,
    rug_kws: Optional[dict] = None,
    kde_kws: Optional[dict] = None,
) -> Tuple:
    # handle defaults
    if sliding_kws is None:
        sliding_kws = {}
    if trace_kws is None:
        trace_kws = {}
    else:
        trace_kws = copy.copy(trace_kws)
    if highlight_kws is None:
        highlight_kws = {}
    if fig_kws is None:
        fig_kws = {}
    else:
        fig_kws = copy.copy(fig_kws)
    if rug_kws is None:
        rug_kws = {}
    else:
        rug_kws = copy.copy(rug_kws)
    if kde_kws is None:
        kde_kws = {}
    else:
        kde_kws = copy.copy(kde_kws)
    if highlight_idx is not None and not hasattr(highlight_idx, "__len__"):
        highlight_idx = [highlight_idx]

    fig_kws.setdefault("figsize", (5.76, 2.2))
    despine_kws = fig_kws.pop("despine_kws", {"offset": 5})

    with plt.style.context(paper_style):
        fig = plt.figure(**fig_kws)
        axs = np.empty((2, 3), dtype=object)
        axs[0, 0] = fig.add_axes([0.09, 0.62, 0.42, 0.34])
        axs[0, 1] = fig.add_axes([0.52, 0.62, 0.07, 0.34])
        axs[0, 2] = fig.add_axes([0.69, 0.22, 0.29, 0.74])
        axs[1, 0] = fig.add_axes([0.09, 0.22, 0.42, 0.23])

        # calculate rolling accuracy scores, unless they exist already
        if not hasattr(results, "rolling_scores"):
            results.rolling_scores = []
            for i, crt_dataset in enumerate(tqdm(dataset)):
                crt_r = results.history[i].r
                crt_inferred = np.argmax(crt_r, axis=1)

                crt_loc, crt_sliding_score = calculate_sliding_score(
                    metric, crt_dataset.usage_seq, crt_inferred, **sliding_kws
                )
                results.rolling_scores.append((crt_loc, crt_sliding_score))

        # calculate convergence times
        results.convergence_times = []
        results.convergence_idxs = []
        for idx, crt_rolling in enumerate(results.rolling_scores):
            crt_threshold = convergence_threshold * results.trial_scores[idx]
            crt_mask = crt_rolling[1] >= crt_threshold

            # find the first index where the score goes above threshold
            if not np.any(crt_mask):
                crt_conv_idx = len(crt_mask) - 1
            else:
                crt_conv_idx = np.nonzero(crt_mask)[0][0]

            results.convergence_idxs.append(crt_conv_idx)
            results.convergence_times.append(crt_rolling[0][crt_conv_idx])

        # draw the accuracy traces
        trace_idxs = np.unique(
            np.clip(
                np.round(np.linspace(0, len(dataset), n_traces)), 0, len(dataset) - 1
            ).astype(int)
        )
        if "c" not in trace_kws and "color" not in trace_kws:
            trace_kws["c"] = "C0"
        trace_color = trace_kws.get("color", trace_kws.get("c"))
        trace_kws.setdefault("alpha", 0.5)
        for idx, crt_rolling in enumerate(results.rolling_scores):
            if idx in trace_idxs:
                axs[0, 0].plot(*crt_rolling, **trace_kws)

        if highlight_idx is not None:
            # using different color for highlight
            trace_kws.pop("c", None)
            trace_kws.pop("color", None)
            trace_kws.pop("alpha", None)
            if "lw" in highlight_kws or "linewidth" in highlight_kws:
                trace_kws.pop("lw", None)
                trace_kws.pop("linewidth", None)
            else:
                def_lw = trace_kws.get("lw", trace_kws.get("linewidth", 1.0))
                highlight_kws["lw"] = 2 * def_lw

            trace_kws.update(highlight_kws)
            # if "c" not in trace_kws and "color" not in trace_kws:
            #     trace_kws["c"] = "C1"
            if "alpha" not in trace_kws:
                trace_kws["alpha"] = 1.0
            for h_idx, crt_idx in enumerate(highlight_idx):
                trace_kws["c"] = f"C{h_idx}"
                axs[0, 0].plot(*results.rolling_scores[crt_idx], **trace_kws)

        axs[0, 0].set_xlim(0, np.max(crt_rolling[0]))
        axs[0, 0].set_ylim(0.5, 1.0)
        axs[0, 0].set_xlabel("time step")
        axs[0, 0].set_ylabel("seg. score")
        axs[0, 0].set_xticks([])

        # draw the distribution of final accuracy scores
        kde_kws.setdefault("shade", True)
        kde_kws.setdefault("color", trace_color)
        sns.kdeplot(y=results.trial_scores, ax=axs[0, 1], **kde_kws)

        rug_kws.setdefault("height", 0.1)
        rug_kws.setdefault("lw", 0.5)
        sns.rugplot(y=results.trial_scores, color=trace_color, ax=axs[0, 1], **rug_kws)
        if highlight_idx is not None:
            highlight_rug_kws = copy.copy(rug_kws)
            highlight_rug_kws["alpha"] = 1.0
            highlight_rug_kws["lw"] = 2 * rug_kws["lw"]
            for h_idx, crt_idx in enumerate(highlight_idx):
                sns.rugplot(
                    y=[results.trial_scores[crt_idx]],
                    color=f"C{h_idx}",
                    ax=axs[0, 1],
                    **highlight_rug_kws,
                )

        axs[0, 1].set_ylim(0.5, 1.0)
        axs[0, 1].set_xticks([])
        axs[0, 1].set_xlabel("pdf")

        # draw the distribution of convergence times
        sns.kdeplot(x=results.convergence_times, ax=axs[1, 0], **kde_kws)

        rug_kws["height"] = 0.75 * rug_kws["height"]
        sns.rugplot(
            x=results.convergence_times, color=trace_color, ax=axs[1, 0], **rug_kws
        )
        if highlight_idx is not None:
            highlight_rug_kws = copy.copy(rug_kws)
            highlight_rug_kws["alpha"] = 1.0
            highlight_rug_kws["lw"] = 2 * rug_kws["lw"]
            for h_idx, crt_idx in enumerate(highlight_idx):
                sns.rugplot(
                    x=[results.convergence_times[crt_idx]],
                    color=f"C{h_idx}",
                    ax=axs[1, 0],
                    **highlight_rug_kws,
                )

        axs[1, 0].set_xlim(0, np.max(crt_rolling[0]))
        # axs[1, 0].set_xlabel("time step")
        axs[1, 0].set_xlabel("convergence time")
        axs[1, 0].set_ylabel("pdf")
        # axs[1, 0].annotate(
        #     "convergence time",
        #     xy=(0.5, 0.98),
        #     xycoords="axes fraction",
        #     horizontalalignment="center",
        #     verticalalignment="top",
        #     fontsize=plt.rcParams["axes.titlesize"],
        #     fontweight=plt.rcParams["axes.titleweight"],
        # )
        axs[1, 0].set_yticks([])

        axs[1, 0].patch.set_alpha(0)

        axs[0, 0].set_xticks([])
        axs[0, 1].set_yticks([])
        axs[0, 2].set_visible(False)

        sns.despine(ax=axs[0, 0], **despine_kws)
        sns.despine(left=True, ax=axs[0, 1], **despine_kws)
        sns.despine(ax=axs[0, 2], **despine_kws)
        sns.despine(ax=axs[1, 0], **despine_kws)

    return fig, axs


def make_accuracy_plot(
    results, oracle, dataset, special_idxs
) -> Tuple[plt.Figure, List[plt.Axes]]:
    with plt.style.context(paper_style):
        fig, axs = make_multi_trajectory_plot(
            results,
            dataset,
            n_traces=25,
            highlight_idx=special_idxs,
            sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
            trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
            rug_kws={"alpha": 0.3},
        )
        axs[0, 0].set_xticks(np.arange(0, 200_000, 50_000))
        axs[1, 0].set_xticks(np.arange(0, 200_000, 50_000))

        axs[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        scatter_ax = axs[0, 2]
        scatter_ax.set_visible(True)
        scatter_ax.plot([0.5, 1], [0.5, 1], "k--", lw=0.5, alpha=0.5)
        scatter_ax.scatter(
            oracle.trial_scores, results.trial_scores, c="gray", s=6, alpha=0.5,
        )
        for i, special_idx in enumerate(special_idxs):
            scatter_ax.scatter(
                [oracle.trial_scores[special_idx]],
                [results.trial_scores[special_idx]],
                s=12,
                c=f"C{i}",
                alpha=1.0,
            )

        scatter_ax.set_xlim(0.5, 1.0)
        scatter_ax.set_ylim(0.5, 1.0)
        scatter_ax.set_aspect(1.0)

        scatter_ax.set_xlabel("fixed weights")
        scatter_ax.set_ylabel("learned weights")

        scatter_ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
        scatter_ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

        scatter_ax.annotate(
            "segmentation\nscore",
            xy=(0.1, 1.00),
            xycoords="axes fraction",
            horizontalalignment="left",
            verticalalignment="top",
            fontsize=plt.rcParams["axes.titlesize"],
            fontweight=plt.rcParams["axes.titleweight"],
        )

    return fig, axs


def make_accuracy_comparison_diagram(
    ax: plt.Axes,
    results: Sequence,
    result_names: Sequence,
    dash_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
    color_cycle: Optional[Sequence] = None,
    score_type: str = "segmentation",
):
    """ score_type can be "segmentation" or "weight". """
    # handle defaults
    if dash_kws is None:
        dash_kws = {}
    if line_kws is None:
        line_kws = {}

    prev_scores = None
    for i, crt_res in enumerate(results):
        if score_type == "segmentation":
            crt_scores = crt_res[1].trial_scores
        elif score_type == "weight":
            crt_scores = [
                np.linalg.norm(_.weight_errors_normalized_[-1])
                for _ in crt_res[1].history
            ]
        else:
            raise ValueError("Unknown score_type.")

        n = len(crt_scores)
        crt_kws = copy.copy(dash_kws)
        if "ls" not in crt_kws and "linestyle" not in crt_kws:
            crt_kws["ls"] = "none"
        if "ms" not in crt_kws and "markersize" not in crt_kws:
            crt_kws["ms"] = 10
        if "mew" not in crt_kws and "markeredgewidth" not in crt_kws:
            crt_kws["mew"] = 2
        crt_kws.setdefault("marker", "_")
        crt_kws.setdefault("alpha", 0.5)
        if color_cycle is None:
            crt_color = f"C{i}"
        else:
            crt_color = color_cycle[i % len(results)]
        ax.plot(i * np.ones(n), crt_scores, c=crt_color, **crt_kws)

        if prev_scores is not None:
            crt_kws = copy.copy(line_kws)
            if "c" not in crt_kws and "color" not in crt_kws:
                crt_kws["c"] = "k"
            if "ls" not in crt_kws and "linestyle" not in crt_kws:
                crt_kws["ls"] = "-"
            if "lw" not in crt_kws and "linewidth" not in crt_kws:
                crt_kws["lw"] = 0.5
            crt_kws.setdefault("alpha", 0.5)
            ax.plot(
                np.row_stack(((i - 1) * np.ones(n), i * np.ones(n))),
                np.row_stack((prev_scores, crt_scores)),
                **crt_kws,
            )
        prev_scores = crt_scores

    ax.set_xticks(np.arange(len(results)))
    ax.set_xticklabels(result_names)

    if score_type == "segmentation":
        ax.set_ylim(0.5, 1)
        ax.set_ylabel("segmentation score")
    else:
        ax.set_ylabel("weight reconstruction error")


def calculate_ar_identification_progress(
    results: Sequence,
    dataset: RandomArmaDataset,
    test_fraction: float = 0.2,
    mapping_policy: str = "weight",
):
    """ Mapping policy can be "segmentation" or "weight". """
    for crt_res, crt_sig in zip(tqdm(results), dataset):
        # collect ground-truth weights
        ground_weights = np.array([_.a for _ in crt_sig.armas])
        p = len(ground_weights[0])
        ground_diff = np.linalg.norm(np.std(ground_weights, axis=0)) / np.sqrt(
            p / len(ground_weights)
        )

        # find the mapping from predicted to ground-truth labels
        inferred_usage = np.argmax(crt_res.r, axis=1)
        crt_n = int(test_fraction * len(crt_res.r))
        _, assignment0 = unordered_accuracy_score(
            crt_sig.usage_seq[-crt_n:], inferred_usage[-crt_n:], return_assignment=True
        )

        # it's more convenient to use ground-truth to predicted mapping
        assignment_segmentation = np.empty(len(assignment0), dtype=int)
        assignment_segmentation[assignment0] = np.arange(len(assignment0))

        crt_res.best_assignment_segmentation = assignment_segmentation

        # find out which inferred weights are closest to which ground-truth ones
        n_gnd = len(ground_weights)
        n_inf = np.shape(crt_res.weights_)[1]
        error_matrix = np.zeros((n_gnd, n_inf))
        for i in range(n_gnd):
            crt_gnd = ground_weights[i]
            for j in range(n_inf):
                crt_inf = crt_res.weights_[-1, j]
                crt_diff_norm = np.linalg.norm(crt_inf - crt_gnd) / ground_diff
                error_matrix[i, j] = crt_diff_norm
        _, assignment_weight = sciopt.linear_sum_assignment(error_matrix)

        crt_res.best_assignment_weight = assignment_weight

        if mapping_policy == "segmentation":
            assignment = assignment_segmentation
        elif mapping_policy == "weight":
            assignment = assignment_weight
        else:
            raise ValueError("Invalid mapping_policy.")

        crt_res.best_assignment = assignment

        # calculate norm of coefficient differences
        weights = crt_res.weights_[:, assignment, :]
        crt_res.weights_shuffled_ = weights

        diffs = weights - ground_weights[None, :]
        norms = np.linalg.norm(diffs, axis=2) / np.sqrt(p)

        crt_res.weight_errors_ = norms

        # normalize weight errors by difference between ARs
        crt_res.weight_errors_normalized_ = norms / ground_diff

        # calculate how different the two models are
        model_diff = np.linalg.norm(np.std(weights, axis=1), axis=1) / np.sqrt(p)
        model_diff_norm = model_diff / ground_diff
        crt_res.weight_diff_ = model_diff
        crt_res.weight_diff_normalized_ = model_diff_norm


def show_weight_progression(
    axs: Sequence,
    results: SimpleNamespace,
    true_armas: Sequence,
    window_step: int = 1000,
    window_size: int = 5000,
    use_same_range: bool = True,
):
    actual_weights = [_.a for _ in true_armas]
    inferred_weights = results.weights_shuffled_
    n = len(true_armas)
    ylims = []
    for i in range(n):
        ax = axs[i]
        crt_true = actual_weights[i]

        # smoothe the inferred weights
        for k in range(len(crt_true)):
            crt_inferred_weights = inferred_weights[:, i, k]
            crt_loc = np.arange(0, len(crt_inferred_weights), window_step)
            crt_smoothed = np.zeros(len(crt_loc))
            for j, crt_start in enumerate(crt_loc):
                crt_smoothed[j] = np.mean(
                    crt_inferred_weights[crt_start : crt_start + window_size]
                )

            ax.plot(
                crt_loc,
                crt_smoothed,
                f"C{k}",
                alpha=0.80,
                label=f"inferred $w_{k + 1}$",
            )
            ax.axhline(crt_true[k], c=f"C{k}", ls=":", lw=2, label=f"true $w_{k + 1}$")
            ylims.append(ax.get_ylim())

        ax.set_xlabel("time step")
        ax.set_ylabel("AR coefficients")

        # ax.legend(ncol=3, frameon=False, fontsize=6)
        ax.set_title(f"model {i + 1}")

    if use_same_range:
        max_ylims = (min(_[0] for _ in ylims), max(_[1] for _ in ylims))
        for ax in axs:
            ax.set_ylim(*max_ylims)


def calculate_smooth_weight_errors(res: SimpleNamespace, window_size: int = 5000):
    rolling_weight_errors = []
    rolling_weight_errors_normalized = []

    rolling_weight_diff = []
    rolling_weight_diff_normalized = []
    for crt_idx, crt_res in enumerate(tqdm(res.history)):
        crt_loc = res.rolling_scores[crt_idx][0]

        crt_err = crt_res.weight_errors_
        crt_err_norm = crt_res.weight_errors_normalized_
        crt_diff = crt_res.weight_diff_
        crt_diff_norm = crt_res.weight_diff_normalized_
        crt_err_smooth = np.zeros(len(crt_loc))
        crt_err_norm_smooth = np.zeros(len(crt_loc))
        crt_diff_smooth = np.zeros(len(crt_loc))
        crt_diff_norm_smooth = np.zeros(len(crt_loc))
        for i, k in enumerate(crt_loc):
            crt_err_smooth[i] = np.mean(crt_err[k : k + window_size])
            crt_err_norm_smooth[i] = np.mean(crt_err_norm[k : k + window_size])
            crt_diff_smooth[i] = np.mean(crt_diff[k : k + window_size])
            crt_diff_norm_smooth[i] = np.mean(crt_diff_norm[k : k + window_size])

        rolling_weight_errors.append((crt_loc, crt_err_smooth))
        rolling_weight_errors_normalized.append((crt_loc, crt_err_norm_smooth))
        rolling_weight_diff.append((crt_loc, crt_diff_smooth))
        rolling_weight_diff_normalized.append((crt_loc, crt_diff_norm_smooth))

    res.rolling_weight_errors = rolling_weight_errors
    res.rolling_weight_errors_normalized = rolling_weight_errors_normalized

    res.rolling_weight_diff = rolling_weight_diff
    res.rolling_weight_diff_normalized = rolling_weight_diff_normalized


def get_accuracy_metrics(
    results: SimpleNamespace,
    metrics: Sequence = (
        "mean_seg_acc",
        "median_seg_acc",
        "seg_good_frac",
        "bottom_seg_acc",
        "median_weight_error",
        "mean_weight_error",
        "bottom_weight_error",
        "mean_convergence_time",
        "median_convergence_time",
        "bottom_convergence_time",
    ),
    good_threshold: float = 0.85,
    bottom_quantile: float = 0.05,
    convergence_threshold: float = 0.90,
    convergence_resolution: float = 1000.0,
) -> dict:
    """ Calculate some accuracy metrics for run.

    Parameters
    ----------
    results
        Results namespace.
    metrics
        Which metrics to include in the output.
    good_threshold
        Minimum segmentation accuracy to consider a run "good".
    bottom_quantile
        What quantile to use for the "bottom_..." metrics.
    convergence_threshold
        Fraction of final segmentation accuracy to use to define convergence time.
    convergence_resolution
        The resolution of the convergence-time estimates. This is used to avoid
        misleadingly precise outputs.

    Returns a dictionary of metric values.
    """
    summary = {}

    # segmentation accuracy
    if "mean_seg_acc" in metrics:
        summary["mean_seg_acc"] = np.mean(results.trial_scores)
    if "median_seg_acc" in metrics:
        summary["median_seg_acc"] = np.median(results.trial_scores)
    if "seg_good_frac" in metrics:
        summary["seg_good_frac"] = np.mean(results.trial_scores >= good_threshold)
    if "bottom_seg_acc" in metrics:
        summary["bottom_seg_acc"] = np.quantile(results.trial_scores, bottom_quantile)

    # convergence times
    if any(
        _ in metrics
        for _ in [
            "mean_convergence_time",
            "median_convergence_time",
            "bottom_convergence_time",
        ]
    ):
        convergence_times = []
        for idx, crt_rolling in enumerate(results.rolling_scores):
            crt_threshold = convergence_threshold * results.trial_scores[idx]
            crt_mask = crt_rolling[1] >= crt_threshold

            # find the first index where the score goes above threshold
            if not np.any(crt_mask):
                crt_conv_idx = len(crt_mask) - 1
            else:
                crt_conv_idx = np.nonzero(crt_mask)[0][0]

            crt_time = crt_rolling[0][crt_conv_idx]
            crt_time = (
                np.round(crt_time / convergence_resolution) * convergence_resolution
            )
            convergence_times.append(crt_time)

        if "mean_convergence_time" in metrics:
            summary["mean_convergence_time"] = np.mean(convergence_times)
        if "median_convergence_time" in metrics:
            summary["median_convergence_time"] = np.median(convergence_times)
        if "bottom_convergence_time" in metrics:
            summary["bottom_convergence_time"] = np.quantile(
                convergence_times, bottom_quantile
            )

    # weight reconstruction
    if any(
        _ in metrics
        for _ in ["mean_weight_error", "median_weight_error", "bottom_weight_error"]
    ):
        # check whether we have weight reconstruction data
        if hasattr(results.history[0], "weight_errors_normalized_"):
            weight_errors = []
            for idx, crt_history in enumerate(results.history):
                crt_error = np.linalg.norm(crt_history.weight_errors_normalized_[-1])
                weight_errors.append(crt_error)

            if "mean_weight_error" in metrics:
                summary["mean_weight_error"] = np.mean(weight_errors)
            if "median_weight_error" in metrics:
                summary["median_weight_error"] = np.median(weight_errors)
            if "bottom_weight_error" in metrics:
                summary["bottom_weight_error"] = np.quantile(
                    weight_errors, 1 - bottom_quantile
                )
        else:
            if "mean_weight_error" in metrics:
                summary["mean_weight_error"] = np.nan
            if "median_weight_error" in metrics:
                summary["median_weight_error"] = np.nan
            if "bottom_weight_error" in metrics:
                summary["bottom_weight_error"] = np.nan

    return summary


def predict_plain_score(armas: Sequence, sigma_ratio: float = 1) -> float:
    delta = np.linalg.norm(armas[1].a - armas[0].a)
    return 0.5 + np.arctan(delta * sigma_ratio * np.sqrt(np.pi / 8)) / np.pi


def calculate_asymmetry_measures(
    results: SimpleNamespace, dataset: RandomArmaDataset, test_fraction: float = 0.1
):
    """ Calculate some metrics of asymmetry in segmentation.

    The function adds a field called `asymmetry` to the results, which is a list of
    namespaces, one for each run. Each namespace contains the following:
        * `confusion`:
            A confusion matrix such that `confusion[i, j]` is the fraction of time steps
            in which the ground-truth process `i` was (mis)identified as process `j`.
            Only samples in the chosen `test_fraction` are used.
        * `confusion_ordered`:
            A version of the confusion matrix in which the columns (i.e., model indices)
            are reordered so as to maximize the rate with which the inferred group
            assignments match the ground truth.
        * `ordering`:
            A sequence specifying the ordering used to generate `confusion_ordered`.
            This obeys `ordering[i]` is the ground-truth label that best matches
            inferred label `i`. Put differently,
                confusion_ordered = confusion[:, ordering]


    Parameters
    ----------
    results
        Results namespace.
    dataset
        Sequence of signals on which the simulations were run.
    test_fraction
        Fraction of samples in each (ground-truth) run to use.
    """
    asymmetry = []
    for history, signal in zip(tqdm(results.history), dataset):
        true_r = signal.usage_seq
        inferred_r = np.argmax(history.r, axis=1)

        # focus on what is there after the burnin
        n_samples = int(np.ceil(test_fraction * len(true_r)))
        true_r = true_r[-n_samples:]
        inferred_r = inferred_r[-n_samples:]

        # build the confusion matrix
        n = max(np.max(true_r) + 1, np.max(inferred_r) + 1)
        confusion = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                confusion[i, j] = np.mean((true_r == i) & (inferred_r == j))

        # calculate the optimal ordering
        _, ordering = unordered_accuracy_score(
            true_r, inferred_r, return_assignment=True
        )
        confusion_ordered = confusion[:, ordering]

        asymmetry.append(
            SimpleNamespace(
                confusion=confusion,
                confusion_ordered=confusion_ordered,
                ordering=ordering,
            )
        )

    results.asymmetry = asymmetry


def load_snippets(snip_type: str, snip_choice: str) -> list:
    snip_file = f"{snip_type}_dataset.pkl"
    with open(snip_file, "rb") as f:
        all_snips = pickle.load(f)

    lst = []
    for item in snip_choice:
        lst.append(all_snips[item])

    return lst
