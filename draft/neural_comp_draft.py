# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: bio_time_series
#     language: python
#     name: bio_time_series
# ---

# %% [markdown]
# # Plots for draft

# %%
# %matplotlib inline
# %config InlineBackend.print_figure_kwargs = {'bbox_inches': None}
import time
import copy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.optimize as sciopt

from types import SimpleNamespace
from typing import Tuple, Callable, Optional, Sequence, Union, List
from tqdm.notebook import tqdm
from matplotlib.legend_handler import HandlerTuple

from sklearn import metrics

from bioslds import sources
from bioslds.regressors import (
    BioWTARegressor,
    CrosscorrelationRegressor,
    CepstralRegressor,
)
from bioslds.plotting import FigureManager, show_latent, colorbar, make_gradient_cmap
from bioslds.cluster_quality import calculate_sliding_score, unordered_accuracy_score
from bioslds.batch import hyper_score_ar
from bioslds.dataset import RandomArmaDataset
from bioslds.arma import Arma, make_random_arma
from bioslds.arma_hsmm import sample_switching_models

fig_path = os.path.join("..", "figs", "draft")
paper_style = [
    "seaborn-paper",
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6},
]


# %% [markdown]
# # Useful definitions

# %%
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
) -> Tuple[plt.Figure, List[plt.Axes]]:
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
        sns.rugplot(x=results.convergence_times, color=trace_color, ax=axs[1, 0], **rug_kws)
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


# %%
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


# %%
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


# %%
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


# %%
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


# %%
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


# %%
def get_accuracy_metrics(
    results: SimpleNamespace,
    metrics: Sequence = [
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
    ],
    good_threshold: float = 0.85,
    bottom_quantile: float = 0.05,
    convergence_threshold: float = 0.90,
    convergence_resolution: float = 1000.0,
) -> dict:
    """ Calculate some accuracy metrics for  run.
    
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
        The resolution of the convergence-time estimates. This is used to avoid misleadingly
        precise outputs.
    
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


# %%
def predict_plain_score(armas: Sequence, sigma_ratio: float = 1) -> float:
    delta = np.linalg.norm(armas[1].a - armas[0].a)
    return 0.5 + np.arctan(delta * sigma_ratio * np.sqrt(np.pi / 8)) / np.pi


# %% [markdown]
# # Make plots explaining the problem setup

# %%
problem_setup = SimpleNamespace(
    n_samples=500,
    orders=[(3, 0), (3, 0)],
    dwell_times=100,
    min_dwell=50,
    max_pole_radius=0.95,
    normalize=True,
    fix_scale=None,
    seed=5,
)
problem_setup.dataset = RandomArmaDataset(
    1,
    problem_setup.n_samples,
    problem_setup.orders,
    dwell_times=problem_setup.dwell_times,
    min_dwell=problem_setup.min_dwell,
    fix_scale=problem_setup.fix_scale,
    normalize=problem_setup.normalize,
    rng=problem_setup.seed,
    arma_kws={"max_pole_radius": problem_setup.max_pole_radius},
)
problem_setup.sig = problem_setup.dataset[0].y
problem_setup.usage_seq = problem_setup.dataset[0].usage_seq

# %%
with plt.style.context(paper_style):
    with FigureManager(
        2, 1, figsize=(5.76, 2.5), sharex=True, gridspec_kw={"height_ratios": (1, 2)}
    ) as (
        fig,
        (ax0, ax1),
    ):
        ax0.plot(problem_setup.usage_seq == 0, "C0", label="$z_1$")
        ax0.plot(problem_setup.usage_seq == 1, "C1", label="$z_2$")

        ax0.fill_between(
            np.arange(problem_setup.n_samples),
            problem_setup.usage_seq == 0,
            color="C0",
            alpha=0.1,
        )
        ax0.fill_between(
            np.arange(problem_setup.n_samples),
            problem_setup.usage_seq == 1,
            color="C1",
            alpha=0.1,
        )
        ax0.legend(frameon=False)

        ax1.plot(problem_setup.sig)
        ax1.set_xlabel("time step")

        ax1.set_xlim(0, problem_setup.n_samples)
        ax1.set_yticks([])

        show_latent(problem_setup.usage_seq, ax=ax1)

    sns.despine(left=True, ax=ax1)

fig.savefig(os.path.join(fig_path, "example_switching_signal_and_z.pdf"))

# %% [markdown]
# ## Problem setup, v2

# %%
problem_setup_v2 = SimpleNamespace(
    n_samples=600, orders=[(3, 0), (2, 0), (1, 0)], max_pole_radius=0.95, seed=5,
)
rng = np.random.default_rng(problem_setup_v2.seed)
problem_setup_v2.armas = [
    make_random_arma(*_, rng=rng, max_pole_radius=problem_setup_v2.max_pole_radius)
    for _ in problem_setup_v2.orders
]

problem_setup_v2.usage_seq = np.zeros(problem_setup_v2.n_samples, dtype=int)
problem_setup_v2.usage_seq[
    problem_setup_v2.n_samples // 7 : 4 * problem_setup_v2.n_samples // 7
] = 1
problem_setup_v2.usage_seq[
    4 * problem_setup_v2.n_samples // 7 : 6 * problem_setup_v2.n_samples // 7
] = 2
problem_setup_v2.usage_seq[6 * problem_setup_v2.n_samples // 7 :] = 1
problem_setup_v2.sig, problem_setup_v2.x = sample_switching_models(
    problem_setup_v2.armas,
    usage_seq=problem_setup_v2.usage_seq,
    X=sources.GaussianNoise(rng),
    return_input=True,
)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        2, 1, figsize=(5.76, 2.5), sharex=True, gridspec_kw={"height_ratios": (3, 1.5)}
    ) as (fig, (ax0, ax1)):
        ax0.axhline(0, ls=":", c="gray", zorder=-1)
        ax0.plot(problem_setup_v2.sig, 'k', lw=0.75)
        yl = np.max(np.abs(problem_setup_v2.sig))
        ax0.set_ylim(-yl, yl)

        show_latent(problem_setup_v2.usage_seq, ax=ax0)

        # ax0.set_xlabel("time step")
        ax0.set_ylabel("signal")

        ax0.set_xlim(0, problem_setup_v2.n_samples)
        ax0.set_yticks([])

        # start with ground truth, then add some noise, and finally smooth and normalize
        problem_setup_v2.mock_z = np.asarray(
            [problem_setup_v2.usage_seq == [0, 1, 2][_] for _ in range(3)]
        )

        rng = np.random.default_rng(problem_setup_v2.seed)
        problem_setup_v2.mock_z = np.clip(
            problem_setup_v2.mock_z
            + 0.20 * rng.normal(size=problem_setup_v2.mock_z.shape),
            0,
            None,
        )

        # smooth
        crt_kernel = np.exp(-0.5 * np.linspace(-3, 3, 36) ** 2)
        problem_setup_v2.mock_z = np.array(
            [
                np.convolve(_, crt_kernel)[: problem_setup_v2.n_samples]
                for _ in problem_setup_v2.mock_z
            ]
        )

        # normalize
        problem_setup_v2.mock_z = problem_setup_v2.mock_z / np.sum(
            problem_setup_v2.mock_z, axis=0
        )

        for i in range(3):
            ax1.plot(problem_setup_v2.mock_z[i], f"C{i}", label=f"$z_{i}$")

        for i in range(3):
            ax1.fill_between(
                np.arange(problem_setup_v2.n_samples),
                problem_setup_v2.mock_z[i],
                color=f"C{i}",
                alpha=0.1,
            )
        ax1.legend(frameon=False, loc=(0.25, 0.27), fontsize=6, ncol=3)
        ax1.set_ylim(0, 1)

        ax1.set_xlabel("time step")
        ax1.set_ylabel("inference")

        show_latent(problem_setup_v2.usage_seq, show_bars=False, ax=ax1)

#     sns.despine(left=True, ax=ax)

fig.savefig(os.path.join(fig_path, "example_switching_signal_and_z_v2.pdf"))

# %% [markdown]
# # Run BioWTA, autocorrelation, and cepstral oracle algorithms on signals based on pairs of AR(3) processes

# %% [markdown]
# ## Define the problem and the parameters for the learning algorithms

# %% [markdown]
# Using best parameters obtained from hyperoptimization runs.

# %%
two_ar3 = SimpleNamespace(
    n_signals=100,
    n_samples=200_000,
    orders=[(3, 0), (3, 0)],
    dwell_times=100,
    min_dwell=50,
    max_pole_radius=0.95,
    normalize=True,
    fix_scale=None,
    seed=153,
    n_models=2,
    n_features=3,
    rate_nsm=0.005028,
    streak_nsm=9.527731,
    rate_cepstral=0.071844,
    order_cepstral=2,
    metric=unordered_accuracy_score,
    good_score=0.85,
    threshold_steps=10_000,
)
two_ar3.dataset = RandomArmaDataset(
    two_ar3.n_signals,
    two_ar3.n_samples,
    two_ar3.orders,
    dwell_times=two_ar3.dwell_times,
    min_dwell=two_ar3.min_dwell,
    fix_scale=two_ar3.fix_scale,
    normalize=two_ar3.normalize,
    rng=two_ar3.seed,
    arma_kws={"max_pole_radius": two_ar3.max_pole_radius},
)

# %% [markdown]
# ## Run BioWTA with all combinations of enhancements

# %%
two_ar3.biowta_configurations = {
    (1, 1, 0): {
        "rate": 0.005460,
        "trans_mat": 1 - 1 / 6.792138,
        "temperature": 0.961581,
        "error_timescale": 1.000000,
    },
    (0, 0, 1): {
        "rate": 0.004187,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.000000,
        "error_timescale": 3.937542,
    },
    (1, 1, 1): {
        "rate": 0.001664,
        "trans_mat": 1 - 1 / 4.635351,
        "temperature": 0.629294,
        "error_timescale": 1.217550,
    },
    (0, 1, 1): {
        "rate": 0.003013,
        "trans_mat": 1 - 1 / 2.179181,
        "temperature": 0.000000,
        "error_timescale": 2.470230,
    },
    (1, 0, 1): {
        "rate": 0.005444,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.062365,
        "error_timescale": 3.842287,
    },
    (0, 1, 0): {
        "rate": 0.001906,
        "trans_mat": 1 - 1 / 2.852480,
        "temperature": 0.000000,
        "error_timescale": 1.000000,
    },
    (0, 0, 0): {
        "rate": 0.005638,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.000000,
        "error_timescale": 1.000000,
    },
    (1, 0, 0): {
        "rate": 0.000394,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.008065,
        "error_timescale": 1.000000,
    },
}
two_ar3.biowta_configurations_human = {
    (0, 0, 0): "plain",
    (0, 0, 1): "avg_error",
    (0, 1, 0): "persistent",
    (1, 0, 0): "soft",
    (0, 1, 1): "persistent+avg_error",
    (1, 1, 0): "soft+persistent",
    (1, 0, 1): "soft+avg_error",
    (1, 1, 1): "full",
}

# %%
two_ar3.result_biowta_mods = {}
for key in tqdm(two_ar3.biowta_configurations, desc="biowta cfg"):
    two_ar3.result_biowta_mods[key] = hyper_score_ar(
        BioWTARegressor,
        two_ar3.dataset,
        two_ar3.metric,
        n_models=two_ar3.n_models,
        n_features=two_ar3.n_features,
        progress=tqdm,
        monitor=["r", "weights_", "prediction_"],
        **two_ar3.biowta_configurations[key],
    )

    crt_scores = two_ar3.result_biowta_mods[key][1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > two_ar3.good_score)
    print(
        f"{''.join(str(_) for _ in key)}: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * two_ar3.good_score)}%={crt_good:.4f}"
    )

# %%
for key in tqdm(
    two_ar3.biowta_configurations, desc="biowta cfg, reconstruction progress"
):
    calculate_ar_identification_progress(
        two_ar3.result_biowta_mods[key][1].history, two_ar3.dataset
    )

# %% [markdown]
# Find some "good" indices in the dataset: one that obtains an accuracy score close to a chosen threshold for "good-enough" (which we set to 85%); and one that has a similar score but also has small reconstruction error for the weights.

# %%
two_ar3.result_biowta_chosen = two_ar3.result_biowta_mods[1, 1, 0]
crt_mask = (
    two_ar3.result_biowta_chosen[1].trial_scores > 0.98 * two_ar3.good_score
) & (two_ar3.result_biowta_chosen[1].trial_scores < 1.02 * two_ar3.good_score)
crt_idxs = crt_mask.nonzero()[0]

crt_errors_norm = np.asarray(
    [
        np.mean(_.weight_errors_normalized_[-1])
        for _ in two_ar3.result_biowta_chosen[1].history
    ]
)

two_ar3.good_biowta_idx = crt_idxs[np.argmax(crt_errors_norm[crt_mask])]
two_ar3.good_biowta_ident_idx = crt_idxs[np.argmin(crt_errors_norm[crt_mask])]
two_ar3.good_idxs = [two_ar3.good_biowta_ident_idx, two_ar3.good_biowta_idx]

# %%
two_ar3.result_biowta_chosen[1].trial_scores[two_ar3.good_idxs]

# %%
crt_errors_norm[two_ar3.good_idxs]

# %%
for key in two_ar3.biowta_configurations:
    make_multi_trajectory_plot(
        two_ar3.result_biowta_mods[key][1],
        two_ar3.dataset,
        n_traces=25,
        highlight_idx=two_ar3.good_idxs,
        sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
        trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
        rug_kws={"alpha": 0.3},
    )

# %% [markdown]
# ## Run learning and inference for autocorrelation and cepstral methods

# %%
t0 = time.time()
two_ar3.result_xcorr = hyper_score_ar(
    CrosscorrelationRegressor,
    two_ar3.dataset,
    two_ar3.metric,
    n_models=two_ar3.n_models,
    n_features=two_ar3.n_features,
    nsm_rate=two_ar3.rate_nsm,
    xcorr_rate=1 / two_ar3.streak_nsm,
    progress=tqdm,
    monitor=["r", "nsm.weights_", "xcorr.coef_"],
)
t1 = time.time()
print(
    f"Median accuracy score xcorr: {two_ar3.result_xcorr[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %%
t0 = time.time()
two_ar3.result_cepstral = hyper_score_ar(
    CepstralRegressor,
    two_ar3.dataset,
    two_ar3.metric,
    cepstral_order=two_ar3.order_cepstral,
    cepstral_kws={"rate": two_ar3.rate_cepstral},
    initial_weights="oracle_ar",
    progress=tqdm,
    monitor=["r"],
)
t1 = time.time()
print(
    f"Median accuracy score cepstral: {two_ar3.result_cepstral[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %% [markdown]
# ## Run BioWTA with weights fixed at ground-truth values

# %%
t0 = time.time()
two_ar3.oracle_biowta = hyper_score_ar(
    BioWTARegressor,
    two_ar3.dataset,
    two_ar3.metric,
    n_models=two_ar3.n_models,
    n_features=two_ar3.n_features,
    rate=0,
    trans_mat=two_ar3.biowta_configurations[1, 1, 0]["trans_mat"],
    temperature=two_ar3.biowta_configurations[1, 1, 0]["temperature"],
    error_timescale=two_ar3.biowta_configurations[1, 1, 0]["error_timescale"],
    initial_weights="oracle_ar",
    progress=tqdm,
    monitor=["r", "prediction_"],
)
t1 = time.time()
print(
    f"Median accuracy score oracle BioWTA: {two_ar3.oracle_biowta[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %% [markdown]
# ## Make plots

# %%
fig, axs = make_accuracy_plot(
    two_ar3.result_biowta_chosen[1],
    two_ar3.oracle_biowta[1],
    two_ar3.dataset,
    two_ar3.good_idxs,
)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("enh. BioWTA")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_biowta.png"), dpi=600
)

# %%
crt_frac_good = np.mean(
    two_ar3.result_biowta_chosen[1].trial_scores > two_ar3.good_score
)
print(
    f"Percentage of runs with BioWTA accuracies over {int(two_ar3.good_score * 100)}%: "
    f"{int(crt_frac_good * 100)}%."
)

crt_frac_fast = np.mean(
    np.asarray(two_ar3.result_biowta_chosen[1].convergence_times)
    <= two_ar3.threshold_steps
)
print(
    f"Percentage of runs with BioWTA convergence times under {two_ar3.threshold_steps}: "
    f"{int(crt_frac_fast * 100)}%."
)

# %%
fig, axs = make_accuracy_plot(
    two_ar3.result_xcorr[1],
    two_ar3.oracle_biowta[1],
    two_ar3.dataset,
    two_ar3.good_idxs,
)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("autocorrelation")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_xcorr.png"), dpi=600
)

# %%
print(
    f"Percentage of runs with xcorr accuracies over {int(two_ar3.good_score * 100)}%: "
    f"{int(np.mean(two_ar3.result_xcorr[1].trial_scores > two_ar3.good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with xcorr convergence times under {two_ar3.threshold_steps}: "
    f"{int(np.mean(np.asarray(two_ar3.result_xcorr[1].convergence_times) <= two_ar3.threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(two_ar3.result_xcorr[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %%
fig, axs = make_accuracy_plot(
    two_ar3.result_cepstral[1],
    two_ar3.oracle_biowta[1],
    two_ar3.dataset,
    two_ar3.good_idxs,
)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("cepstral oracle")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_cepstral.png"), dpi=600
)

# %%
print(
    f"Percentage of runs with cepstral accuracies over {int(two_ar3.good_score * 100)}%: "
    f"{int(np.mean(two_ar3.result_cepstral[1].trial_scores > two_ar3.good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with cepstral convergence times under {two_ar3.threshold_steps}: "
    f"{int(np.mean(np.asarray(two_ar3.result_cepstral[1].convergence_times) <= two_ar3.threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with cepstral convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(two_ar3.result_cepstral[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %% [markdown]
# # Explain variability in BioWTA accuracy scores, show effect of algorithm improvements

# %%
predicted_plain_scores = [
    predict_plain_score(crt_sig.armas, sigma_ratio=1.0 / crt_sig.scale)
    for crt_sig in tqdm(two_ar3.dataset)
]

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1,
        2,
        gridspec_kw={"width_ratios": (12, 2)},
        despine_kws={"offset": 5},
        figsize=(2.8, 1.5),
        constrained_layout=True,
    ) as (
        fig,
        axs,
    ):
        crt_sigma = 0.5
        crt_pred1 = -crt_sigma
        crt_pred2 = crt_sigma

        crt_thresh = 0.5 * (crt_pred1 + crt_pred2)

        crt_samples = [-0.3, 1.0, -0.7, 0.4, -1.3, -0.6, 0.3, -0.2, -0.5]
        crt_n = len(crt_samples)
        crt_usage = np.zeros(crt_n + 1, dtype=int)
        axs[0].plot(crt_samples, ".-", c="gray")
        # axs[0].axhline(0, ls=":", c="gray")

        crt_box = [[crt_n - 0.4, crt_n + 0.4], [-1.4, 1.4]]
        axs[0].plot(
            crt_box[0] + crt_box[0][::-1] + [crt_box[0][0]],
            [crt_box[1][0]] + crt_box[1] + crt_box[1][::-1],
            "k-",
        )

        crt_p_range = (-1.5, 1.5)
        axs[0].set_ylim(*crt_p_range)
        axs[0].set_xlabel("time step")
        axs[0].set_ylabel("signal $y(t)$")

        axs[0].set_xticks([0, len(crt_samples)])
        axs[0].set_xticklabels([0, "$\\tau$"])

        show_latent(crt_usage, ax=axs[0])

        axs[0].annotate(
            "ground truth: model 1",
            (0.5, axs[0].get_ylim()[1] - 0.03),
            color="w",
            verticalalignment="top",
            fontsize=6,
            fontweight="bold",
        )

        crt_ps = np.linspace(*crt_p_range, 100)
        crt_dist = (
            1
            / np.sqrt(2 * np.pi * crt_sigma ** 2)
            * np.exp(-0.5 * ((crt_ps - crt_pred1) / crt_sigma) ** 2)
        )
        for crt_y, crt_p in zip(crt_ps, crt_dist):
            if crt_y < crt_box[1][0] or crt_y >= crt_box[1][1]:
                continue
            axs[0].plot(
                [crt_n - 1, crt_box[0][0]],
                [crt_samples[-1], crt_y],
                c="gray",
                alpha=0.5 * crt_p,
            )
            axs[0].plot(
                [crt_box[0][0] + 0.01, crt_box[0][1] - 0.01],
                [crt_y, crt_y],
                c="gray",
                alpha=0.5 * crt_p,
            )

        crt_col1 = "C0"
        crt_col2 = "C1"
        crt_col_err1 = "C1"
        crt_col_err2 = "C4"

        crt_x0 = 1.00
        axs[1].annotate(
            "model 1",
            xy=(crt_x0, crt_pred1),
            verticalalignment="center",
            # fontweight="bold",
            fontsize=7,
            color=crt_col1,
        )
        axs[1].annotate(
            "model 2",
            xy=(crt_x0, crt_pred2),
            verticalalignment="center",
            # fontweight="bold",
            fontsize=7,
            color=crt_col2,
        )
        axs[1].annotate(
            "decision\nboundary",
            xy=(crt_x0, crt_thresh),
            verticalalignment="center",
            # fontweight="bold",
            fontsize=7,
            color="gray",
            linespacing=0.8,
        )

        crt_cut_idx = np.argmin(np.abs(crt_ps - crt_thresh))
        axs[1].plot(
            crt_dist[: crt_cut_idx + 1],
            crt_ps[: crt_cut_idx + 1],
            c=crt_col1,
            alpha=0.8,
        )
        axs[1].plot(
            crt_dist[crt_cut_idx:], crt_ps[crt_cut_idx:], c=crt_col_err1, alpha=0.8
        )
        axs[1].plot(-crt_dist, crt_ps, c="gray", alpha=0.8)

        axs[1].fill_betweenx(
            crt_ps, -crt_dist, color="gray", alpha=0.3,
        )
        axs[1].fill_betweenx(
            crt_ps[: crt_cut_idx + 1],
            crt_dist[: crt_cut_idx + 1],
            color=crt_col1,
            alpha=0.3,
        )
        axs[1].fill_betweenx(
            crt_ps[crt_cut_idx:], crt_dist[crt_cut_idx:], color=crt_col_err1, alpha=0.3,
        )

        axs[1].axhline(crt_pred1, c=crt_col1, ls=":")
        axs[1].axhline(crt_pred2, c=crt_col2, ls=":")
        axs[1].axhline(crt_thresh, c="gray", ls="--")

        axs[1].set_xlim(-1.0, crt_x0)
        axs[1].set_ylim(*crt_p_range)
        axs[1].set_xlabel("pdf $y(t=\\tau)$")
        axs[1].set_yticks([])
        axs[1].set_xticks([0])
        axs[1].set_xticklabels([" "])

sns.despine(left=True, ax=axs[1])

fig.savefig(
    os.path.join(fig_path, "explanation_for_biowta_segmentation_errors.pdf"),
    transparent=True,
)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1,
        2,
        despine_kws={"offset": 5},
        figsize=(3, 1.5),
        constrained_layout=True,
    ) as (
        fig,
        axs,
    ):
        axs[0].plot([0.5, 1], [0.5, 1], "--", c="gray", zorder=-15)
        axs[0].scatter(
            predicted_plain_scores,
            two_ar3.result_biowta_mods[0, 0, 0][1].trial_scores,
            s=6,
            c="C2",
            alpha=0.5,
        )

        axs[0].set_aspect(1)
        axs[0].set_xlabel("expectation")
        axs[0].set_ylabel("plain BioWTA")

        axs[0].set_xlim([0.5, 1])
        axs[0].set_ylim([0.5, 1])

        axs[1].plot([0.5, 1], [0.5, 1], "--", c="gray", zorder=-15)
        axs[1].scatter(
            predicted_plain_scores,
            two_ar3.result_biowta_chosen[1].trial_scores,
            s=6,
            c="C3",
            alpha=0.5,
        )

        axs[1].set_aspect(1)
        axs[1].set_xlabel("expectation")
        axs[1].set_ylabel("enh. BioWTA")

        axs[1].set_xlim([0.5, 1])
        axs[1].set_ylim([0.5, 1])
        
fig.savefig(
    os.path.join(fig_path, "plain_vs_enh_biowta.pdf"),
    transparent=True,
)

# %%
with plt.style.context(paper_style):
    with FigureManager(despine_kws={"offset": 5}, figsize=(5.76, 1.5)) as (
        fig,
        ax,
    ):
        crt_x_values = []
        crt_y_values = []
        mod_sel = {
            "no enhancements\n(plain BioWTA)": (0, 0, 0),
            "persistent": (0, 1, 0),
            "soft\npersistent\n(enh. BioWTA)": (1, 1, 0),
            "soft\npersistent\naveraging": (1, 1, 1),
            "averaging\nonly": (0, 0, 1),
        }

        for i, (crt_name, crt_mod) in enumerate(mod_sel.items()):
            crt_scores = two_ar3.result_biowta_mods[crt_mod][1].trial_scores
            crt_y_values.extend(crt_scores)
            crt_x_values.extend([i] * len(crt_scores))

        sns.violinplot(
            x=crt_x_values,
            y=crt_y_values,
            palette=["C2", "gray", "C3", "gray", "gray", "gray"],
            order=[0, 1, 2, 3, 4],
            cut=0,
            ax=ax,
        )

        ax.set_ylabel("segmentation\naccuracy")

        ax.set_xticks([0, 1, 2, 3, 4])
        ax.set_xticklabels(mod_sel.keys())

fig.savefig(
    os.path.join(fig_path, "effect_of_biowta_improvements.pdf"), transparent=True
)

# %% [markdown]
# # Compare our algorithms with cepstral method

# %%
with plt.style.context(paper_style):
    with FigureManager(despine_kws={"offset": 5}, figsize=(5.76, 3)) as (fig, ax):
        make_accuracy_comparison_diagram(
            ax,
            [
                two_ar3.result_xcorr,
                two_ar3.result_biowta_mods[0, 0, 0],
                two_ar3.result_biowta_chosen,
                two_ar3.result_cepstral,
            ],
            ["autocorrelation", "plain BioWTA", "enhanced BioWTA", "cepstral"],
            color_cycle=["C0", "C2", "C3", "C1"],
        )
    ax.tick_params(axis='x', labelsize=8)

fig.savefig(os.path.join(fig_path, "algo_comparisons.pdf"))

# %% [markdown]
# # Make table of algorithm performance

# %%
for_table = {
    "autocorr.": two_ar3.result_xcorr,
    "plain BioWTA": two_ar3.result_biowta_mods[0, 0, 0],
    "enh. BioWTA": two_ar3.result_biowta_chosen,
    "cepstral": two_ar3.result_cepstral,
}
# measures = [
#     # "mean_seg_acc",
#     "median_seg_acc",
#     "seg_good_frac",
#     "bottom_seg_acc",
#     "median_weight_error",
#     # "mean_weight_error",
#     # "bottom_weight_error",
#     # "mean_convergence_time",
#     "median_convergence_time",
#     # "bottom_convergence_time",
# ]
measures = [
    "mean_seg_acc",
    # "median_seg_acc",
    "seg_good_frac",
    "bottom_seg_acc",
    # "median_weight_error",
    "mean_weight_error",
    # "bottom_weight_error",
    "mean_convergence_time",
    # "median_convergence_time",
    # "bottom_convergence_time",
]

accuracy_results = {_: [] for _ in measures}
accuracy_index = []
for crt_name, crt_results in for_table.items():
    crt_summary = get_accuracy_metrics(crt_results[1], metrics=measures)
    for key in crt_summary:
        accuracy_results[key].append(crt_summary[key])
    accuracy_index.append(crt_name)

accuracy_table0 = pd.DataFrame(accuracy_results, index=accuracy_index)
accuracy_table0.rename(
    columns={
        "median_seg_acc": "Median seg. score",
        "mean_seg_acc": "Mean segmentation score",
        "seg_good_frac": "Fraction well-segmented",
        "bottom_seg_acc": "Seg. score of bottom 5%",
        "median_weight_error": "Median weight error",
        "mean_weight_error": "Mean weight error",
        "mean_convergence_time": "Mean convergence time",
        "median_convergence_time": "Median convergence time",
    },
    inplace=True,
)
accuracy_table = accuracy_table0.transpose()

# %%
accuracy_table

# %%
print(accuracy_table.to_latex(na_rep="N/A", float_format="%.2f"))

# %% [markdown]
# # Plot progression of system identification

# %%
calculate_smooth_weight_errors(two_ar3.result_biowta_chosen[1])

# %%
# for how many runs does the best ground-truth-to-inferred assignment depend on
# whether it's judged by segmentation vs. weight accuracy?
np.sum(
    [
        np.any(_.best_assignment_segmentation != _.best_assignment_weight)
        for _ in two_ar3.result_biowta_chosen[1].history
    ]
)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1,
        3,
        despine_kws={"offset": 5},
        gridspec_kw={"width_ratios": (5, 1, 2)},
        figsize=(5.76, 1.7),
        constrained_layout=True,
    ) as (fig, axs):
        ax_weight_progress, ax_weight_histo, ax_weight_vs_seg = axs

        # draw weight error progression
        max_weight_error = 1.4
        ax = ax_weight_progress
        for crt_idx, crt_roll in enumerate(
            two_ar3.result_biowta_chosen[1].rolling_weight_errors_normalized
        ):
            crt_kws = {"c": "gray", "lw": 0.5, "alpha": 0.3}
            if crt_idx in two_ar3.good_idxs:
                crt_kws["lw"] = 2.0
                crt_kws["alpha"] = 1.0
                crt_kws["c"] = f"C{two_ar3.good_idxs.index(crt_idx)}"
            ax.plot(*crt_roll, **crt_kws)
        ax.set_ylim(0, max_weight_error)
        ax.set_xlabel("time step")
        ax.set_ylabel("normalized\nreconstruction error")
        ax.set_xlim(0, two_ar3.n_samples)

        # draw the late error distribution
        ax = ax_weight_histo
        late_errors = [
            _[1][-1]
            for _ in two_ar3.result_biowta_chosen[1].rolling_weight_errors_normalized
        ]
        sns.kdeplot(y=late_errors, shade=True, color="gray", ax=ax)
        sns.rugplot(y=late_errors, height=0.1, alpha=0.5, color="gray", ax=ax)
        for i, special_idx in enumerate(two_ar3.good_idxs):
            sns.rugplot(
                y=[late_errors[special_idx]],
                height=0.1,
                alpha=0.5,
                color=f"C{i}",
                lw=2,
                ax=ax,
            )
        ax.set_ylim(0, max_weight_error)
        ax.set_yticks([])
        ax.set_xlabel("pdf")

        # show relation b/w weight reconstruction and segmentation score
        ax = ax_weight_vs_seg
        ar_diffs = [
            np.linalg.norm(np.std([__.a for __ in _], axis=0))
            for _ in two_ar3.dataset.armas
        ]
        h = ax.scatter(
            two_ar3.result_biowta_chosen[1].trial_scores,
            late_errors,
            s=6,
            c="gray",
            alpha=0.4,
        )
        for i, special_idx in enumerate(two_ar3.good_idxs):
            ax.scatter(
                [two_ar3.result_biowta_chosen[1].trial_scores[special_idx]],
                [late_errors[special_idx]],
                s=12,
                c=f"C{i}",
                alpha=1.0,
            )
        # colorbar(h)
        ax.set_xlabel("segmentation accuracy")
        ax.set_ylabel("normalized\nreconstruction error")
        ax.set_ylim(0.0, max_weight_error)
        ax.set_xlim(0.5, 1.0)

        ax.yaxis.set_label_position("right")

    sns.despine(ax=ax_weight_vs_seg, left=True, right=False, offset=5)
    sns.despine(ax=ax_weight_histo, left=True)

fig.savefig(
    os.path.join(fig_path, "biowta_weight_reconstruction.png"), dpi=600,
)

# %%
with plt.style.context(paper_style):
    fig = plt.figure(figsize=(5.76, 2.2))
    axs = np.empty(2, dtype=object)
    axs[0] = fig.add_axes([0.0, 0.15, 0.4, 0.7])
    axs[1] = fig.add_axes([0.6, 0.15, 0.4, 0.7])
    crt_idxs = two_ar3.good_idxs
    for i, ax in enumerate(axs):
        ax.axhline(0, ls=":", c="gray", lw=0.5, xmax=1.05, clip_on=False)
        ax.axvline(0, ls=":", c="gray", lw=0.5, ymax=1.05, clip_on=False)

        crt_idx = crt_idxs[i]
        crt_true = [_.calculate_poles() for _ in two_ar3.dataset.armas[crt_idx]]
        crt_inferred = [
            Arma(_, []).calculate_poles()
            for _ in two_ar3.result_biowta_chosen[1]
            .history[crt_idx]
            .weights_shuffled_[-1]
        ]

        ax.set_title(["good reconstruction", "bad reconstruction"][i], color=f"C{i}")

        h_true = []
        h_inferred = []
        for k in range(len(crt_true)):
            (crt_h_true,) = ax.plot(
                np.real(crt_true[k]),
                np.imag(crt_true[k]),
                c=f"C{2 + k}",
                marker="^v"[k],
                markersize=7,
                linestyle="none",
                alpha=0.3,
            )
            (crt_h_inferred,) = ax.plot(
                np.real(crt_inferred[k]),
                np.imag(crt_inferred[k]),
                c=f"C{2 + k}",
                marker="^v"[k],
                markersize=3,
                linestyle="none",
            )
            h_true.append(crt_h_true)
            h_inferred.append(crt_h_inferred)

        crt_theta = np.linspace(0, 2 * np.pi, 60)
        crt_x = np.cos(crt_theta)
        crt_y = np.sin(crt_theta)
        ax.plot(crt_x, crt_y, "k", lw=2, c=f"C{i}", clip_on=False)

        ax.set_aspect(1)
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)

        ax.set_xticks([])
        ax.set_yticks([])

        ax.annotate(
            "Re $z$", (1, 0), xytext=(2, 1), textcoords="offset points", color="gray",
        )
        ax.annotate(
            "Im $z$", (0, 1), xytext=(1, 2), textcoords="offset points", color="gray",
        )

        ax.annotate(
            "unit\ncircle",
            (np.sqrt(2) / 2, -np.sqrt(2) / 2),
            xytext=(3, 0),
            textcoords="offset points",
            verticalalignment="top",
            color=f"C{i}",
        )

    # draw the legend
    legend_ax = fig.add_axes([0.38, 0.41, 0.24, 0.21])
    legend_ax.set_xticks([])
    legend_ax.set_yticks([])

    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)

    label_xs = [0.52, 0.82]
    label_ys = [0.48, 0.17]
    top_y = max(label_ys) + 0.32

    for i, crt_y in enumerate(label_ys):
        legend_ax.annotate(
            f"model {i + 1}",
            (0.35, crt_y),
            verticalalignment="center",
            horizontalalignment="right",
        )

    for i, crt_x in enumerate(label_xs):
        crt_name = ["true", "inferred"][i]
        legend_ax.annotate(
            crt_name,
            (crt_x, top_y),
            verticalalignment="center",
            horizontalalignment="center",
        )

    for k, crt_y in enumerate(label_ys):
        legend_ax.plot(
            [label_xs[0]],
            [crt_y],
            c=f"C{2 + k}",
            marker="^v"[k],
            markersize=7,
            linestyle="none",
            alpha=0.3,
        )
        legend_ax.plot(
            [label_xs[1]],
            [crt_y],
            c=f"C{2 + k}",
            marker="^v"[k],
            markersize=3,
            linestyle="none",
        )
        
    legend_ax.plot([0, 1], [1, 1], c="gray", lw=1, clip_on=False)
    legend_ax.plot([label_xs[0] - 0.5, label_xs[1] + 0.16], 2 * [top_y - 0.14], c="gray", lw=0.5)
    legend_ax.plot([0, 1], [0, 0], c="gray", lw=1, clip_on=False)

for ax in axs:
    sns.despine(left=True, bottom=True, ax=ax)
sns.despine(left=True, bottom=True, ax=legend_ax)

fig.suptitle("Poles of\nAR processes", y=1.0)
fig.savefig(os.path.join(fig_path, "biowta_good_vs_bad_reconstruction.pdf"))

# %% [markdown]
# # Comparing all enhancements

# %%
biowta_mods_seg_scores = pd.DataFrame(
    {
        two_ar3.biowta_configurations_human[key]: crt_res[1].trial_scores
        for key, crt_res in two_ar3.result_biowta_mods.items()
    }
)
biowta_mods_weight_errors = pd.DataFrame(
    {
        two_ar3.biowta_configurations_human[key]: np.asarray(
            [
                np.linalg.norm(_.weight_errors_normalized_[-1])
                for _ in crt_res[1].history
            ]
        )
        for key, crt_res in two_ar3.result_biowta_mods.items()
    }
)

# %%
with plt.style.context(paper_style):
    with FigureManager(8, 8, despine_kws={"offset": 0}, figsize=(5.76, 5.76)) as (
        fig,
        axs,
    ):
        for i in range(8):
            crt_col_i = biowta_mods_seg_scores.columns[i]
            crt_series_i = biowta_mods_seg_scores[crt_col_i]
            for j in range(8):
                crt_col_j = biowta_mods_seg_scores.columns[j]
                crt_series_j = biowta_mods_seg_scores[crt_col_j]

                ax = axs[i, j]

                if i != j:
                    ax.plot(
                        [0.45, 1.05], [0.45, 1.05], "--", c="gray", alpha=0.7, lw=0.5
                    )
                    ax.scatter(crt_series_j, crt_series_i, 1, alpha=0.7)

                    ax.set_yticks([0.5, 1.0])
                    ax.set_ylim(0.45, 1.05)
                    ax.set_aspect(1.0)
                else:
                    ax.hist(crt_series_i, 15)
                    ax.set_yticks([])
                    ax.annotate(
                        "{:.0f}%".format(100 * np.median(crt_series_i)),
                        (0.15, 0.80),
                        xycoords="axes fraction",
                    )

                ax.set_xlim(0.45, 1.05)
                ax.set_xticks([0.5, 1.0])
                ax.tick_params(axis="both", which="major", length=1, pad=1, labelsize=4)
                for which in ["bottom", "left"]:
                    ax.spines[which].set_linewidth(0.5)

                if i == 7:
                    ax.set_xlabel(crt_col_j.replace("+", "\n"))
                if j == 0:
                    ax.set_ylabel(crt_col_i.replace("+", "\n"))

for i in range(8):
    sns.despine(left=True, ax=axs[i, i])

fig.savefig(os.path.join(fig_path, "all_mods_pair_coparison.pdf"))

# %% [markdown]
# ## SCRATCH

# %%
calculate_smooth_weight_errors(two_ar3.result_biowta_mods[0, 0, 0][1])

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1,
        3,
        despine_kws={"offset": 5},
        gridspec_kw={"width_ratios": (5, 1, 2)},
        figsize=(5.76, 1.7),
        constrained_layout=True,
    ) as (fig, axs):
        ax_weight_progress, ax_weight_histo, ax_weight_vs_seg = axs

        # draw weight error progression
        max_weight_error = 1.4
        ax = ax_weight_progress
        for crt_idx, crt_roll in enumerate(
            two_ar3.result_biowta_mods[0, 0, 0][1].rolling_weight_errors_normalized
        ):
            crt_kws = {"c": "gray", "lw": 0.5, "alpha": 0.3}
            if crt_idx in two_ar3.good_idxs:
                crt_kws["lw"] = 2.0
                crt_kws["alpha"] = 1.0
                crt_kws["c"] = f"C{two_ar3.good_idxs.index(crt_idx)}"
            ax.plot(*crt_roll, **crt_kws)
        ax.set_ylim(0, max_weight_error)
        ax.set_xlabel("time step")
        ax.set_ylabel("normalized\nreconstruction error")
        ax.set_xlim(0, two_ar3.n_samples)

        # draw the late error distribution
        ax = ax_weight_histo
        late_errors = [
            _[1][-1]
            for _ in two_ar3.result_biowta_mods[0, 0, 0][
                1
            ].rolling_weight_errors_normalized
        ]
        sns.kdeplot(y=late_errors, shade=True, color="gray", ax=ax)
        sns.rugplot(y=late_errors, height=0.1, alpha=0.5, color="gray", ax=ax)
        for i, special_idx in enumerate(two_ar3.good_idxs):
            sns.rugplot(
                y=[late_errors[special_idx]],
                height=0.1,
                alpha=0.5,
                color=f"C{i}",
                lw=2,
                ax=ax,
            )
        ax.set_ylim(0, max_weight_error)
        ax.set_yticks([])
        ax.set_xlabel("pdf")

        # show relation b/w weight reconstruction and segmentation score
        ax = ax_weight_vs_seg
        ar_diffs = [
            np.linalg.norm(np.std([__.a for __ in _], axis=0))
            for _ in two_ar3.dataset.armas
        ]
        h = ax.scatter(
            two_ar3.result_biowta_mods[0, 0, 0][1].trial_scores,
            late_errors,
            s=6,
            c="gray",
            alpha=0.4,
        )
        for i, special_idx in enumerate(two_ar3.good_idxs):
            ax.scatter(
                [two_ar3.result_biowta_mods[0, 0, 0][1].trial_scores[special_idx]],
                [late_errors[special_idx]],
                s=12,
                c=f"C{i}",
                alpha=1.0,
            )
        # colorbar(h)
        ax.set_xlabel("segmentation accuracy")
        ax.set_ylabel("normalized\nreconstruction error")
        ax.set_ylim(0.0, max_weight_error)
        ax.set_xlim(0.5, 1.0)

        ax.yaxis.set_label_position("right")

    sns.despine(ax=ax_weight_vs_seg, left=True, right=False, offset=5)
    sns.despine(ax=ax_weight_histo, left=True)

# %%
