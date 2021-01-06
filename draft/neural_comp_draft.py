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
import time
import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from types import SimpleNamespace
from typing import Tuple, Callable, Optional, Sequence, Union
from tqdm.notebook import tqdm

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
    {"font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 6, "ytick.labelsize": 6,},
]


# %% [markdown]
# # Useful definitions

# %%
def make_multi_trajector_plot(
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
) -> Tuple[plt.Figure, plt.Axes]:
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
    if highlight_idx is not None and not hasattr(highlight_idx, "__len__"):
        highlight_idx = [highlight_idx]

    fig_kws.setdefault("gridspec_kw", {"width_ratios": (3, 1), "height_ratios": (2, 1)})
    fig_kws.setdefault("figsize", (6, 3))
    with FigureManager(2, 2, **fig_kws) as (
        fig,
        axs,
    ):
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
            # find the index starting at which all scores are above threshold
            # if np.all(crt_mask):
            #     crt_conv_idx = 0
            # else:
            #     crt_conv_idx = len(crt_mask) - np.nonzero(~crt_mask[::-1])[0][0]

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
                # crt_conv_idx = results.convergence_idxs[idx]
                # axs[0, 0].plot(
                #     [crt_rolling[0][crt_conv_idx]],
                #     [crt_rolling[1][crt_conv_idx]],
                #     c="C1",
                #     marker=".",
                # )

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
                trace_kws["c"] = f"C{1 + h_idx}"
                axs[0, 0].plot(*results.rolling_scores[crt_idx], **trace_kws)

            # highlight_color = trace_kws.get("color", trace_kws.get("c"))
        # else:
        # highlight_color = trace_color

        axs[0, 0].set_xlim(0, np.max(crt_rolling[0]))
        axs[0, 0].set_ylim(0.5, 1.0)
        axs[0, 0].set_xlabel("time")
        axs[0, 0].set_ylabel("accuracy")

        # draw the distribution of final accuracy scores
        sns.kdeplot(y=results.trial_scores, shade=True, ax=axs[0, 1])
        rug_kws.setdefault("height", 0.05)

        sns.rugplot(y=results.trial_scores, color=trace_color, ax=axs[0, 1], **rug_kws)
        if highlight_idx is not None:
            highlight_rug_kws = copy.copy(rug_kws)
            highlight_rug_kws["alpha"] = 1.0
            for h_idx, crt_idx in enumerate(highlight_idx):
                sns.rugplot(
                    y=[results.trial_scores[crt_idx]],
                    color=f"C{1 + h_idx}",
                    ax=axs[0, 1],
                    **highlight_rug_kws,
                )

        axs[0, 1].set_ylim(0.5, 1.0)
        axs[0, 1].set_xlabel("pdf")

        # draw the distribution of convergence times
        sns.kdeplot(x=results.convergence_times, shade=True, ax=axs[1, 0])

        rug_kws["height"] = 2 * rug_kws["height"]
        sns.rugplot(x=results.convergence_times, ax=axs[1, 0], **rug_kws)
        if highlight_idx is not None:
            highlight_rug_kws = copy.copy(rug_kws)
            highlight_rug_kws["alpha"] = 1.0
            for h_idx, crt_idx in enumerate(highlight_idx):
                sns.rugplot(
                    x=[results.convergence_times[crt_idx]],
                    color=f"C{1 + h_idx}",
                    ax=axs[1, 0],
                    **highlight_rug_kws,
                )

        axs[1, 0].set_xlim(0, np.max(crt_rolling[0]))
        axs[1, 0].set_xlabel("convergence time")
        axs[1, 0].set_ylabel("pdf")

    sns.despine(left=True, ax=axs[0, 1])
    axs[0, 1].set_yticks([])
    axs[1, 1].set_visible(False)

    return fig, axs


# %%
def make_accuracy_comparison_diagram(
    ax: plt.Axes,
    results: Sequence,
    result_names: Sequence,
    dash_kws: Optional[dict] = None,
    line_kws: Optional[dict] = None,
):
    # handle defaults
    if dash_kws is None:
        dash_kws = {}
    if line_kws is None:
        line_kws = {}

    prev_scores = None
    for i, crt_res in enumerate(results):
        crt_scores = crt_res[1].trial_scores

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
        ax.plot(i * np.ones(n), crt_scores, c=f"C{i}", **crt_kws)

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

    ax.set_ylim(0.5, 1)
    ax.set_ylabel("accuracy score")


# %%
def calculate_ar_identification_progress(
    results: Sequence, dataset: RandomArmaDataset, test_fraction: float = 0.2
):
    for crt_res, crt_sig in zip(tqdm(results), dataset):
        # find the mapping from predicted to ground-truth labels
        inferred_usage = np.argmax(crt_res.r, axis=1)
        crt_n = int(test_fraction * len(crt_res.r))
        _, assignment0 = unordered_accuracy_score(
            crt_sig.usage_seq[-crt_n:], inferred_usage[-crt_n:], return_assignment=True
        )

        # it's more convenient to use ground-truth to predicted mapping
        p = len(assignment0)
        assignment = np.empty(p, dtype=int)
        assignment[assignment0] = np.arange(p)

        # calculate norm of coefficient differences
        ground_weights = np.array([_.a for _ in crt_sig.armas])
        weights = crt_res.weights_[:, assignment, :]
        crt_res.weights_shuffled_ = weights

        diffs = weights - ground_weights[None, :]
        norms = np.linalg.norm(diffs, axis=2) / np.sqrt(p)

        crt_res.weight_errors_ = norms

        # normalize weight errors by difference between ARs
        ground_diff = np.linalg.norm(np.std(ground_weights, axis=0)) / np.sqrt(p)
        crt_res.weight_errors_normalized_ = norms / ground_diff


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

        ax.set_xlabel("time")
        ax.set_ylabel("AR coefficients")

        # ax.legend(ncol=3, frameon=False, fontsize=6)
        ax.set_title(f"model {i + 1}")

    if use_same_range:
        max_ylims = (min(_[0] for _ in ylims), max(_[1] for _ in ylims))
        for ax in axs:
            ax.set_ylim(*max_ylims)

# %% [markdown]
# # Problem setup

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
with FigureManager(despine_kws={"left": True}) as (_, ax):
    ax.plot(problem_setup.sig)
    ax.set_xlabel("time")

    ax.set_xlim(0, problem_setup.n_samples)
    ax.set_yticks([])
    
    show_latent(problem_setup.usage_seq)

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
        ax1.set_xlabel("time")

        ax1.set_xlim(0, problem_setup.n_samples)
        ax1.set_yticks([])

        show_latent(problem_setup.usage_seq, ax=ax1)

    sns.despine(left=True, ax=ax1)

fig.savefig(os.path.join(fig_path, "example_switching_signal_and_z.pdf"))

# %% [markdown]
# # Problem setup, v2

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
with FigureManager(despine_kws={"left": True}) as (_, ax):
    ax.plot(problem_setup_v2.sig)
    ax.set_xlabel("time")

    ax.set_xlim(0, problem_setup_v2.n_samples)
    ax.set_yticks([])
    
    show_latent(problem_setup_v2.usage_seq)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        2, 1, figsize=(5.76, 2.5), sharex=True, gridspec_kw={"height_ratios": (3, 1.5)}
    ) as (fig, (ax0, ax1)):
        ax0.axhline(0, ls=":", c="gray", zorder=-1)
        ax0.plot(problem_setup_v2.sig)
        yl = np.max(np.abs(problem_setup_v2.sig))
        ax0.set_ylim(-yl, yl)

        show_latent(problem_setup_v2.usage_seq, ax=ax0)

        # ax0.set_xlabel("time")
        ax0.set_ylabel("signal")

        ax0.set_xlim(0, problem_setup_v2.n_samples)
        ax0.set_yticks([])

        # start with ground truth, then add some noise, and finally smooth and normalize
        problem_setup_v2.mock_z = np.asarray(
            [problem_setup_v2.usage_seq == [0, 2, 1][_] for _ in range(3)]
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

        ax1.set_xlabel("time")
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
    # n_signals=10,
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
    # rate_biowta=0.00247,
    # streak_biowta=5.38,
    # temperature_biowta=0.800,
    # timescale_biowta=2.40,
    rate_biowta=0.001992,
    streak_biowta=7.794633,
    temperature_biowta=1.036228,
    timescale_biowta=1.0,
    # rate_nsm=0.00585,
    # streak_nsm=10.4,
    rate_nsm=0.005028,
    streak_nsm=9.527731,
    # rate_cepstral=0.0755,
    # order_cepstral=3,
    rate_cepstral=0.071844,
    order_cepstral=2,
    metric=unordered_accuracy_score,
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
# ## Run learning and inference

# %%
t0 = time.time()
two_ar3.result_biowta = hyper_score_ar(
    BioWTARegressor,
    two_ar3.dataset,
    two_ar3.metric,
    n_models=two_ar3.n_models,
    n_features=two_ar3.n_features,
    rate=two_ar3.rate_biowta,
    trans_mat=1 - 1 / two_ar3.streak_biowta,
    temperature=two_ar3.temperature_biowta,
    error_timescale=two_ar3.timescale_biowta,
    progress=tqdm,
    monitor=["r", "weights_", "prediction_"],
)
t1 = time.time()
print(
    f"Median accuracy score BioWTA: {two_ar3.result_biowta[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %%
print(
    f"Accuracy score for bottom 5% of runs: "
    f"{np.quantile(two_ar3.result_biowta[1].trial_scores, 0.05):.3f}."
)

# %% [markdown]
# Calculate how quickly each run's inferred weights approach the ground truth.

# %%
calculate_ar_identification_progress(two_ar3.result_biowta[1].history, two_ar3.dataset)

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
    trans_mat=1 - 1 / two_ar3.streak_biowta,
    temperature=two_ar3.temperature_biowta,
    error_timescale=two_ar3.timescale_biowta,
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

# %% [markdown]
# Find some "good" indices: one that yields a reasonably good score despite having not-so-good reproduction of the weights; and one that minimizes the reconstruction error for the weights.

# %%
good_score = 0.85
good_idx = np.argmin(np.abs(two_ar3.result_biowta[1].trial_scores - good_score))

late_errors_norm = np.asarray(
    [np.mean(_.weight_errors_normalized_[-1]) for _ in two_ar3.result_biowta[1].history]
)
good_ident_idx = np.argmin(late_errors_norm)

# %%
# tmp_mask = late_errors_norm > 1.0
# tmp_mask_idxs = tmp_mask.nonzero()[0]
# tmp_idx = tmp_mask_idxs[
#     np.argmin(np.abs(two_ar3.result_biowta[1].trial_scores[tmp_mask] - good_score))
# ]

# %%
with plt.style.context(paper_style):
    fig, axs = make_multi_trajector_plot(
        two_ar3.result_biowta[1],
        two_ar3.dataset,
        n_traces=25,
        highlight_idx=[good_idx, good_ident_idx],
        sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
        trace_kws={"alpha": 0.85, "lw": 0.75},
        fig_kws={"figsize": (5.76, 3), "despine_kws": {"offset": 5}},
        rug_kws={"alpha": 0.3},
    )
    axs[0, 0].set_xticks(np.arange(0, 200_000, 50_000))
    axs[1, 0].set_xticks(np.arange(0, 200_000, 50_000))

    axs[1, 1].set_visible(True)
    axs[1, 1].plot([0.5, 1], [0.5, 1], "k--", lw=0.5, alpha=0.5)
    axs[1, 1].scatter(
        two_ar3.oracle_biowta[1].trial_scores,
        two_ar3.result_biowta[1].trial_scores,
        s=2,
        alpha=0.5,
    )
    axs[1, 1].scatter(
        [two_ar3.oracle_biowta[1].trial_scores[good_idx]],
        [two_ar3.result_biowta[1].trial_scores[good_idx]],
        s=2,
        c="C1",
        alpha=1.0,
    )
    axs[1, 1].set_xlim(0.5, 1.0)
    axs[1, 1].set_ylim(0.5, 1.0)
    axs[1, 1].set_aspect(1.0)

    axs[1, 1].set_xlabel("oracle")
    axs[1, 1].set_ylabel("actual")

    fig.savefig(
        os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_biowta.png"), dpi=600
    )

# %%
print(
    f"Percentage of runs with BioWTA accuracies over {int(good_score * 100)}%: "
    f"{int(np.mean(two_ar3.result_biowta[1].trial_scores > good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with BioWTA convergence times under {threshold_steps}: "
    f"{int(np.mean(np.asarray(two_ar3.result_biowta[1].convergence_times) <= threshold_steps) * 100)}%."
)

# %%
with plt.style.context(paper_style):
    fig, axs = make_multi_trajector_plot(
        two_ar3.result_xcorr[1],
        two_ar3.dataset,
        n_traces=25,
        highlight_idx=[good_idx, good_ident_idx],
        sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
        trace_kws={"alpha": 0.85, "lw": 0.75},
        fig_kws={"figsize": (5.76, 3), "despine_kws": {"offset": 5}},
        rug_kws={"alpha": 0.3},
    )
    axs[0, 0].set_xticks(np.arange(0, 200_000, 50_000))
    axs[1, 0].set_xticks(np.arange(0, 200_000, 50_000))

    axs[1, 1].set_visible(True)
    axs[1, 1].plot([0.5, 1], [0.5, 1], "k--", lw=0.5, alpha=0.5)
    axs[1, 1].scatter(
        two_ar3.oracle_biowta[1].trial_scores,
        two_ar3.result_xcorr[1].trial_scores,
        s=2,
        alpha=0.5,
    )
    axs[1, 1].scatter(
        [two_ar3.oracle_biowta[1].trial_scores[good_idx]],
        [two_ar3.result_xcorr[1].trial_scores[good_idx]],
        s=2,
        c="C1",
        alpha=1.0,
    )
    axs[1, 1].set_xlim(0.5, 1.0)
    axs[1, 1].set_ylim(0.5, 1.0)
    axs[1, 1].set_aspect(1.0)

    axs[1, 1].set_xlabel("oracle")
    axs[1, 1].set_ylabel("actual")

    fig.savefig(
        os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_xcorr.png"), dpi=600
    )

# %%
print(
    f"Percentage of runs with xcorr accuracies over {int(good_score * 100)}%: "
    f"{int(np.mean(two_ar3.result_xcorr[1].trial_scores > good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps}: "
    f"{int(np.mean(np.asarray(two_ar3.result_xcorr[1].convergence_times) <= threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(two_ar3.result_xcorr[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %%
with plt.style.context(paper_style):
    fig, axs = make_multi_trajector_plot(
        two_ar3.result_cepstral[1],
        two_ar3.dataset,
        n_traces=25,
        highlight_idx=[good_idx, good_ident_idx],
        sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
        trace_kws={"alpha": 0.85, "lw": 0.75},
        fig_kws={"figsize": (5.76, 3), "despine_kws": {"offset": 5}},
        rug_kws={"alpha": 0.3},
    )
    axs[0, 0].set_xticks(np.arange(0, 200_000, 50_000))
    axs[1, 0].set_xticks(np.arange(0, 200_000, 50_000))

    axs[1, 1].set_visible(True)
    axs[1, 1].plot([0.5, 1], [0.5, 1], "k--", lw=0.5, alpha=0.5)
    axs[1, 1].scatter(
        two_ar3.oracle_biowta[1].trial_scores,
        two_ar3.result_cepstral[1].trial_scores,
        s=2,
        alpha=0.5,
    )
    axs[1, 1].scatter(
        [two_ar3.oracle_biowta[1].trial_scores[good_idx]],
        [two_ar3.result_cepstral[1].trial_scores[good_idx]],
        s=2,
        c="C1",
        alpha=1.0,
    )
    axs[1, 1].set_xlim(0.5, 1.0)
    axs[1, 1].set_ylim(0.5, 1.0)
    axs[1, 1].set_aspect(1.0)

    axs[1, 1].set_xlabel("oracle")
    axs[1, 1].set_ylabel("actual")

    fig.savefig(
        os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_cepstral.png"), dpi=600
    )

# %%
print(
    f"Percentage of runs with cepstral accuracies over {int(good_score * 100)}%: "
    f"{int(np.mean(two_ar3.result_cepstral[1].trial_scores > good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with cepstral convergence times under {threshold_steps}: "
    f"{int(np.mean(np.asarray(two_ar3.result_cepstral[1].convergence_times) <= threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with cepstral convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(two_ar3.result_cepstral[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %% [markdown]
# # Run BioWTA with all combinations of improvements

# %%
two_ar3.configurations = {
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
two_ar3.configurations_human = {
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
two_ar3.result_mods = {}
for key in tqdm(two_ar3.configurations, desc="cfg"):
    two_ar3.result_mods[key] = hyper_score_ar(
        BioWTARegressor,
        two_ar3.dataset,
        two_ar3.metric,
        n_models=two_ar3.n_models,
        n_features=two_ar3.n_features,
        progress=tqdm,
        monitor=["r", "weights_", "prediction_"],
        **two_ar3.configurations[key],
    )

    crt_scores = two_ar3.result_mods[key][1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > good_score)
    print(
        f"{''.join(str(_) for _ in key)}: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
    )


# %% [markdown]
# # Explain variability in BioWTA accuracy scores, show effect of algorithm improvements

# %%
def predict_plain_score(armas: Sequence, sigma: float = 1) -> float:
    delta = np.linalg.norm(armas[1].a - armas[0].a)
    return 0.5 + np.arctan(delta * sigma * np.sqrt(np.pi / 8)) / np.pi


# %%
predicted_plain_scores = [
    predict_plain_score(crt_sig.armas) for crt_sig in tqdm(two_ar3.dataset)
]

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1, 3, gridspec_kw={"width_ratios": (3, 5, 1)}, sharey=True, despine_kws={"offset": 5}, figsize=(5.76, 1.5)
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
        axs[0].plot(crt_samples, ".-", c="C0")
        axs[0].axhline(0, ls=":", c="gray")

        crt_box = [[crt_n - 0.4, crt_n + 0.4], [-1.4, 1.4]]
        axs[0].plot(
            crt_box[0] + crt_box[0][::-1] + [crt_box[0][0]],
            [crt_box[1][0]] + crt_box[1] + crt_box[1][::-1],
            "k-",
        )

        crt_p_range = (-1.5, 1.5)
        axs[0].set_ylim(*crt_p_range)
        axs[0].set_xlabel("sample")
        axs[0].set_ylabel("signal")

        show_latent(crt_usage, ax=axs[0])

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
                c="C0",
                alpha=0.5 * crt_p,
            )
            axs[0].plot(
                [crt_box[0][0] + 0.01, crt_box[0][1] - 0.01],
                [crt_y, crt_y],
                c="C0",
                alpha=0.5 * crt_p,
            )

        crt_col1 = "C0"
        crt_col2 = "C1"
        crt_col_err1 = "C1"
        crt_col_err2 = "C4"

        axs[1].axhline(crt_pred1, c=crt_col1, ls=":")
        axs[1].axhline(crt_pred2, c=crt_col2, ls=":")
        axs[1].axhline(crt_thresh, c="gray", ls="--")

        rng = np.random.default_rng(0)
        crt_n = 100
        crt_samples1 = rng.normal(crt_pred1, crt_sigma, size=crt_n)
        # crt_samples2 = rng.normal(crt_pred2, crt_sigma, size=crt_n)

        crt_correct1 = crt_samples1 < crt_thresh
        crt_idxs = np.arange(crt_n)
        crt_ms = 6
        axs[1].plot(
            crt_idxs[crt_correct1],
            crt_samples1[crt_correct1],
            ".",
            c=crt_col1,
            ms=crt_ms,
        )
        axs[1].plot(
            crt_idxs[~crt_correct1],
            crt_samples1[~crt_correct1],
            ".",
            c=crt_col_err1,
            ms=crt_ms,
        )

        crt_x0 = -35
        axs[1].set_xlim(crt_x0, crt_n)

        axs[1].annotate(
            "prediction 1",
            xy=(crt_x0, crt_pred1),
            verticalalignment="bottom",
            fontweight="bold",
            fontsize=7,
            color=crt_col1,
        )
        axs[1].annotate(
            "prediction 2",
            xy=(crt_x0, crt_pred2),
            verticalalignment="bottom",
            fontweight="bold",
            fontsize=7,
            color=crt_col2,
        )
        axs[1].annotate(
            "threshold",
            xy=(crt_x0, crt_thresh),
            verticalalignment="bottom",
            fontweight="bold",
            fontsize=7,
            color="gray",
        )

        axs[1].set_xlabel("random draw")
        axs[1].set_ylabel("possible $y(t)$")

        crt_cut_idx = np.argmin(np.abs(crt_ps - crt_thresh))
        axs[2].plot(crt_dist[: crt_cut_idx + 1], crt_ps[: crt_cut_idx + 1], c=crt_col1)
        axs[2].plot(crt_dist[crt_cut_idx:], crt_ps[crt_cut_idx:], c=crt_col_err1)

        axs[2].fill_betweenx(
            crt_ps[: crt_cut_idx + 1],
            crt_dist[: crt_cut_idx + 1],
            color=crt_col1,
            alpha=0.3,
        )
        axs[2].fill_betweenx(
            crt_ps[crt_cut_idx:], crt_dist[crt_cut_idx:], color=crt_col_err1, alpha=0.3,
        )

        axs[2].axhline(crt_pred1, c=crt_col1, ls=":")
        axs[2].axhline(crt_pred2, c=crt_col2, ls=":")
        axs[2].axhline(crt_thresh, c="gray", ls="--")

        axs[2].set_xlim(0, None)

        axs[2].set_xlabel("pdf")

fig.savefig(
    os.path.join(fig_path, "explanation_for_biowta_segmentation_errors.pdf"), transparent=True
)

# %%
# with plt.style.context(paper_style):
#     with FigureManager(1, 4, despine_kws={"offset": 5}, figsize=(5.76, 1.5)) as (
#         fig,
#         axs,
#     ):
#         results_sequence = [
#             ("plain", two_ar3.result_biowta_plain),
#             ("soft", two_ar3.result_biowta_soft),
#             ("persistent", two_ar3.result_biowta_soft_persistent),
#             ("full", two_ar3.result_biowta),
#         ]
#         additions = ["", "soft", "persistent", "averaging"]
#         for i, (ax, crt_res) in enumerate(zip(axs, results_sequence)):
#             ax.plot([0.5, 1], [0.5, 1], "--", c="gray", zorder=-15)
#             ax.scatter(
#                 predicted_plain_scores, crt_res[1][1].trial_scores, s=4, alpha=0.5
#             )

#             ax.set_aspect(1)
#             ax.set_xlabel("naive score")

#             if i == 0:
#                 ax.set_ylabel("score plain")
#             else:
#                 ax.set_ylabel("+ " + additions[i])

#             ax.set_xlim([0.5, 1])
#             ax.set_ylim([0.5, 1])

# fig.savefig(
#     os.path.join(fig_path, "effect_of_biowta_improvements.pdf"), transparent=True
# )

# %%
with plt.style.context(paper_style):
    with FigureManager(1, 4, despine_kws={"offset": 5}, figsize=(5.76, 1.5)) as (
        fig,
        axs,
    ):
        results_sequence = [
            ("plain", two_ar3.result_mods[0, 0, 0]),
            ("persistent", two_ar3.result_mods[0, 1, 0]),
            ("soft+persist.", two_ar3.result_mods[1, 1, 0]),
            ("avg. only", two_ar3.result_mods[0, 0, 1]),
        ]
        for i, (ax, crt_res) in enumerate(zip(axs, results_sequence)):
            ax.plot([0.5, 1], [0.5, 1], "--", c="gray", zorder=-15)
            ax.scatter(
                predicted_plain_scores, crt_res[1][1].trial_scores, s=4, alpha=0.5
            )

            ax.set_aspect(1)
            ax.set_xlabel("naive score")

            ax.set_ylabel(crt_res[0])

            ax.set_xlim([0.5, 1])
            ax.set_ylim([0.5, 1])

#         ax = axs[-1]
#         ax.bar(
#             np.arange(8),
#             [
#                 np.mean(_[1].trial_scores > good_score)
#                 for _ in two_ar3.result_mods.values()
#             ],
#             width=0.3,
#         )
#         ax.set_xticks(np.arange(8))
#         ax.set_xticklabels(
#             [two_ar3.configurations_human[_] for _ in two_ar3.result_mods.keys()]
#         )

#     ax.xaxis.set_tick_params(rotation=90)

# axs[-1].set_visible(False)

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
                two_ar3.result_mods[0, 0, 0],
                two_ar3.result_biowta,
                two_ar3.result_cepstral,
            ],
            ["autocorrelation", "plain BioWTA", "full BioWTA", "cepstral"],
        )

fig.savefig(os.path.join(fig_path, "algo_comparisons.pdf"))

# %% [markdown]
# # Plot progression of system identification

# %%
late_errors_norm = np.asarray(
    [np.mean(_.weight_errors_normalized_[-1]) for _ in two_ar3.result_biowta[1].history]
)
with plt.style.context(paper_style):
    fig = plt.figure(figsize=(5.76, 2.75), constrained_layout=True)
    gs = fig.add_gridspec(2, 10)

    axs = []
    weight_axs = []

    # axes for showing weight learning in good_ident_idx run
    axs.append(fig.add_subplot(gs[0, 0:3]))
    axs.append(fig.add_subplot(gs[0, 3:6]))
    weight_axs.append(axs[:2])

    # axes for showing weight learning in good_idx run
    axs.append(fig.add_subplot(gs[1, 0:3]))
    axs.append(fig.add_subplot(gs[1, 3:6]))
    weight_axs.append(axs[2:4])

    # axes for showing overall progression of weight learning
    axs.append(fig.add_subplot(gs[0, 6:9]))

    # axes for relation b/w weight reconstruction and segmentation score
    axs.append(fig.add_subplot(gs[1, 6:9]))

    # axes for showing histogram of final weight errors
    axs.append(fig.add_subplot(gs[0, -1]))

    # now draw weight learning
    weight_axs = np.asarray(weight_axs)
    for i, trial_idx in enumerate([good_ident_idx, good_idx]):
        show_weight_progression(
            weight_axs[i, :],
            two_ar3.result_biowta[1].history[trial_idx],
            two_ar3.dataset.armas[trial_idx],
        )
        weight_axs[i, 0].set_ylabel("AR coeffs.")
        weight_axs[i, 1].set_ylabel("")

    # next draw weight error progression
    crt_win_step = 1000
    crt_win_size = 5000
    crt_loc = np.arange(0, two_ar3.n_samples, crt_win_step)
    ax = axs[4]
    max_weight_error = 3.0
    for crt_idx, crt_res in enumerate(tqdm(two_ar3.result_biowta[1].history)):
        crt_err = crt_res.weight_errors_
        crt_err_norm = crt_res.weight_errors_normalized_
        crt_err_smooth = np.zeros(len(crt_loc))
        crt_err_norm_smooth = np.zeros(len(crt_loc))
        for i, k in enumerate(crt_loc):
            crt_err_smooth[i] = np.mean(crt_err[k : k + crt_win_size])
            crt_err_norm_smooth[i] = np.mean(crt_err_norm[k : k + crt_win_size])

        crt_kws = {"c": "C0", "lw": 0.5, "alpha": 0.5}
        if crt_idx == good_idx or crt_idx == good_ident_idx:
            crt_kws["lw"] = 2.0
            crt_kws["alpha"] = 1.0
            crt_kws["c"] = "C1" if crt_idx == good_idx else "C2"
        ax.plot(crt_loc, crt_err_norm_smooth, **crt_kws)
    ax.set_ylim(0, max_weight_error)
    ax.set_xlabel("time")
    ax.set_ylabel("AR coeff. error")
    ax.set_xlim(0, two_ar3.n_samples)

    # draw the late error distribution
    ax = axs[-1]
    sns.kdeplot(y=late_errors_norm, shade=True, ax=ax)
    sns.rugplot(y=late_errors_norm, height=0.1, alpha=0.5, ax=ax)
    ax.set_ylim(0, max_weight_error)
    ax.set_yticks([])
    ax.set_xlabel("pdf")

    # show relation b/w weight reconstruction and segmentation score
    ax = axs[5]
    ar_diffs = [
        np.linalg.norm(np.std([__.a for __ in _], axis=0))
        for _ in two_ar3.dataset.armas
    ]
    h = ax.scatter(
        two_ar3.result_biowta[1].trial_scores,
        late_errors_norm,
        s=6,
        # c=predicted_plain_scores,
        # c=ar_diffs,
        # cmap="Reds",
        # vmin=0.5,
        # vmax=1.0,
        alpha=0.4,
    )
    # colorbar(h)
    ax.set_xlabel("segmentation accuracy")
    ax.set_ylabel("AR coeff. error")
    ax.set_ylim(0, 2.5)
    ax.set_xlim(0.5, 1.0)

    # despine
    for ax in axs:
        sns.despine(offset=5, ax=ax)

fig.savefig(
    os.path.join(fig_path, "biowta_weight_reconstruction.png"), dpi=600,
)

# %% [markdown]
# ## Other attempts at visualization

# %%
actual_poles = np.asarray(
    [[_.calculate_poles() for _ in crt_sig.armas] for crt_sig in tqdm(two_ar3.dataset)]
)
inferred_poles = np.asarray(
    [
        [Arma(crt_w, []).calculate_poles() for crt_w in crt_res.weights_shuffled_[-1]]
        for crt_res in two_ar3.result_biowta[1].history
    ]
)

# %%
with FigureManager() as (_, ax):
    all_actual = np.ravel(actual_poles)
    all_inferred = np.ravel(inferred_poles)

    crt_theta = np.linspace(0, 2 * np.pi, 360)
    ax.plot(np.cos(crt_theta), np.sin(crt_theta), "k--")
    ax.plot(
        two_ar3.max_pole_radius * np.cos(crt_theta),
        two_ar3.max_pole_radius * np.sin(crt_theta),
        "--",
        c="gray",
    )

    ax.plot(np.real(all_actual), np.imag(all_actual), "x", ls="none", c="C0", alpha=0.6)
    ax.plot(
        np.real(all_inferred), np.imag(all_inferred), "x", ls="none", c="C1", alpha=0.3
    )

    crt_dashes = np.column_stack(
        (all_actual, all_inferred, np.repeat(np.nan, len(all_actual)))
    ).ravel()
    ax.plot(np.real(crt_dashes), np.imag(crt_dashes), c="gray", alpha=0.2)

    ax.set_aspect(1)

# %%
with FigureManager() as (_, ax):
    crt_range = slice(-1500, None)
    crt_sig = two_ar3.dataset[good_idx]
    crt_n = len(crt_sig.y)
    ax.plot(np.arange(crt_n)[crt_range], crt_sig.y[crt_range])
    show_latent(crt_sig.usage_seq, bar_location="bottom")
    crt_history = two_ar3.result_biowta[1].history[good_idx]
    show_latent(np.argmax(crt_history.r, axis=1), show_vlines=False)

# %%
with FigureManager(1, 2) as (_, (ax1, ax2)):
    crt_win_step = 1000
    crt_win_size = 5000
    crt_loc = np.arange(0, two_ar3.n_samples, crt_win_step)
    for crt_idx, crt_res in enumerate(tqdm(two_ar3.result_biowta[1].history)):
        crt_err = crt_res.weight_errors_
        crt_err_norm = crt_res.weight_errors_normalized_
        crt_err_smooth = np.zeros(len(crt_loc))
        crt_err_norm_smooth = np.zeros(len(crt_loc))
        for i, k in enumerate(crt_loc):
            crt_err_smooth[i] = np.mean(crt_err[k : k + crt_win_size])
            crt_err_norm_smooth[i] = np.mean(crt_err_norm[k : k + crt_win_size])

        crt_kws = {"c": "C0", "lw": 0.5, "alpha": 0.8}
        if crt_idx == good_idx:
            crt_kws["lw"] = 2.0
            crt_kws["alpha"] = 1.0
            crt_kws["c"] = "C1"
        ax1.plot(crt_loc, crt_err_smooth, **crt_kws)
        ax2.plot(crt_loc, crt_err_norm_smooth, **crt_kws)
    
    ax2.set_ylim(0, 2)

# %%
with FigureManager() as (_, ax):
    late_errors_norm = np.asarray(
        [
            np.mean(_.weight_errors_normalized_[-1])
            for _ in two_ar3.result_biowta[1].history
        ]
    )
    # sns.regplot(x=two_ar3.result_biowta[1].trial_scores, y=late_errors_norm, scatter_kws={"c": predicted_plain_scores}, ax=ax)
    ar_diffs = [
        np.linalg.norm(np.std([__.a for __ in _], axis=0))
        for _ in two_ar3.dataset.armas
    ]
    h = ax.scatter(
        two_ar3.result_biowta[1].trial_scores,
        late_errors_norm,
        # c=predicted_plain_scores,
        c=ar_diffs,
        cmap="Reds",
        # vmin=0.5,
        # vmax=1.0,
        alpha=0.7,
    )
    colorbar(h)

# %%
with FigureManager() as (_, ax):
    sns.kdeplot(late_errors_norm, shade=True, log_scale=True)
    sns.rugplot(late_errors_norm)

# %%
