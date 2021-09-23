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

# %%
# %load_ext autoreload
# %autoreload 2

# %matplotlib inline
# %config InlineBackend.print_figure_kwargs = {'bbox_inches': None}

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pandas as pd

from tqdm.notebook import tqdm

from bioslds.arma import Arma
from bioslds.dataset import RandomArmaDataset
from bioslds.plotting import FigureManager, show_latent
from bioslds.cluster_quality import unordered_accuracy_score
from bioslds.batch import hyper_score_ar
from bioslds.regressors import (
    BioWTARegressor,
    CrosscorrelationRegressor,
    CepstralRegressor,
)

from draft_helpers import (
    paper_style,
    calculate_ar_identification_progress,
    make_multi_trajectory_plot,
    make_accuracy_plot,
    predict_plain_score,
    make_accuracy_comparison_diagram,
    get_accuracy_metrics,
    calculate_smooth_weight_errors,
)

fig_path = os.path.join("..", "figs", "draft")

# %% [markdown]
# # Run BioWTA, autocorrelation, and cepstral oracle algorithms on signals based on pairs of AR(3) processes

# %% [markdown]
# ## Define the problem and the parameters for the learning algorithms

# %% [markdown]
# Using best parameters obtained from hyperoptimization runs.

# %%
n_signals = 100
n_samples = 200_000
orders = [(3, 0), (3, 0)]
dwell_times = 100
min_dwell = 50
max_pole_radius = 0.95
normalize = True
fix_scale = None
seed = 153
n_models = 2
n_features = 3
rate_nsm = 0.005028
streak_nsm = 9.527731
rate_cepstral = 0.071844
order_cepstral = 2
metric = unordered_accuracy_score
good_score = 0.85
threshold_steps = 10_000

dataset = RandomArmaDataset(
    n_signals,
    n_samples,
    orders,
    dwell_times=dwell_times,
    min_dwell=min_dwell,
    fix_scale=fix_scale,
    normalize=normalize,
    rng=seed,
    arma_kws={"max_pole_radius": max_pole_radius},
)

# %% [markdown]
# ## Run BioWTA with all combinations of enhancements

# %%
biowta_configurations = {
    (1, 1, 0): {
        "rate": 0.001992,
        "trans_mat": 1 - 1 / 7.794633,
        "temperature": 1.036228,
        "error_timescale": 1.000000,
    },
    (0, 0, 1): {
        "rate": 0.004718,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.000000,
        "error_timescale": 4.216198,
    },
    (1, 1, 1): {
        "rate": 0.004130,
        "trans_mat": 1 - 1 / 5.769690,
        "temperature": 0.808615,
        "error_timescale": 1.470822,
    },
    (0, 1, 1): {
        "rate": 0.004826,
        "trans_mat": 1 - 1 / 2.154856,
        "temperature": 0.000000,
        "error_timescale": 4.566321,
    },
    (1, 0, 1): {
        "rate": 0.006080,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.117712,
        "error_timescale": 4.438448,
    },
    (0, 1, 0): {
        "rate": 0.001476,
        "trans_mat": 1 - 1 / 2.984215,
        "temperature": 0.000000,
        "error_timescale": 1.000000,
    },
    (0, 0, 0): {
        "rate": 0.001199,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.000000,
        "error_timescale": 1.000000,
    },
    (1, 0, 0): {
        "rate": 0.005084,
        "trans_mat": 1 - 1 / 2.000000,
        "temperature": 0.011821,
        "error_timescale": 1.000000,
    },
}
biowta_configurations_human = {
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
result_biowta_mods = {}
for key in tqdm(biowta_configurations, desc="biowta cfg"):
    result_biowta_mods[key] = hyper_score_ar(
        BioWTARegressor,
        dataset,
        metric,
        n_models=n_models,
        n_features=n_features,
        progress=tqdm,
        monitor=["r", "weights_", "prediction_"],
        **biowta_configurations[key],
    )

    crt_scores = result_biowta_mods[key][1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > good_score)
    print(
        f"{''.join(str(_) for _ in key)}: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
    )

# %%
for key in tqdm(biowta_configurations, desc="biowta cfg, reconstruction progress"):
    calculate_ar_identification_progress(result_biowta_mods[key][1].history, dataset)

# %% [markdown]
# Find some "good" indices in the dataset: one that obtains an accuracy score close to a chosen threshold for "good-enough" (which we set to 85%); and one that has a similar score but also has small reconstruction error for the weights.

# %%
result_biowta_chosen = result_biowta_mods[1, 1, 0]
crt_mask = (result_biowta_chosen[1].trial_scores > 0.98 * good_score) & (
    result_biowta_chosen[1].trial_scores < 1.02 * good_score
)
crt_idxs = crt_mask.nonzero()[0]

crt_errors_norm = np.asarray(
    [np.mean(_.weight_errors_normalized_[-1]) for _ in result_biowta_chosen[1].history]
)

good_biowta_idx = crt_idxs[np.argmax(crt_errors_norm[crt_mask])]
good_biowta_ident_idx = crt_idxs[np.argmin(crt_errors_norm[crt_mask])]
good_idxs = [good_biowta_ident_idx, good_biowta_idx]

# %%
result_biowta_chosen[1].trial_scores[good_idxs]

# %%
crt_errors_norm[good_idxs]

# %%
for key in biowta_configurations:
    make_multi_trajectory_plot(
        result_biowta_mods[key][1],
        dataset,
        n_traces=25,
        highlight_idx=good_idxs,
        sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
        trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
        rug_kws={"alpha": 0.3},
    )

# %% [markdown]
# ## Run learning and inference for autocorrelation and cepstral methods

# %%
t0 = time.time()
result_xcorr = hyper_score_ar(
    CrosscorrelationRegressor,
    dataset,
    metric,
    n_models=n_models,
    n_features=n_features,
    nsm_rate=rate_nsm,
    xcorr_rate=1 / streak_nsm,
    progress=tqdm,
    monitor=["r", "nsm.weights_", "xcorr.coef_"],
)
t1 = time.time()
print(
    f"Median accuracy score xcorr: {result_xcorr[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %%
t0 = time.time()
result_cepstral = hyper_score_ar(
    CepstralRegressor,
    dataset,
    metric,
    cepstral_order=order_cepstral,
    cepstral_kws={"rate": rate_cepstral},
    initial_weights="oracle_ar",
    progress=tqdm,
    monitor=["r"],
)
t1 = time.time()
print(
    f"Median accuracy score cepstral: {result_cepstral[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %% [markdown]
# ## Run BioWTA with weights fixed at ground-truth values

# %%
t0 = time.time()
oracle_biowta = hyper_score_ar(
    BioWTARegressor,
    dataset,
    metric,
    n_models=n_models,
    n_features=n_features,
    rate=0,
    trans_mat=biowta_configurations[1, 1, 0]["trans_mat"],
    temperature=biowta_configurations[1, 1, 0]["temperature"],
    error_timescale=biowta_configurations[1, 1, 0]["error_timescale"],
    initial_weights="oracle_ar",
    progress=tqdm,
    monitor=["r", "prediction_"],
)
t1 = time.time()
print(
    f"Median accuracy score oracle BioWTA: {oracle_biowta[0]:.2}. "
    f"(Took {t1 - t0:.2f} seconds.)"
)

# %% [markdown]
# ## Make plots

# %%
fig, axs = make_accuracy_plot(
    result_biowta_chosen[1], oracle_biowta[1], dataset, good_idxs
)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("enh. BioWTA")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_biowta.png"), dpi=600
)

# %%
crt_frac_good = np.mean(result_biowta_chosen[1].trial_scores > good_score)
print(
    f"Percentage of runs with BioWTA accuracies over {int(good_score * 100)}%: "
    f"{int(crt_frac_good * 100)}%."
)

crt_frac_fast = np.mean(
    np.asarray(result_biowta_chosen[1].convergence_times) <= threshold_steps
)
print(
    f"Percentage of runs with BioWTA convergence times under {threshold_steps}: "
    f"{int(crt_frac_fast * 100)}%."
)

# %%
fig, axs = make_accuracy_plot(result_xcorr[1], oracle_biowta[1], dataset, good_idxs)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("autocorrelation")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_xcorr.png"), dpi=600
)

# %%
print(
    f"Percentage of runs with xcorr accuracies over {int(good_score * 100)}%: "
    f"{int(np.mean(result_xcorr[1].trial_scores > good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps}: "
    f"{int(np.mean(np.asarray(result_xcorr[1].convergence_times) <= threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(result_xcorr[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %%
fig, axs = make_accuracy_plot(result_cepstral[1], oracle_biowta[1], dataset, good_idxs)
axs[0, 2].set_xlabel("enh. BioWTA oracle")
axs[0, 2].set_ylabel("cepstral oracle")

fig.savefig(
    os.path.join(fig_path, "rolling_accuracy_2x_ar3_100trials_cepstral.png"), dpi=600
)

# %%
print(
    f"Percentage of runs with cepstral accuracies over {int(good_score * 100)}%: "
    f"{int(np.mean(result_cepstral[1].trial_scores > good_score) * 100)}%."
)
threshold_steps = 10_000
print(
    f"Percentage of runs with cepstral convergence times under {threshold_steps}: "
    f"{int(np.mean(np.asarray(result_cepstral[1].convergence_times) <= threshold_steps) * 100)}%."
)
threshold_steps_small = 1000
print(
    f"Percentage of runs with cepstral convergence times under {threshold_steps_small}: "
    f"{int(np.mean(np.asarray(result_cepstral[1].convergence_times) <= threshold_steps_small) * 100)}%."
)

# %% [markdown]
# # Explain variability in BioWTA accuracy scores, show effect of algorithm improvements

# %%
predicted_plain_scores = [
    predict_plain_score(crt_sig.armas, sigma_ratio=1.0 / crt_sig.scale)
    for crt_sig in tqdm(dataset)
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
    ) as (fig, axs):
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
            crt_ps[crt_cut_idx:], crt_dist[crt_cut_idx:], color=crt_col_err1, alpha=0.3
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
        1, 2, despine_kws={"offset": 5}, figsize=(3, 1.5), constrained_layout=True
    ) as (fig, axs):
        axs[0].plot([0.5, 1], [0.5, 1], "--", c="gray", zorder=-15)
        axs[0].scatter(
            predicted_plain_scores,
            result_biowta_mods[0, 0, 0][1].trial_scores,
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
            result_biowta_chosen[1].trial_scores,
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
    os.path.join(fig_path, "plain_vs_enh_biowta.pdf"), transparent=True,
)

# %%
with plt.style.context(paper_style):
    with FigureManager(despine_kws={"offset": 5}, figsize=(5.76, 1.5)) as (fig, ax):
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
            crt_scores = result_biowta_mods[crt_mod][1].trial_scores
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
                result_biowta_mods[0, 0, 0],
                result_xcorr,
                result_biowta_chosen,
                result_cepstral,
            ],
            ["plain BioWTA", "autocorrelation", "enhanced BioWTA", "cepstral"],
            color_cycle=["C0", "C2", "C3", "C1"],
        )
    ax.tick_params(axis="x", labelsize=8)

# fig.savefig(os.path.join(fig_path, "algo_comparisons.pdf"))

# %%
with plt.style.context(paper_style):
    with FigureManager(despine_kws={"offset": 5}, figsize=(5.76, 3)) as (fig, ax):
        make_accuracy_comparison_diagram(
            ax,
            [
                result_biowta_mods[0, 0, 0],
                result_xcorr,
                result_biowta_chosen,
                result_cepstral,
            ],
            ["plain BioWTA", "autocorrelation", "enhanced BioWTA", "cepstral"],
            color_cycle=["#BBBBBB"],
            highlight_idxs=good_idxs,
            highlight_colors=["C0", "C1"],
        )
    ax.tick_params(axis="x", labelsize=8)

fig.savefig(os.path.join(fig_path, "algo_comparisons.pdf"))

# %% [markdown]
# # Make table of algorithm performance

# %%
for_table = {
    "autocorr.": result_xcorr,
    "plain BioWTA": result_biowta_mods[0, 0, 0],
    "enh. BioWTA": result_biowta_chosen,
    "cepstral": result_cepstral,
}
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
calculate_smooth_weight_errors(result_biowta_chosen[1])

# %%
# for how many runs does the best ground-truth-to-inferred assignment depend on
# whether it's judged by segmentation vs. weight accuracy?
np.sum(
    [
        np.any(_.best_assignment_segmentation != _.best_assignment_weight)
        for _ in result_biowta_chosen[1].history
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
            result_biowta_chosen[1].rolling_weight_errors_normalized
        ):
            crt_kws = {"c": "gray", "lw": 0.5, "alpha": 0.3}
            if crt_idx in good_idxs:
                crt_kws["lw"] = 2.0
                crt_kws["alpha"] = 1.0
                crt_kws["c"] = f"C{good_idxs.index(crt_idx)}"
            ax.plot(*crt_roll, **crt_kws)
        ax.set_ylim(0, max_weight_error)
        ax.set_xlabel("time step")
        ax.set_ylabel("normalized\nreconstruction error")
        ax.set_xlim(0, n_samples)

        # draw the late error distribution
        ax = ax_weight_histo
        late_errors = [
            _[1][-1] for _ in result_biowta_chosen[1].rolling_weight_errors_normalized
        ]
        sns.kdeplot(y=late_errors, shade=True, color="gray", ax=ax)
        sns.rugplot(y=late_errors, height=0.1, alpha=0.5, color="gray", ax=ax)
        for i, special_idx in enumerate(good_idxs):
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
            np.linalg.norm(np.std([__.a for __ in _], axis=0)) for _ in dataset.armas
        ]
        h = ax.scatter(
            result_biowta_chosen[1].trial_scores, late_errors, s=6, c="gray", alpha=0.4,
        )
        for i, special_idx in enumerate(good_idxs):
            ax.scatter(
                [result_biowta_chosen[1].trial_scores[special_idx]],
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
    crt_idxs = good_idxs
    for i, ax in enumerate(axs):
        ax.axhline(0, ls=":", c="gray", lw=0.5, xmax=1.05, clip_on=False)
        ax.axvline(0, ls=":", c="gray", lw=0.5, ymax=1.05, clip_on=False)

        crt_idx = crt_idxs[i]
        crt_true = [_.calculate_poles() for _ in dataset.armas[crt_idx]]
        crt_inferred = [
            Arma(_, []).calculate_poles()
            for _ in result_biowta_chosen[1].history[crt_idx].weights_shuffled_[-1]
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
    legend_ax.plot(
        [label_xs[0] - 0.5, label_xs[1] + 0.16], 2 * [top_y - 0.14], c="gray", lw=0.5
    )
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
        biowta_configurations_human[key]: crt_res[1].trial_scores
        for key, crt_res in result_biowta_mods.items()
    }
)
biowta_mods_weight_errors = pd.DataFrame(
    {
        biowta_configurations_human[key]: np.asarray(
            [
                np.linalg.norm(_.weight_errors_normalized_[-1])
                for _ in crt_res[1].history
            ]
        )
        for key, crt_res in result_biowta_mods.items()
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

# %%
