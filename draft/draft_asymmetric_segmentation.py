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
# %matplotlib inline
# %config InlineBackend.print_figure_kwargs = {'bbox_inches': None}

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm.notebook import tqdm

from bioslds.arma import make_random_arma
from bioslds.dataset import RandomArmaDataset
from bioslds.plotting import FigureManager, show_latent
from bioslds.batch import hyper_score_ar
from bioslds.regressors import BioWTARegressor
from bioslds.cluster_quality import unordered_accuracy_score

from draft_helpers import (
    paper_style,
    make_multi_trajectory_plot,
    calculate_asymmetry_measures,
)

fig_path = os.path.join("..", "figs", "draft")

# %% [markdown]
# # Analyze asymmetry of segmentation errors

# %% [markdown]
# We first generate `n_datasets` pairs of AR(3) processes, similar to `draft_two_ar3.ipynb`. For each of these, we generate `n_signals` random datasets that use the same AR(3) processes but different random realizations of the semi-Markov chain of latent states. We then run `BioWTA` on all of these and measure the asymmetry in the confusion matrix (see `calculate_asymmetry_measures`). This should help us get an idea for what property of a process makes it harder to identify correctly.

# %%
n_datasets = 50
n_signals = 100
n_samples = 200_000
n_models = 2
n_features = 3
dwell_times = 100
min_dwell = 50
normalize = True
max_pole_radius = 0.95
seed = 121
metric = unordered_accuracy_score
wta_params = {
    "rate": 0.005460,
    "trans_mat": 1 - 1 / 6.792138,
    "temperature": 0.961581,
    "error_timescale": 1.000000,
}
arma_order = (3, 0)
good_score = 0.85

rng = np.random.default_rng(seed)
arma_pairs = []
for i in range(n_datasets):
    arma_pairs.append(
        (
            make_random_arma(*arma_order, rng=rng, max_pole_radius=max_pole_radius),
            make_random_arma(*arma_order, rng=rng, max_pole_radius=max_pole_radius),
        )
    )

datasets = [
    RandomArmaDataset(
        n_signals,
        n_samples,
        armas=_,
        dwell_times=dwell_times,
        min_dwell=min_dwell,
        normalize=normalize,
        rng=seed,
    )
    for _ in arma_pairs
]

# %%
results_biowta = []
for dataset in tqdm(datasets, desc="dataset"):
    crt_res = hyper_score_ar(
        BioWTARegressor,
        dataset,
        metric,
        n_models=n_models,
        n_features=n_features,
        progress=tqdm,
        monitor=["r"],
        **wta_params,
    )

    crt_scores = crt_res[1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > good_score)
    print(
        f"BioWTA on repeated runs of same ARMA pair score: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
    )

    results_biowta.append(crt_res)

# %%
make_multi_trajectory_plot(
    results_biowta[0][1],
    datasets[0],
    n_traces=25,
    sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
    highlight_idx=10,
    trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
    rug_kws={"alpha": 0.3},
)

# %%
for i, crt_res in enumerate(tqdm(results_biowta, desc="dataset")):
    calculate_asymmetry_measures(crt_res[1], datasets[i])

# %%
confusion_bias = []
confusion_bias_range = []
max_pole_radius = np.zeros((n_datasets, n_models))
median_pole_radius = np.zeros((n_datasets, n_models))
min_pole_radius = np.zeros((n_datasets, n_models))

max_abs_angle = np.zeros((n_datasets, n_models))
min_abs_angle = np.zeros((n_datasets, n_models))

neg_pole = np.zeros((n_datasets, n_models))
for i, crt_res in enumerate(results_biowta):
    crt_diffs = []
    for crt_asymmetry in crt_res[1].asymmetry:
        crt_confusion = crt_asymmetry.confusion_ordered
        crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
        crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
        crt_diffs.append(crt_diff)

    confusion_bias.append(np.median(crt_diffs))
    confusion_bias_range.append(np.quantile(crt_diffs, [0.25, 0.75]))

    crt_armas = arma_pairs[i]
    crt_poles = [_.calculate_poles() for _ in crt_armas]
    crt_pole_radii = [[np.linalg.norm(crt_pole) for crt_pole in _] for _ in crt_poles]
    crt_abs_angles = [[np.abs(np.angle(crt_pole)) for crt_pole in _] for _ in crt_poles]

    crt_max_radii = [np.max(_) for _ in crt_pole_radii]
    crt_median_radii = [np.median(_) for _ in crt_pole_radii]
    crt_min_radii = [np.min(_) for _ in crt_pole_radii]

    crt_max_angles = [np.max(_) for _ in crt_abs_angles]
    crt_min_angles = [np.min(_) for _ in crt_abs_angles]

    max_pole_radius[i, :] = crt_max_radii
    median_pole_radius[i, :] = crt_median_radii
    min_pole_radius[i, :] = crt_min_radii

    max_abs_angle[i, :] = crt_max_angles
    min_abs_angle[i, :] = crt_min_angles

    neg_pole[i, :] = [np.abs(_[np.isreal(_)]) for _ in crt_poles]

confusion_bias = np.array(confusion_bias)
confusion_bias_range = np.array(confusion_bias_range)

# %%
crt_poles

# %%
crt_abs_angles

# %%
measures = {
    "max pole radii": max_pole_radius[:, 0] - max_pole_radius[:, 1],
    "median pole radii": median_pole_radius[:, 0] - median_pole_radius[:, 1],
    "min pole radii": min_pole_radius[:, 0] - min_pole_radius[:, 1],
    "max abs angle": max_abs_angle[:, 0] - max_abs_angle[:, 1],
    "min abs angle": max_abs_angle[:, 0] - max_abs_angle[:, 1],
    "value of real pole": neg_pole[:, 0] - neg_pole[:, 1],
}

with FigureManager(len(measures) // 2, 2) as (_, axs):
    axs = axs.ravel()
    for i, measure in enumerate(measures):
        axs[i].errorbar(
            measures[measure],
            confusion_bias,
            yerr=[
                confusion_bias - confusion_bias_range[:, 0],
                confusion_bias_range[:, 1] - confusion_bias,
            ],
            ecolor="gray",
            ls="none",
        )
        # axs[i].scatter(measures[measure], confusion_bias)
        sns.regplot(
            x=measures[measure], y=confusion_bias, color="C0", truncate=False, ax=axs[i]
        )
        axs[i].set_xlabel("difference in " + measure)

        axs[i].set_ylim(-0.3, 0.3)

# %%
with FigureManager() as (_, ax):
    crt_res = results_biowta[0]
    crt_diffs = []
    for crt_asymmetry in crt_res[1].asymmetry:
        crt_confusion = crt_asymmetry.confusion_ordered
        crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
        crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
        crt_diffs.append(crt_diff)
    
    ax.hist(crt_diffs)

# %%
with FigureManager() as (_, ax):
    crt_diffs = []
    for crt_res in results_biowta:
        for crt_asymmetry in crt_res[1].asymmetry:
            crt_confusion = crt_asymmetry.confusion_ordered
            crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
            crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
            crt_diffs.append(crt_diff)
    
    ax.hist(crt_diffs, 100)

# %%
(np.argmin(confusion_bias), np.argmax(confusion_bias))

# %%
np.argsort(confusion_bias)


# %%
def get_stds(sig):
    return (np.std(sig.y[sig.usage_seq == 0]), np.std(sig.y[sig.usage_seq == 1]))


# %%
(
    get_stds(datasets[9][0]),
    get_stds(datasets[45][0]),
    get_stds(datasets[39][0]),
    get_stds(datasets[13][0]),
    get_stds(datasets[36][0]),
    get_stds(datasets[33][0]),
)

# %%
confusion_bias_range[[9, 45, 39, 13, 36, 33]]

# %%
[get_stds(datasets[_][0]) for _ in np.argsort(confusion_bias)]

# %%
crt_idxs = [9, 45, 36, 33]
with FigureManager(len(crt_idxs), 1) as (_, axs):
    for crt_idx, ax in zip(crt_idxs, axs):
        crt_sig = datasets[crt_idx][0]
        ax.plot(crt_sig.y[:1000])
        show_latent(crt_sig.usage_seq, ax=ax)

# %%
