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

from bioslds.arma import make_random_arma
from bioslds.dataset import RandomSnippetDataset
from bioslds.plotting import FigureManager, make_gradient_cmap
from bioslds.cluster_quality import unordered_accuracy_score
from bioslds.regressors import BioWTARegressor, CrosscorrelationRegressor
from bioslds.batch import hyper_score_ar

from tqdm.notebook import tqdm

from draft_helpers import (
    paper_style,
    load_snippets,
    make_multi_trajectory_plot,
    calculate_asymmetry_measures,
)

fig_path = os.path.join("..", "figs", "draft")

# %% [markdown]
# # Run BioWTA and autocorrelation on signals based on snippets of vowel sounds

# %% [markdown]
# ## Define the problem and the parameters for the learning algorithms

# %% [markdown]
# Using best parameters obtained from hyperoptimization runs.

# %%
n_signals = 100
n_samples = 200_000
dwell_times = 100
min_dwell = 50
normalize = True
seed = 3487
n_models = 2
n_features = 4
wta_params = {"rate": 0.006872, "temperature": 23.75966, "error_timescale": 6.590304}
xcorr_params = {"n_features": 33, "nsm_rate": 0.51e-4, "xcorr_rate": 1 / 5.318138}
metric = unordered_accuracy_score
good_score = 0.70
threshold_steps = 10_000

snippets = load_snippets("vowel", "ao")
dataset = RandomSnippetDataset(
    n_signals,
    n_samples,
    snippets,
    dwell_times=dwell_times,
    min_dwell=min_dwell,
    normalize=normalize,
    rng=seed,
)

# %%
result_biowta = hyper_score_ar(
    BioWTARegressor,
    dataset,
    metric,
    n_models=n_models,
    n_features=n_features,
    progress=tqdm,
    monitor=["r", "weights_", "prediction_"],
    **wta_params,
)

crt_scores = result_biowta[1].trial_scores
crt_median = np.median(crt_scores)
crt_quantile = np.quantile(crt_scores, 0.05)
crt_good = np.mean(crt_scores > good_score)
print(
    f"BioWTA on vowels score: median={crt_median:.4f}, "
    f"5%={crt_quantile:.4f}, "
    f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
)

# %%
make_multi_trajectory_plot(
    result_biowta[1],
    dataset,
    n_traces=25,
    sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
    highlight_idx=50,
    trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
    rug_kws={"alpha": 0.3},
)

# %%
crt_frac_good = np.mean(result_biowta[1].trial_scores > good_score)
print(
    f"Percentage of runs with BioWTA accuracies over {int(good_score * 100)}%: "
    f"{int(crt_frac_good * 100)}%."
)

crt_frac_fast = np.mean(
    np.asarray(result_biowta[1].convergence_times) <= threshold_steps
)
print(
    f"Percentage of runs with BioWTA convergence times under {threshold_steps}: "
    f"{int(crt_frac_fast * 100)}%."
)

# %%
result_xcorr = hyper_score_ar(
    CrosscorrelationRegressor,
    dataset,
    metric,
    n_models=n_models,
    **xcorr_params,
    progress=tqdm,
    monitor=["r", "nsm.weights_", "xcorr.coef_"],
)

crt_scores = result_xcorr[1].trial_scores
crt_median = np.median(crt_scores)
crt_quantile = np.quantile(crt_scores, 0.05)
crt_good = np.mean(crt_scores > good_score)
print(
    f"xcorr on vowels score: median={crt_median:.4f}, "
    f"5%={crt_quantile:.4f}, "
    f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
)

# %%
make_multi_trajectory_plot(
    result_xcorr[1],
    dataset,
    n_traces=25,
    sliding_kws={"window_size": 5000, "overlap_fraction": 0.8},
    highlight_idx=50,
    trace_kws={"alpha": 0.85, "lw": 0.75, "color": "gray"},
    rug_kws={"alpha": 0.3},
)

# %%
crt_frac_good = np.mean(result_xcorr[1].trial_scores > good_score)
print(
    f"Percentage of runs with xcorr accuracies over {int(good_score * 100)}%: "
    f"{int(crt_frac_good * 100)}%."
)

crt_frac_fast = np.mean(
    np.asarray(result_xcorr[1].convergence_times) <= threshold_steps
)
print(
    f"Percentage of runs with xcorr convergence times under {threshold_steps}: "
    f"{int(crt_frac_fast * 100)}%."
)

# %%
calculate_asymmetry_measures(result_biowta[1], dataset)

# %%
with FigureManager() as (_, ax):
    crt_asymmetry = result_biowta[1].asymmetry[0]
    crt_confusion = crt_asymmetry.confusion
    crt_cmap = make_gradient_cmap("white_to_C1", "w", "C1")
    sns.heatmap(crt_confusion, cmap=crt_cmap, vmin=0, vmax=0.5, ax=ax)
    ax.set_aspect(1)

# %%
with FigureManager() as (_, ax):
    crt_diffs = []
    for crt_asymmetry in result_biowta[1].asymmetry:
        crt_confusion = crt_asymmetry.confusion_ordered
        crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
        crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
        crt_diffs.append(crt_diff)
    
    ax.hist(crt_diffs)

# %%
