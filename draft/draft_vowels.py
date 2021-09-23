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

from types import SimpleNamespace

from bioslds.arma import make_random_arma
from bioslds.dataset import RandomSnippetDataset
from bioslds.plotting import FigureManager, make_gradient_cmap, show_latent
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
# ## Useful functions

# %%
def make_simple_trajectory_plot(*args, figsize=(5.76, 1.4), **kwargs):
    kwargs["n_traces"] = 25
    kwargs["sliding_kws"] = {"window_size": 20000, "overlap_fraction": 0.8}
    kwargs["trace_kws"] = {"alpha": 0.25, "lw": 0.75, "color": "gray"}
    kwargs["rug_kws"] = {"alpha": 0.3}
    kwargs["show_time_pdf"] = False
    with plt.style.context(paper_style):
        fig = plt.figure(figsize=figsize)
        axs = np.empty((2, 2), dtype=object)
        axs[0, 0] = fig.add_axes([0.09, 0.18, 0.80, 0.70])
        axs[0, 1] = fig.add_axes([0.90, 0.18, 0.10, 0.70])
        
        kwargs["axs"] = axs
        make_multi_trajectory_plot(*args, **kwargs)

        axs[0, 0].set_yticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    
    return fig, axs


# %% [markdown]
# ## Define the problem and the parameters for the learning algorithms

# %% [markdown]
# Using best parameters obtained from hyperoptimization runs.

# %%
def make_bio_wta_with_stable_initial(*args, **kwargs) -> BioWTARegressor:
    """ Call the BioWTARegressor constructor, ensuring that the initial coefficients are
    chosen to correspond to stable AR processes.
    """
    weights = [
        make_random_arma(kwargs["n_features"], 0, rng=kwargs["rng"]).a
        for _ in range(kwargs["n_models"])
    ]
    return BioWTARegressor(*args, weights=weights, **kwargs)


# %%
n_signals = 100
n_samples = 200_000
dwell_times = 1500
min_dwell = 800
normalize = True
seed = 2398
n_models = 2
n_features = 4
wta_params = {
    "rate": 0.01426,
    "temperature": 17.30638,
    "error_timescale": 85.761315,
    "trans_mat": 1 - 1 / 4.209253,
}
xcorr_params = {
    "n_features": 4,
    "nsm_rate": 0.000079,
    "xcorr_rate": 1 / 39.252925,
    "fit_kws": {"step": 288},
}
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
with FigureManager() as (_, ax):
    crt_sig = dataset[0]
    ax.plot(crt_sig.y[:10000])
    show_latent(crt_sig.usage_seq, ax=ax)

# %%
result_biowta = hyper_score_ar(
    # BioWTARegressor,
    make_bio_wta_with_stable_initial,
    dataset,
    metric,
    n_models=n_models,
    n_features=n_features,
    progress=tqdm,
    monitor=["r", "weights_", "prediction_"],
    rng=137,
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
    sliding_kws={"window_size": 20000, "overlap_fraction": 0.8},
    highlight_idx=50,
    trace_kws={"alpha": 0.25, "lw": 0.75, "color": "gray"},
    rug_kws={"alpha": 0.3},
)

# %%
make_simple_trajectory_plot(result_biowta[1], dataset, highlight_idx=50)

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
    rng=900,
    **xcorr_params,
    progress=tqdm,
    monitor=["r", "xcorr.coef_"],
    # monitor=["r"],
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
    sliding_kws={"window_size": 20000, "overlap_fraction": 0.8},
    highlight_idx=50,
    trace_kws={"alpha": 0.25, "lw": 0.75, "color": "gray"},
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
calculate_asymmetry_measures(result_xcorr[1], dataset)

# %%
with FigureManager(1, 2) as (_, (ax1, ax2)):
    crt_cmap = make_gradient_cmap("white_to_C1", "w", "C1")
    
    crt_asymmetry = result_biowta[1].asymmetry[0]
    crt_confusion = crt_asymmetry.confusion_ordered
    sns.heatmap(crt_confusion, cmap=crt_cmap, vmin=0, vmax=0.5, ax=ax1)
    ax1.set_aspect(1)
    ax1.set_title("BioWTA")
    
    crt_asymmetry = result_xcorr[1].asymmetry[0]
    crt_confusion = crt_asymmetry.confusion_ordered
    sns.heatmap(crt_confusion, cmap=crt_cmap, vmin=0, vmax=0.5, ax=ax2)
    ax2.set_aspect(1)
    ax2.set_title("autocorrelation")

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
with FigureManager() as (_, ax):
    crt_sig = dataset[0]
    crt_end_range = (20000, 10000)
    crt_range = slice(
        len(crt_sig.y) - crt_end_range[0], len(crt_sig.y) - crt_end_range[1]
    )
    ax.plot(np.arange(crt_range.start, crt_range.stop), crt_sig.y[crt_range])
    show_latent(crt_sig.usage_seq, bar_location="bottom", ax=ax)

    crt_inferred = np.argmax(result_xcorr[1].history[0].r, axis=1)
    show_latent(crt_inferred, show_vlines=False, ax=ax)

# %%
crt_vowel0_idx = 186500
crt_vowel1_idx = 182000
with FigureManager() as (_, ax):
    for i in range(100):
        ax.plot(
            result_xcorr[1].history[0].xcorr.coef_[crt_vowel0_idx + i],
            c="C0",
            alpha=0.05,
        )
        ax.plot(
            result_xcorr[1].history[0].xcorr.coef_[crt_vowel1_idx + i],
            c="C1",
            alpha=0.05,
        )

    ax.plot([], [], c="C0", label="vowel 0")
    ax.plot([], [], c="C1", label="vowel 1")
    ax.legend(frameon=False)

    ax.set_xlabel(f"lag (multiples of {xcorr_params['fit_kws']['step']})")
    ax.set_ylabel("autocorrelation")
    
    ax.set_xlim(0, 6)

# %%
with FigureManager(figsize=(10, 4)) as (_, ax):
    crt_sig = dataset[0]
    ax.specgram(crt_sig.y[:40000], NFFT=128, noverlap=0, Fs=1)
    show_latent(crt_sig.usage_seq, bar_location="bottom", show_vlines=False, ax=ax)


# %%
def autocorr(x):
    result = np.correlate(x, x, mode="full")
    acorr = result[result.size // 2 :]
    return acorr / acorr[0]


# %%
from bioslds.utils import rle_encode

# %%
crt_sig = dataset[0]
crt_rle = rle_encode(crt_sig.usage_seq)
acorrs = ([], [])

i = 0
for crt_id, crt_len in tqdm(crt_rle):
    crt_snippet = crt_sig.y[i : i + crt_len]
    crt_acorr = autocorr(crt_snippet)
    
    acorrs[crt_id].append(crt_acorr)

    i += crt_len

# %%
acorr_max_len = 700
with FigureManager() as (_, ax):
    for i, crt_acorrs in enumerate(acorrs):
        for crt_acorr in crt_acorrs:
            ax.plot(
                crt_acorr[:acorr_max_len],
                c=f"C{i}",
                alpha=0.05,
            )

    ax.plot([], [], c="C0", label="vowel 0")
    ax.plot([], [], c="C1", label="vowel 1")
    ax.legend(frameon=False)

    ax.set_xlabel("lag")
    ax.set_ylabel("autocorrelation")
    
    ax.set_xlim(0, 100)

# %% [markdown]
# # Test on other vowel pairs

# %%
others = SimpleNamespace(vowels="aeiou")
others.pairs = [
    others.vowels[_] + others.vowels[__]
    for _ in range(len(others.vowels))
    for __ in range(_ + 1, len(others.vowels))
]

others.snippets = []
others.datasets = []
for pair in others.pairs:
    crt_snippets = load_snippets("vowel", pair)
    others.snippets.append(crt_snippets)
    others.datasets.append(
        RandomSnippetDataset(
            n_signals,
            n_samples,
            crt_snippets,
            dwell_times=dwell_times,
            min_dwell=min_dwell,
            normalize=normalize,
            rng=seed,
        )
    )

# %%
others.result_biowta = []
others.result_xcorr = []
for i, pair in enumerate(tqdm(others.pairs)):
    crt_dataset = others.datasets[i]
    crt_result_biowta = hyper_score_ar(
        make_bio_wta_with_stable_initial,
        crt_dataset,
        metric,
        n_models=n_models,
        n_features=n_features,
        progress=tqdm,
        monitor=["r"],
        rng=137,
        **wta_params,
    )

    crt_scores = crt_result_biowta[1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > good_score)
    print(
        f"BioWTA on vowels '{pair}' score: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
    )
    
    others.result_biowta.append(crt_result_biowta)
    
    crt_result_xcorr = hyper_score_ar(
        CrosscorrelationRegressor,
        crt_dataset,
        metric,
        n_models=n_models,
        rng=900,
        **xcorr_params,
        progress=tqdm,
        monitor=["r"],
    )

    crt_scores = crt_result_xcorr[1].trial_scores
    crt_median = np.median(crt_scores)
    crt_quantile = np.quantile(crt_scores, 0.05)
    crt_good = np.mean(crt_scores > good_score)
    print(
        f"xcorr on vowels '{pair}' score: median={crt_median:.4f}, "
        f"5%={crt_quantile:.4f}, "
        f"fraction>{int(100 * good_score)}%={crt_good:.4f}"
    )
    
    others.result_xcorr.append(crt_result_xcorr)

# %%
with plt.style.context(paper_style):
    with FigureManager(figsize=(5.76, 2.0), despine_kws={"offset": 5}) as (fig, ax):
        others.biowta_scores = np.asarray([_[0] for _ in others.result_biowta])
        others.xcorr_scores = np.asarray([_[0] for _ in others.result_xcorr])

        others.biowta_ranges = np.quantile(
            [_[1].trial_scores for _ in others.result_biowta], [0.25, 0.75], axis=1
        ).T
        others.xcorr_ranges = np.quantile(
            [_[1].trial_scores for _ in others.result_xcorr], [0.25, 0.75], axis=1
        ).T

        crt_n = len(others.pairs)
        crt_offset = 0.15
        ax.bar(
            np.arange(crt_n) - crt_offset,
            others.biowta_scores,
            width=0.25,
            label="BioWTA",
        )
        ax.bar(
            np.arange(crt_n) + crt_offset,
            others.xcorr_scores,
            width=0.25,
            label="autocorr",
        )

        ax.errorbar(
            np.arange(crt_n) - crt_offset,
            others.biowta_scores,
            yerr=[
                others.biowta_scores - others.biowta_ranges[:, 0],
                others.biowta_ranges[:, 1] - others.biowta_scores,
            ],
            ecolor="k",
            ls="none",
            label="inter-quartile range",
        )
        ax.errorbar(
            np.arange(crt_n) + crt_offset,
            others.xcorr_scores,
            yerr=[
                others.xcorr_scores - others.xcorr_ranges[:, 0],
                others.xcorr_ranges[:, 1] - others.xcorr_scores,
            ],
            ecolor="k",
            ls="none",
        )

        ax.set_xticks(np.arange(crt_n))
        ax.set_xticklabels(others.pairs)
        
        ax.set_yticks(np.arange(0.5, 1.05, 0.1))

        ax.legend(frameon=False, loc="upper left", fontsize=7, ncol=2)

        ax.set_ylim(0.5, 1.1)

        ax.set_ylabel("segmentation score")
        ax.set_xlabel("vowel pair")

    fig.savefig(os.path.join(fig_path, "all_vowel_performance_both.pdf"))

# %%
crt_choice = [_ == "ao" for _ in others.pairs].index(True)
with plt.style.context(paper_style):
    fig = plt.figure(figsize=(5.76, 1.0))

    kwargs = {
        "n_traces": 25,
        "sliding_kws": {"window_size": 20000, "overlap_fraction": 0.8},
        "trace_kws": {"alpha": 0.25, "lw": 0.75, "color": "gray"},
        "rug_kws": {"alpha": 0.3},
        "show_time_pdf": False,
    }

    axs = [
        fig.add_axes([0.09, 0.30, 0.34, 0.60]),
        fig.add_axes([0.44, 0.30, 0.03, 0.60]),
        fig.add_axes([0.60, 0.30, 0.34, 0.60]),
        fig.add_axes([0.95, 0.30, 0.03, 0.60]),
    ]

    make_multi_trajectory_plot(
        others.result_biowta[crt_choice][1],
        others.datasets[crt_choice],
        highlight_idx=50,
        axs=np.asarray([axs[:2]]),
        **kwargs
    )

    make_multi_trajectory_plot(
        others.result_xcorr[crt_choice][1],
        others.datasets[crt_choice],
        highlight_idx=50,
        axs=np.asarray([axs[2:]]),
        **kwargs
    )

    axs[0].set_yticks([0.6, 0.8, 1.0])
    axs[2].set_yticks([0.6, 0.8, 1.0])

    axs[0].set_title("BioWTA, [ao]", y=0.8, fontsize=8)
    axs[2].set_title("autocorr, [ao]", y=0.8, fontsize=8)

    fig.savefig(os.path.join(fig_path, "rolling_accuracy_vowel_both.png"), dpi=600)

# %%
