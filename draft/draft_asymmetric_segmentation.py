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


# %% [markdown]
# # Analyze asymmetry of segmentation errors

# %% [markdown]
# We first generate `n_datasets` pairs of AR(3) processes, similar to `draft_two_ar3.ipynb`. For each of these, we generate `n_signals` random datasets that use the same AR(3) processes but different random realizations of the semi-Markov chain of latent states. We then run `BioWTA` on all of these and measure the asymmetry in the confusion matrix (see `calculate_asymmetry_measures`). This should help us get an idea for what property of a process makes it harder to identify correctly.

# %%
n_datasets = 100
n_signals = 50
n_samples = 150_000
n_models = 2
n_features = 3
dwell_times = 100
min_dwell = 50
normalize = True
max_pole_radius = 0.95
seed = 121
metric = unordered_accuracy_score
wta_params = {
    "rate": 0.001992,
    "trans_mat": 1 - 1 / 7.794633,
    "temperature": 1.036228,
    "error_timescale": 1.000000,
}
# wta_params = {
#     "rate": 0.005460,
#     "trans_mat": 1 - 1 / 6.792138,
#     "temperature": 0.961581,
#     "error_timescale": 1.000000,
# }
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
        rng=seed + i,
    )
    for i, _ in enumerate(arma_pairs)
]

# %%
results_biowta = []
for i, dataset in enumerate(tqdm(datasets, desc="dataset")):
    crt_res = hyper_score_ar(
        make_bio_wta_with_stable_initial,
        dataset,
        metric,
        n_models=n_models,
        n_features=n_features,
        progress=tqdm,
        monitor=["r"],
        rng=i,
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
crt_idx = 14
make_multi_trajectory_plot(
    results_biowta[crt_idx][1],
    datasets[crt_idx],
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
confusion_bias_full_range = []
max_pole_radius = np.zeros((n_datasets, n_models))
median_pole_radius = np.zeros((n_datasets, n_models))
min_pole_radius = np.zeros((n_datasets, n_models))

max_abs_angle = np.zeros((n_datasets, n_models))
min_abs_angle = np.zeros((n_datasets, n_models))

neg_pole = np.zeros((n_datasets, n_models))

all_poles = np.zeros((n_datasets, n_models, n_features), dtype=complex)

all_diffs = []


def order_poles(poles):
    return poles[np.argsort(np.imag(poles))]


for i, crt_res in enumerate(results_biowta):
    crt_diffs = []
    for crt_asymmetry in crt_res[1].asymmetry:
        crt_confusion = crt_asymmetry.confusion_ordered
        crt_confusion_normalized = crt_confusion / (crt_confusion.sum(axis=1)[:, None])
        crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
        crt_diffs.append(crt_diff)

    all_diffs.append(crt_diffs)
    confusion_bias.append(np.median(crt_diffs))
    confusion_bias_range.append(np.quantile(crt_diffs, [0.25, 0.75]))
    confusion_bias_full_range.append(np.quantile(crt_diffs, [0.00, 1.00]))

    crt_armas = arma_pairs[i]
    crt_poles = [order_poles(_.calculate_poles()) for _ in crt_armas]

    crt_pole_radii = [[np.linalg.norm(crt_pole) for crt_pole in _] for _ in crt_poles]
    crt_abs_angles = [[np.abs(np.angle(crt_pole)) for crt_pole in _] for _ in crt_poles]

    all_poles[i, :, :] = crt_poles

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
confusion_bias_full_range = np.array(confusion_bias_full_range)


# %%
def get_stds(sig):
    return (np.std(sig.y[sig.usage_seq == 0]), np.std(sig.y[sig.usage_seq == 1]))


# %%
stds = np.zeros((n_datasets, n_models))
pooled_std_diffs = []
for i, dataset in enumerate(tqdm(datasets)):
    crt_stds = np.zeros((n_signals, 2))
    for j, sig in enumerate(dataset):
        crt_stds[j, :] = get_stds(sig)
        
    pooled_std_diffs.extend(crt_stds[:, 0] - crt_stds[:, 1])
    
    stds[i, :] = np.median(crt_stds, axis=0)

pooled_std_diffs = np.asarray(pooled_std_diffs)

# %%
with FigureManager() as (_, ax):
    crt_diffs = []

    all_scores = np.asarray([_[0] for _ in results_biowta])
    low_score = 0.0
    high_score = 1.0
    # mask = (all_scores > low_score) & (all_scores < high_score)
    mask = np.ones(len(all_scores), dtype=bool)
    print(
        f"Focusing on scores between {low_score} and {high_score} "
        f"({100 * np.mean(mask):.1f}% of the samples)."
    )

    for i, crt_res in enumerate(results_biowta):
        if not mask[i]:
            continue

        for crt_asymmetry in crt_res[1].asymmetry:
            crt_confusion = crt_asymmetry.confusion_ordered
            crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
            crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]
            crt_diffs.append(crt_diff)

    ax.hist(crt_diffs, 100)

# %%
measures = {
    "max pole radii": max_pole_radius[:, 0] - max_pole_radius[:, 1],
    "median pole radii": median_pole_radius[:, 0] - median_pole_radius[:, 1],
    "min pole radii": min_pole_radius[:, 0] - min_pole_radius[:, 1],
    "max abs angle": max_abs_angle[:, 0] - max_abs_angle[:, 1],
    "min abs angle": min_abs_angle[:, 0] - min_abs_angle[:, 1],
    "value of real pole": neg_pole[:, 0] - neg_pole[:, 1],
    "std": stds[:, 0] - stds[:, 1],
}

with FigureManager((len(measures) + 1) // 2, 2) as (_, axs):
    axs = axs.ravel()
    crt_max = 0
    for i, measure in enumerate(measures):
        axs[i].errorbar(
            measures[measure][mask],
            confusion_bias[mask],
            yerr=[
                (confusion_bias - confusion_bias_range[:, 0])[mask],
                (confusion_bias_range[:, 1] - confusion_bias)[mask],
            ],
            ecolor="gray",
            ls="none",
        )
        # axs[i].scatter(measures[measure], confusion_bias)
        sns.regplot(
            x=measures[measure][mask],
            y=confusion_bias[mask],
            color="C0",
            truncate=False,
            ax=axs[i],
        )
        axs[i].set_xlabel("difference in " + measure)
        
        crt_max = max(crt_max, max(np.abs(axs[i].get_ylim())))

    for ax in axs:
        ax.set_ylim(-crt_max, crt_max)

# %%
import statsmodels.formula.api as smf
import pandas as pd

df = pd.DataFrame({_.replace(" ", "_"): measures[_][mask] for _ in measures})
df["confusion_bias"] = confusion_bias[mask]
df["pole_real_diff"] = (np.real(all_poles[:, 0, 1]) - np.real(all_poles[:, 1, 1]))[mask]
df["pole_cplx_diff"] = (all_poles[:, 0, 2] - all_poles[:, 1, 2])[mask]
df["pole_cplx_diff_re"] = np.real(df["pole_cplx_diff"])
df["pole_cplx_diff_im"] = np.imag(df["pole_cplx_diff"])

# model = smf.ols(formula="confusion_bias ~ min_pole_radii + max_abs_angle + value_of_real_pole", data=df)
model = smf.ols(
    formula="confusion_bias ~ min_pole_radii + max_abs_angle + std", data=df
)
# model = smf.ols(formula="confusion_bias ~ pole_real_diff + pole_cplx_diff_re + pole_cplx_diff_im", data=df)
fit = model.fit()

# %%
fit.summary()

# %%
pooled_scores = []
pooled_asymmetries = []

for i, crt_res in enumerate(results_biowta):
    pooled_scores.extend(n_signals * [all_scores[i]])
    for crt_asymmetry in crt_res[1].asymmetry:
        crt_confusion = crt_asymmetry.confusion_ordered
        crt_confusion_normalized = crt_confusion / crt_confusion.sum(axis=1)
        crt_diff = crt_confusion_normalized[0, 1] - crt_confusion_normalized[1, 0]

        pooled_asymmetries.append(crt_diff)

pooled_scores = np.asarray(pooled_scores)
pooled_asymmetries = np.asarray(pooled_asymmetries)

with FigureManager(1, 2) as (_, (ax1, ax2)):
    ax1.scatter(pooled_asymmetries, pooled_scores, c=pooled_scores, alpha=0.2)
    ax1.set_xlabel("asymmetry score")
    ax1.set_ylabel("segmentation score")

    ax1.axvline(0, c="k", ls=":", alpha=0.5, lw=2)

    ax2.scatter(pooled_asymmetries, pooled_std_diffs, c=pooled_scores, alpha=0.2)
    ax2.set_xlabel("asymmetry score")
    ax2.set_ylabel("difference in standard deviation")

    ax2.axvline(0, c="k", ls=":", alpha=0.5, lw=2)

# %% [markdown]
# ## Make figures for draft

# %%
high_bias_mask = np.all(confusion_bias_full_range < 0, axis=1) | np.all(
    confusion_bias_full_range > 0, axis=1
)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        1,
        2,
        gridspec_kw={"width_ratios": (12, 2)},
        figsize=(5.76, 2),
        despine_kws={"offset": 5},
    ) as (fig, (ax, axh)):
        crt_order = np.argsort(confusion_bias)

        ax.axhline(0, ls=":", c="gray", alpha=0.5)

        crt_err_h = ax.errorbar(
            np.arange(len(confusion_bias)),
            confusion_bias[crt_order],
            yerr=(
                confusion_bias[crt_order] - confusion_bias_full_range[crt_order, 0],
                confusion_bias_full_range[crt_order, 1] - confusion_bias[crt_order],
            ),
            ls="none",
            ecolor="gray",
            alpha=0.5,
            label="full range (100 trials)",
            clip_on=False,
        )

        crt_noskew_h = ax.scatter(
            np.arange(len(confusion_bias))[~high_bias_mask[crt_order]],
            confusion_bias[crt_order[~high_bias_mask[crt_order]]],
            s=5,
            c="C0",
            label="median",
            clip_on=False,
        )
        crt_skew_h = ax.scatter(
            np.arange(len(confusion_bias))[high_bias_mask[crt_order]],
            confusion_bias[crt_order[high_bias_mask[crt_order]]],
            s=5,
            c="C1",
            label="skewed",
            clip_on=False,
        )

        # crt_yl = max(ax.get_ylim())
        crt_yl = 0.5
        ax.set_ylim(-crt_yl, crt_yl)

        ax.set_xlim(0, len(confusion_bias))

        # ax.legend(frameon=False, loc="upper center", fontsize=7, ncol=2)
        ax.legend(
            [crt_noskew_h, crt_err_h, crt_skew_h],
            ["median", f"full range ({n_signals} trials)", "skewed"],
            frameon=False,
            loc="upper center",
            fontsize=7,
            ncol=2,
        )

        ax.set_xlabel("dataset index (ordered by asymmetry measure)")
        ax.set_ylabel("asymmetry measure")

        axh.axhline(0, ls=":", c="gray", alpha=0.5)
        sns.kdeplot(
            y=confusion_bias, shade=True, alpha=0.25, lw=0.75, color="gray", ax=axh
        )
        axh.set_xlabel("pdf")

        axh.set_yticks([])
        axh.set_ylim(-crt_yl, crt_yl)

        # tmp_sigma = np.std(confusion_bias)
        tmp_sigma = 1.4826 * np.median(np.abs(confusion_bias))

        tmp_y = np.linspace(-crt_yl, crt_yl, 50)
        tmp_x = (
            1
            / np.sqrt(2 * np.pi * tmp_sigma ** 2)
            * np.exp(-0.5 * (tmp_y / tmp_sigma) ** 2)
        )
        # XXX found scaling factor by hand to match data pdf
        axh.plot(0.68 * tmp_x, tmp_y, c="C0", alpha=0.75, label="normal")
        axh.legend(frameon=False, fontsize=6)

    sns.despine(ax=axh, left=True)

    fig.savefig(os.path.join(fig_path, "asymmetry_outcomes.pdf"))

# %%
crt_cols = {
    "pole_cplx_diff_re": "$\\Delta\, $Re(complex pole)",
    "pole_cplx_diff_im": "$\\Delta\, $Im(complex pole)",
    "pole_real_diff": "$\\Delta\, $(real pole)",
    "std": "$\\Delta\, $(standard deviation)",
    "max_pole_radii": "$\\Delta\, \\max\, \\| $pole$\\|$",
    "min_pole_radii": "$\\Delta\, \\min\, \\| $pole$\\|$",
}
with plt.style.context(paper_style):
    with FigureManager(
        (len(measures) + 1) // 3, 3, figsize=(5.76, 3.5), despine_kws={"offset": 5}
    ) as (fig, axs):
        axs = axs.ravel()
        crt_max = 0
        for i, crt_col in enumerate(crt_cols):
            axs[i].errorbar(
                df[crt_col].values,
                confusion_bias,
                yerr=[
                    confusion_bias - confusion_bias_range[:, 0],
                    confusion_bias_range[:, 1] - confusion_bias,
                ],
                ecolor="gray",
                ls="none",
                elinewidth=1.0,
                alpha=0.8,
                barsabove=False,
            )

            sns.regplot(
                x=df[crt_col].values,
                y=confusion_bias,
                color="C0",
                truncate=False,
                scatter_kws={"s": 8, "alpha": 1.0},
                line_kws={"lw": 1.0, "color": "gray"},
                ax=axs[i],
            )
            axs[i].set_xlabel(crt_cols[crt_col])

            crt_max = max(crt_max, max(np.abs(axs[i].get_ylim())))

        for ax in axs:
            ax.set_ylim(-crt_max, crt_max)
            
    fig.savefig(os.path.join(fig_path, "asymmetry_explanation_attempt.pdf"))

# %%
model = smf.ols(formula="confusion_bias ~ " + " + ".join(crt_cols) + " + 0", data=df)
fit = model.fit()
fit.summary()

# %%
with FigureManager() as (_, ax):
    sns.regplot(x=fit.predict(df), y=confusion_bias, ax=ax, truncate=False)
    ax.set_xlabel("OLS prediction")
    ax.set_ylabel("actual asymmetry")
    
    crt_xl = max(ax.get_xlim())
    crt_yl = max(ax.get_ylim())
    crt_l = max(crt_xl, crt_yl)
    
#     ax.set_xlim(-crt_l, crt_l)
#     ax.set_ylim(-crt_l, crt_l)
#     ax.set_aspect(1)

# %%
