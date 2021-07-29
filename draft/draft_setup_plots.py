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
from bioslds.arma_hsmm import sample_switching_models
from bioslds.plotting import FigureManager, show_latent

from bioslds import sources

from draft_helpers import paper_style

fig_path = os.path.join("..", "figs", "draft")

# %% [markdown]
# # Problem setup

# %%
n_samples = 600
orders = [(3, 0), (2, 0), (1, 0)]
max_pole_radius = 0.95
seed = 5

rng = np.random.default_rng(seed)
armas = [make_random_arma(*_, rng=rng, max_pole_radius=max_pole_radius) for _ in orders]

usage_seq = np.zeros(n_samples, dtype=int)
usage_seq[n_samples // 7 : 4 * n_samples // 7] = 1
usage_seq[4 * n_samples // 7 : 6 * n_samples // 7] = 2
usage_seq[6 * n_samples // 7 :] = 1
sig, x = sample_switching_models(
    armas, usage_seq=usage_seq, X=sources.GaussianNoise(rng), return_input=True
)

# %%
with plt.style.context(paper_style):
    with FigureManager(
        2, 1, figsize=(5.76, 2.5), sharex=True, gridspec_kw={"height_ratios": (3, 1.5)}
    ) as (fig, (ax0, ax1)):
        ax0.axhline(0, ls=":", c="gray", zorder=-1)
        ax0.plot(sig, "k", lw=0.75)
        yl = np.max(np.abs(sig))
        ax0.set_ylim(-yl, yl)

        show_latent(usage_seq, ax=ax0)

        # ax0.set_xlabel("time step")
        ax0.set_ylabel("signal")

        ax0.set_xlim(0, n_samples)
        ax0.set_yticks([])

        # start with ground truth, then add some noise, and finally smooth and normalize
        mock_z = np.asarray([usage_seq == [0, 1, 2][_] for _ in range(3)])

        rng = np.random.default_rng(seed)
        mock_z = np.clip(mock_z + 0.20 * rng.normal(size=mock_z.shape), 0, None,)

        # smooth
        crt_kernel = np.exp(-0.5 * np.linspace(-3, 3, 36) ** 2)
        mock_z = np.array([np.convolve(_, crt_kernel)[:n_samples] for _ in mock_z])

        # normalize
        mock_z = mock_z / np.sum(mock_z, axis=0)

        for i in range(3):
            ax1.plot(mock_z[i], f"C{i}", label=f"$z_{i}$")

        for i in range(3):
            ax1.fill_between(
                np.arange(n_samples), mock_z[i], color=f"C{i}", alpha=0.1,
            )
        ax1.legend(frameon=False, loc=(0.25, 0.27), fontsize=6, ncol=3)
        ax1.set_ylim(0, 1)

        ax1.set_xlabel("time step")
        ax1.set_ylabel("inference")

        show_latent(usage_seq, show_bars=False, ax=ax1)

fig.savefig(os.path.join(fig_path, "example_switching_signal_and_z_v2.pdf"))

# %%
