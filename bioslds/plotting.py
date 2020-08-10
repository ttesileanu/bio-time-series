""" Define various classes and functions used for plotting. """

import copy

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib import patches
from typing import Optional, Sequence, Tuple


class FigureManager(object):
    """ A context manager to prettify Matplotlib figures. """

    def __init__(
        self,
        *args,
        offset: float = 10,
        despine_kws: Optional[dict] = None,
        do_despine: bool = True,
        **kwargs
    ):
        """ Initialize the manager.

        This is used like this:

        with FigureManager() as (fig, ax):
            ax.plot(...)

        This creates a figure, adjusting its size if multiple panels are
        requested so that each panel is the same size as the default figure
        size (unless the `figsize` keyword is explicitly used). By default,
        `tight_layout` is set to `True`, unless either this keyword is present
        in the arguments to `__init__`, or `constrained_layout` is set to true.

        When the context manager is destroyed, Seaborn's `despine` is run on all
        the axis that were generated.

        Parameters
        ----------
        offset
            Offset to use in `sns.despine`.
        despine_kws
            Additional options to pass to `sns.despine`.
        do_despine
            Whether to run `despine` or not.
        Other parameters are passed directly to `plt.subplots`.
        """
        self.offset = offset
        self.despine_kws = despine_kws
        self.do_despine = do_despine

        if len(args) == 2 and "figsize" not in kwargs:
            default_figsize = plt.rcParams["figure.figsize"]
            kwargs["figsize"] = np.flip(args) * default_figsize

        if (
            "constrained_layout" not in kwargs
            or not kwargs["constrained_layout"]
        ):
            kwargs.setdefault("tight_layout", True)

        self.fig, self.ax = plt.subplots(*args, **kwargs)

    def __enter__(self):
        return self.fig, self.ax

    def _despine(self, ax):
        """ Call despine on all the given axes. """
        if hasattr(ax, "__len__"):
            for sub_ax in ax:
                self._despine(sub_ax)
        else:
            if self.despine_kws is not None:
                kws = dict(self.despine_kws)
            else:
                kws = {}
            if "offset" not in kws:
                kws["offset"] = self.offset
            sns.despine(ax=ax, **kws)

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if self.do_despine:
            self._despine(self.ax)
        # self.fig.tight_layout()


def show_latent(
    seq: Sequence,
    ax: plt.Axes = None,
    bounds: Optional[Tuple] = None,
    colors: Sequence = None,
    show_bars: bool = True,
    bar_width: float = 0.1,
    bar_location: str = "top",
    show_vlines: bool = True,
    vline_kws: Optional[dict] = None,
):
    """ Display a bar plot showing how the latent state changes with time.

    The bars are drawn either above or below the current extents of the plot, expanding
    the y limits appropriately.

    Parameters
    ----------
    seq
        Sequence indicating the latent state.
    ax
        Axes in which to draw the bars. If not given, `plt.gca()` is used.
    bounds
        If not `None`, this should be a tuple `(t0, t1)` such that the latent state is
        shown only for time points `t >= t0` and `t < t1`. If this is `None`, the
        extents are inferred from the current axis limits.
    colors
        Sequence of colors to use for the identities. By default Matplotlib's default
        color cycle is used.
    show_bars
        If `True`, colored bars are drawn to indicate the current state.
    bar_width
        Width of the bars, given as a fraction of the vertical extent of the plot. Note
        that this is calculated at the moment the function is called.
    bar_location
        Location of the bars. This can be "top" or "bottom".
    show_vlines
        If `True`, vertical lines are drawn to show transition points.
    vline_kws
        Keywords to pass to `axvline`.
    """
    # handle trivial case
    if len(seq) == 0:
        return

    # handle defaults
    if ax is None:
        ax = plt.gca()
    if colors is None:
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]
    if bounds is None:
        bounds = ax.get_xlim()

    # find transition points
    transitions = np.diff(seq).nonzero()[0] + 1

    # find the first transition in the given range
    visible_mask = (transitions >= bounds[0])
    if np.any(visible_mask):
        idx0 = visible_mask.argmax()
    else:
        idx0 = None

    if show_vlines and idx0 is not None:
        # set up the vline parameters
        if vline_kws is not None:
            crt_vline_kws = copy.copy(vline_kws)
        else:
            crt_vline_kws = {}
        crt_vline_kws.setdefault("ls", ":")
        crt_vline_kws.setdefault("lw", 0.5)
        crt_vline_kws.setdefault("c", "k")

        for transition in transitions[idx0:]:
            if transition >= bounds[1]:
                break
            ax.axvline(transition, **crt_vline_kws)

    if show_bars:
        # find how big the bar is in data coordinates...
        yl = ax.get_ylim()
        yrange = yl[1] - yl[0]
        bar_width_data = yrange * bar_width

        # ...and where to place it
        if bar_location == "top":
            bar_y = yl[1]
            # adjust limits
            yl = (yl[0], bar_y + bar_width_data)
        elif bar_location == "bottom":
            bar_y = yl[0] - bar_width_data
            # adjust limits
            yl = (bar_y, yl[1])
        else:
            raise ValueError("Unknown bar location option.")

        # start drawing!
        x0 = max(bounds[0], 0)
        if idx0 is not None:
            next_idx = idx0
        else:
            next_idx = int(x0)
        while x0 < bounds[1] and int(x0) < len(seq):
            crt_id = seq[int(x0)]
            x1 = (
                transitions[next_idx]
                if next_idx < len(transitions)
                else min(bounds[1], len(seq))
            )
            x1 = min(x1, bounds[1])
            if x1 > x0:
                patch = patches.Rectangle(
                    (x0, bar_y),
                    x1 - x0,
                    bar_width_data,
                    edgecolor="none",
                    facecolor=colors[crt_id % len(colors)],
                )
                ax.add_patch(patch)

            next_idx += 1
            x0 = x1

        # adjust limits
        ax.set_ylim(*yl)
