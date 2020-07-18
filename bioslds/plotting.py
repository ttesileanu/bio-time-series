""" Define various classes and functions used for plotting. """

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Optional


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
