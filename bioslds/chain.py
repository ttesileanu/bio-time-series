""" Define a mixin for chaining transformers. """

import numpy as np

from types import SimpleNamespace
from typing import Optional, Sequence, Tuple, Callable, Union

from bioslds.monitor import AttributeMonitor


class ChainMixin(object):
    """ A mixin for chaining together a set of transformers.

    This implements a `transform` method that goes through the transformers whose names
    must be in `self._transformer_names`, processing the data from input to output. It
    handles proper monitoring and progress updating.
    """

    def __init__(self):
        """ Initialize a chain of transformers.

        This creates `self._transformers` based on `self._transformer_names`, then sets
        up te number of input and output dimensions, `self.n_features` and
        `self.n_components`.

        Thus this constructor should be called *after* the transformers are initialized
        in the base-class constructor.
        """
        if not hasattr(self, "_transformer_names"):
            self._transformer_names = []

        self._transformers = [getattr(self, name) for name in self._transformer_names]

        self.n_features = self._transformers[0].n_features
        self.n_components = self._transformers[-1].n_components

    def transform(
        self,
        X: Optional = None,
        y: Optional = None,
        n_samples: Optional[int] = None,
        progress: Optional[Callable] = None,
        monitor: Union[None, AttributeMonitor, Sequence] = None,
        return_history: bool = False,
        chunk_hint: int = 10000,
    ) -> Union[np.ndarray, Tuple[np.ndarray, SimpleNamespace]]:
        """ Feed a set of samples through the chain of transformers.

        The arguments `X`, `y`, and `n_samples` are simply passed to the first
        transformer in `self._transformers`.

        Parameters
        ----------
        X
            The values of the input or predictor variables. Shape `(n_samples,
            n_features)`. Whether this needs to be provided or not, and what exactly it
            means depends on the first transformer.
        y
            The values of the dependent variable. Shape `(n_samples,)`. Whether this
            needs to be provided or not, and what exactly it means depends on the first
            transformer.
        n_samples
            If `X` and `y` are not provided, or if they are callables, `n_samples` sets
            the number of samples to transform. How exactly this value is used is up to
            the firs transformer.
        progress
            Progress function that can be used either as a wrapper or manually, like
            `tqdm.tqdm`. Specifically, this needs to support wrapping an iterable, so
            you can write, e.g., `for x in progress(X): ...`; and it needs to support
            calling with a `total` argument, an `update`, and a `close` method, e.g.:
                pbar = progress(total=100)
                for i in range(100):
                    pbar.update(1)  # note that arg is step (=1), not i!
                pbar.close()
        monitor
            This can be an object for monitoring the evolution of the parameters during
            learning (e.g., an instance of `AttributeMonitor`), or a sequence of
            attribute names indicating the parameters to be tracked. If the latter, the
            function generates an `AttributeMonitor`, and `return_history` is forced to
            be true so that its `history_` attribute can be accessed. Parameter values
            are stored before their updates.
        return_history
            If true, return the monitor's `history_` namespace as a second return value.
            If `monitor` is `None`, returns an empty namespace.
        chunk_hint
            A hint about how to chunk the learning. This may or may not be used. If it
            is, the progress function will only be called once per chunk.

        If `return_history` is not false, returns just the output from the last
        transformer. If it is true, returns a tuple `(r, history)` where `r` is the
        output from above, and `history` is the `history_` attribute of the monitor that
        was used.
        """
        # prepare monitor
        n = len(X) if X is not None else len(y) if y is not None else n_samples
        if monitor is not None:
            if hasattr(monitor, "__len__") and not hasattr(monitor, "setup"):
                monitor = AttributeMonitor(monitor)
                # sequence implies return_history
                return_history = True
            monitor.setup(n)
        else:
            return_history = False

        # prepare progress bar
        if progress is not None:
            pbar = progress(total=n)
        else:
            pbar = None

        # split the data into chunks, feed each through chain
        out = np.zeros((n, self.n_components))
        for chunk_start in range(0, n, chunk_hint):
            crt_range = slice(chunk_start, chunk_start + chunk_hint)
            crt_out = out[crt_range]

            # find out data for first transformer
            crt_kwargs = {}
            if X is not None:
                crt_kwargs["X"] = X[crt_range]
            if y is not None:
                crt_kwargs["y"] = y[crt_range]
            if n_samples is not None:
                crt_kwargs["n_samples"] = len(crt_out)

            res = None
            to_track = SimpleNamespace()
            for name, trafo in zip(self._transformer_names, self._transformers):
                if monitor is not None:
                    trafo_prefix = name + "."
                    trafo_n = len(trafo_prefix)
                    crt_names = [
                        _[trafo_n:] for _ in monitor.names if _.startswith(trafo_prefix)
                    ]
                    if len(crt_names) > 0:
                        crt_kwargs["monitor"] = AttributeMonitor(crt_names)

                res = trafo.transform(**crt_kwargs, chunk_hint=chunk_hint)

                if "monitor" in crt_kwargs:
                    setattr(to_track, name, crt_kwargs["monitor"].history_)

                crt_kwargs = {"X": res}

            crt_out[:] = res

            if monitor is not None:
                monitor.record_batch(to_track)

            if pbar is not None:
                pbar.update(len(crt_out))

        if pbar is not None:
            pbar.close()

        if not return_history:
            return out
        else:
            return out, monitor.history_
