""" Define a biologically-plausible online learner using a similarity matching
objective.
"""

import numpy as np

from bioslds.monitor import AttributeMonitor

from typing import Optional, Sequence, Union, Callable


class NonRecurrent(object):
    """ Solve argmin ||X^T X - Y^T Y||^2_F, online, potentially with a
    non-negativity constraint.

    This uses the non-recurrent algorithm from Minden, Pehlevan, Chklovskii (2018).

    Attributes
    ----------
    n_components : int
        Number of components in the output.
    learning_rate : float
        Learning rate or learning rate schedule.
    tau : float
        Ratio between learning rates for feed-forward connection strengths and lateral
        connection strengths, respectively.
    scalings : np.ndarray
        Amounts by which the principal components are scaled at the optimum,
        corresponding to the `Lambda` matrix in Minden, Pehlevan, Chklovskii.
    non_negative : bool
        Whether the outputs are forced to be non-negative.
    whiten : bool
        Whether the outputs should be whitened.
    n_features_ : int
        Number of features in the training data.
    output_ : np.ndarray
        Current output values.
    weights_ : np.ndarray
        Current input-output connection strengths, shape
        `(n_components, n_features_)`. In the long term, this should converge to
        `Y @ X.T`, where `X` and `Y` are input and output matrices with one
        sample per column.
    lateral_ : np.ndarray
        Current lateral connection strengths, shape
        `(n_components, n_components)`. In the long term, this should converge to
        `isc @ Y @ Y.T @ isc` where `isc = np.inv(pc_scalings)`.
    n_samples_ : int
        Number of samples seen.
    """

    def __init__(
        self,
        n_features: Optional[int] = None,
        n_components: Optional[int] = None,
        weights: Optional[Sequence] = None,
        lateral: Optional[Sequence] = None,
        tau: float = 0.5,
        learning_rate: Union[float, Sequence, Callable[[float], float]] = 0.001,
        scalings: Optional[Sequence] = None,
        non_negative: bool = False,
        whiten: bool = False,
        rng: Optional[Union[int, np.random.Generator]] = None,
    ):
        """ Initialize the circuit.

        Parameters
        ----------
        n_features
            Number of dimensions in the input. If not provided, this is inferred from
            the shape of `weights`.
        n_components
            Number of dimensions in the output. This must be provided unless one of
            `weights` or `lateral` is given.
        weights
            Initial input-output connection strengths, should have shape `(n_components,
            n_features)`. Set to all 1 by default, unless `rng` is provided.
        lateral
            Initial lateral connection strengths, should have shape `(n_components,
            n_components)`. It will be automatically symmetrized and made
            positive-definite, if necessary. These connection strengths are set to the
            identity matrix if not explicitly provided, unless `rng` is used.
        tau
            Ratio between learning rates for feed-forward connection strengths and
            lateral connection strengths, respectively.
        learning_rate : float
            Learning rate or learning schedule for feed-forward weights.  If this is a
            sequence, the `i`th element is used as the learning rate at the `i`th step.
            The last element is used for any steps beyond the length of the sequence. If
            this is a callable, the learning rate is obtained by calling it with the
            current step number as an argument.
        scalings
            Amounts by which the principal components are scaled in the optimum
            solution. This corresponds to the diagonal elements of the `Lambda` matrix
            in Minden, Pehlevan, Chklovskii (2018), and is useful for breaking the
            degeneracy in the solution set. By default it is set to all 1, which implies
            *no* degeneracy breaking.
        non_negative
            Set to true to force the outputs to be non-negative.
        whiten
            Set to true to impose a whitening constraint on the output (as in the PSW
            problem from Minden, Pehlevan, Chklovskii (2018)).
        rng
            Random number generator or seed used to initialize the weights. If not
            provided, the initial values from above are used.
        """
        # copy over the trivial parameters
        self.tau = tau
        self.non_negative = non_negative
        self.whiten = whiten

        if callable(learning_rate) or not hasattr(learning_rate, "__len__"):
            self.learning_rate = learning_rate
        else:
            self.learning_rate = np.array(learning_rate)
        self._learning_rate_vector = None

        # infer input and output dimensions
        if weights is not None:
            self.n_components, self.n_features_ = np.shape(weights)
        else:
            if n_features is None:
                raise ValueError(
                    "Need either weights or n_features to specify the "
                    "dimension of the inputs."
                )
            self.n_features_ = n_features
            if lateral is not None:
                self.n_components = len(lateral)
            elif n_components is not None:
                self.n_components = n_components
            else:
                raise ValueError(
                    "Need either weights, lateral, or n_components to specify the "
                    "dimension of the outputs."
                )

        # check that all dimensions match
        if (
            n_features is not None
            and weights is not None
            and n_features != np.shape(weights)[1]
        ):
            raise ValueError("Weights shape does not match n_features.")

        # I don't know why but the type checker thinks weights is None here...
        # noinspection PyTypeChecker
        if (
            n_components is not None
            and weights is not None
            and n_components != np.shape(weights)[0]
        ):
            raise ValueError("Weights shape does not match n_components.")
        # noinspection PyTypeChecker
        if lateral is not None and weights is not None and len(lateral) != len(weights):
            raise ValueError("Weights and lateral shapes do not match.")
        if lateral is not None and len(lateral) != np.shape(lateral)[1]:
            raise ValueError("Lateral must be a square matrix.")

        # handle some defaults
        if scalings is not None:
            self.scalings = np.array(scalings)
        else:
            self.scalings = np.ones(self.n_components)

        # construct the state variables
        # initial value of output doesn't matter: gets replaced with every `fit`
        self.output_ = np.zeros(self.n_components)

        # use rng to initialize weights, if provided
        if rng is not None:
            rng = np.random.default_rng(rng)

        # initialize connection weights
        if weights is not None:
            self.weights_ = np.array(weights)
        elif rng is not None:
            self.weights_ = rng.normal(
                size=(self.n_components, self.n_features_)
            ) / np.sqrt(self.n_features_)
        else:
            self.weights_ = np.ones((self.n_components, self.n_features_))

        if lateral is not None:
            # make sure m0 is symmetric and positive definite
            lateral = np.asarray(lateral)
            lateral = 0.5 * (lateral + lateral.T)

            evals, evecs = np.linalg.eigh(lateral)
            # XXX should make clipping threshold configurable
            clipping_threshold = 1e-6
            if np.any(evals <= clipping_threshold):
                # clip eigenvalues that are smaller than a small, positive value
                evals = np.clip(evals, clipping_threshold, None)
                lateral = evecs @ np.diag(evals) @ evecs.T

            self.lateral_ = np.array(lateral)
        else:
            self.lateral_ = np.eye(self.n_components)

        # initialize step counter
        self.n_samples_ = 0

        self._mode = "naive"

    # noinspection PyUnusedLocal
    def fit(
        self,
        X: Sequence,
        y: None = None,
        progress: Optional[Callable] = None,
        monitor: Optional[AttributeMonitor] = None,
    ) -> "NonRecurrent":
        """ Feed data to the circuit, updating the output and the weights.

        Note that in this implementation, the non-negativity of the outputs is
        enforced before the synaptic plasticity updates take place.

        Parameters
        ----------
        X
            Dataset to feed into the circuit. Shape `(n_samples, n_features_)`.
        y
            Unused.
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
            An object for monitoring the evolution of the parameters during learning
            (e.g., an instance of `AttributeMonitor`). Parameter values are stored and
            calculated before their updates.

        Returns `self`.
        """
        # figure out per-step rates
        n = len(X)
        if callable(self.learning_rate):
            self._learning_rate_vector = np.array(
                [self.learning_rate(self.n_samples_ + _) for _ in range(n)]
            )
        elif hasattr(self.learning_rate, "__len__"):
            n0 = self.n_samples_
            n1 = n0 + n
            if n1 <= len(self.learning_rate):
                self._learning_rate_vector = self.learning_rate[n0:n1]
            else:
                if n0 < len(self.learning_rate):
                    self._learning_rate_vector = self.learning_rate[n0:]
                    n_extra = n1 - len(self.learning_rate)
                else:
                    self._learning_rate_vector = []
                    n_extra = n1 - n0

                last_rate = self.learning_rate[-1]
                self._learning_rate_vector = np.hstack(
                    (self._learning_rate_vector, n_extra * [last_rate])
                )
        else:
            self._learning_rate_vector = np.repeat(self.learning_rate, n)

        it = X if progress is None else progress(X)

        if monitor is not None:
            monitor.setup(n)

        for i, x in enumerate(it):
            if monitor is not None:
                monitor.record(self)

            self._feed(x, self._learning_rate_vector[i])

        return self

    def _feed(self, x: Sequence, learning_rate: float):
        """ Feed a single data sample into the circuit.

        Parameters
        ----------
        x
            Sample to feed.
        learning_rate
            Learning rate to use.
        """
        # following the first steps from Algorithms 1 and 2 in Minden, Pehlevan,
        # Chklovskii (2018).
        diag_m = np.diag(self.lateral_)
        inv_diag_m = 1 / diag_m
        m_off = self.lateral_ - np.diag(diag_m)

        # the matrix multiplication by the diagonal M_d^{-1} matrix is
        # equivalent to an element-wise multiplication using broadcast
        y_tilde = inv_diag_m * (self.weights_ @ x)

        # now correcting for the off-diagonal terms
        self.output_ = y_tilde - inv_diag_m * (m_off @ y_tilde)

        if self.non_negative:
            self.output_ = np.clip(self.output_, 0, None)

        if learning_rate != 0:
            self.weights_ += learning_rate * (np.outer(self.output_, x) - self.weights_)

            if not self.whiten:
                scaled_m = self.scalings * self.lateral_ * self.scalings[:, None]
                self.lateral_ += (learning_rate / self.tau) * (
                    np.outer(self.output_, self.output_) - scaled_m
                )
            else:
                self.lateral_ += (learning_rate / self.tau) * (
                    np.outer(self.output_, self.output_) - np.diag(self.scalings ** 2)
                )

            self.n_samples_ += 1

    def clone(self):
        """ Make a clone of the current instance. """
        clone = NonRecurrent(
            n_features=self.n_features_,
            n_components=self.n_components,
            weights=self.weights_,
            lateral=self.lateral_,
            tau=self.tau,
            learning_rate=self.learning_rate,
            scalings=self.scalings,
            non_negative=self.non_negative,
            whiten=self.whiten,
        )
        clone.n_samples_ = self.n_samples_
        clone.output_ = np.copy(self.output_)
        return clone

    def __repr__(self):
        return (
            "NonRecurrent(n_features={}, n_components={}, non_negative={}, "
            "whiten={}, learning_rate={}, tau={}, scalings={}, output_={},\n"
            "weights_={},\nlateral_={})".format(
                self.n_features_,
                self.n_components,
                self.non_negative,
                self.whiten,
                self.learning_rate,
                self.tau,
                self.scalings,
                self.output_,
                self.weights_,
                self.lateral_,
            )
        )

    def __str__(self):
        return (
            "NonRecurrent(n_features={}, n_components={}, non_negative={}, "
            "whiten={}, learning_rate={}, tau={})".format(
                self.n_features_,
                self.n_components,
                self.non_negative,
                self.whiten,
                self.learning_rate,
                self.tau,
            )
        )

    _available_modes = ["naive"]
