import unittest

import numpy as np

from unittest import mock

from bioslds.monitor import AttributeMonitor
from bioslds.nsm import NonRecurrent


def generate_random_from_kernel(
    a: np.ndarray, d: int, rng: np.random.Generator
) -> np.ndarray:
    """ Use the random number generator `rng` to draw a random `d`-dimensional
    subspace from the kernel of matrix `a`. If `dim(ker(a)) < d`, an exception
    is raised.
    """
    evals, evecs = np.linalg.eigh(a.T @ a)
    # kernel is where evals are (almost) zero
    ker_dims = (evals < 1e-10).nonzero()[0]
    size_ker = len(ker_dims)

    if size_ker < d:
        raise ValueError("Kernel dimension lower than requested subspace dimension")

    chosen_dims = rng.choice(ker_dims, size=d, replace=False)
    chosen_basis = evecs[:, chosen_dims]

    b = rng.normal(size=(d, d)) @ chosen_basis.T

    return b


class TestNonRecurrentConstructor(unittest.TestCase):
    def test_raises_if_all_of_w0_m0_and_n_components_missing(self):
        with self.assertRaises(ValueError):
            NonRecurrent(n_features=3)

    def test_raises_if_both_w0_and_n_features_missing(self):
        with self.assertRaises(ValueError):
            NonRecurrent(n_components=4)

    def test_raises_if_n_features_does_not_match_weights_shape(self):
        with self.assertRaises(ValueError):
            NonRecurrent(n_features=3, weights=[[1, 2], [2, 3], [4, 5]])

    def test_raises_if_n_components_does_not_match_weights_shape(self):
        with self.assertRaises(ValueError):
            NonRecurrent(n_components=3, weights=[[1, 2, 3]])

    def test_raises_if_weights_shape_does_not_match_lateral_shape(self):
        with self.assertRaises(ValueError):
            NonRecurrent(weights=[[1, 2, 3]], lateral=[[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_raises_if_lateral_not_square(self):
        with self.assertRaises(ValueError):
            NonRecurrent(lateral=[[1, 2]])

    def test_whiten_is_false_by_default(self):
        circuit = NonRecurrent(n_features=5, n_components=4)
        self.assertFalse(circuit.whiten)

    def test_n_samples_starts_at_0(self):
        circuit = NonRecurrent(n_features=3, n_components=4)
        self.assertEqual(0, circuit.n_samples_)


class TestNonRecurrentSpecialCases(unittest.TestCase):
    def test_output_identically_zero_if_input_is_in_ker_weights(self):
        rng = np.random.default_rng(0)

        n_steps = 10
        input_dim = 4
        output_dim = 2  # <input_dim so we have non-trivial kernel

        w0 = rng.normal(size=(output_dim, input_dim))
        circuit = NonRecurrent(weights=w0)

        for k in range(n_steps):
            x = generate_random_from_kernel(w0, 1, rng).ravel()
            circuit.fit([x])
            np.testing.assert_allclose(circuit.output_, 0, atol=1e-10)

    def test_weights_just_decaying_if_input_is_in_ker_weights(self):
        rng = np.random.default_rng(1)
        n_steps = 10

        input_dim = 5
        output_dim = 4  # <input_dim so we have non-trivial kernel

        w0 = rng.normal(size=(output_dim, input_dim))
        alpha = 0.0009
        tau = 0.4
        circuit = NonRecurrent(weights=w0, tau=tau, learning_rate=alpha)

        gamma_w = 1 - circuit.learning_rate

        for k in range(n_steps):
            x = generate_random_from_kernel(w0, 1, rng).ravel()
            circuit.fit([x])
            np.testing.assert_allclose(circuit.weights_, (gamma_w ** (k + 1)) * w0)

    def test_lateral_just_decaying_if_input_is_in_ker_weights(self):
        # assuming default is PSP problem, not PSW (i.e., whiten == False)
        rng = np.random.default_rng(2)
        n_steps = 10

        input_dim = 3
        output_dim = 2  # <input_dim so we have non-trivial kernel

        w0 = rng.normal(size=(output_dim, input_dim))
        sqrt_m0 = rng.normal(size=(output_dim, output_dim))
        m0 = sqrt_m0 @ sqrt_m0.T
        alpha = 0.0015
        tau = 0.7
        circuit = NonRecurrent(weights=w0, lateral=m0, tau=tau, learning_rate=alpha)

        gamma_m = 1 - circuit.learning_rate / circuit.tau

        for k in range(n_steps):
            x = generate_random_from_kernel(w0, 1, rng).ravel()
            circuit.fit([x])
            np.testing.assert_allclose(circuit.lateral_, (gamma_m ** (k + 1)) * m0)


class TestNonRecurrentInvariants(unittest.TestCase):
    def test_constructor_ensures_lateral_is_positive_definite(self):
        # this implies that it should be symmetric
        rng = np.random.default_rng(0)
        output_dim = 3

        m0 = rng.normal(size=(output_dim, output_dim))
        circuit = NonRecurrent(n_features=5, lateral=m0)

        # symmetric...
        self.assertEqual(circuit.lateral_.shape[0], circuit.lateral_.shape[1])

        # ...and positive-definite
        evals, _ = np.linalg.eigh(circuit.lateral_)
        self.assertGreaterEqual(np.min(evals), 0)

    def test_constructor_leaves_lateral_unchanged_if_it_is_positive_definite(self):
        rng = np.random.default_rng(0)
        output_dim = 3

        sqrt_m0 = rng.normal(size=(output_dim, output_dim))
        m0 = sqrt_m0 @ sqrt_m0.T
        circuit = NonRecurrent(n_features=5, lateral=m0)

        np.testing.assert_allclose(circuit.lateral_, m0)


class TestNonRecurrentFit(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

        self.input_dim = 5
        self.output_dim = 3

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.001
        self.tau = 0.6

        self.circuit = NonRecurrent(
            weights=self.w0, lateral=self.m0, tau=self.tau, learning_rate=self.alpha
        )

    def test_n_samples_is_advanced_by_fit(self):
        n_samples = 3
        self.circuit.fit(n_samples * [[1, 2, 3, 1, 2]])
        self.assertEqual(n_samples, self.circuit.n_samples_)

    def test_output_after_fit_single_sample(self):
        # testing the algorithm from Minden, Pehlevan, Chklovskii.
        n_steps = 10
        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            m_diag = np.diag(np.diag(self.circuit.lateral_))
            y_tilde = np.linalg.inv(m_diag) @ self.circuit.weights_ @ x
            expected_y = (
                np.eye(self.output_dim)
                - np.linalg.inv(m_diag) @ (self.circuit.lateral_ - m_diag)
            ) @ y_tilde

            self.circuit.fit([x])

            np.testing.assert_allclose(self.circuit.output_, expected_y)

    def test_forward_weights_evolution(self):
        # testing the algorithm from Minden, Pehlevan, Chklovskii.
        n_steps = 10
        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_w = np.array(self.circuit.weights_)

            self.circuit.fit([x])

            expected_w = old_w + self.alpha * (
                np.outer(self.circuit.output_, x) - old_w
            )
            np.testing.assert_allclose(self.circuit.weights_, expected_w)

    def test_lateral_weights_evolution_no_whitening_no_pc_scaling(self):
        # testing algorithm 1 from Minden, Pehlevan, Chklovskii.
        n_steps = 10
        self.circuit.whiten = False

        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_m = np.array(self.circuit.lateral_)

            self.circuit.fit([x])

            expected_m = old_m + (self.alpha / self.tau) * (
                np.outer(self.circuit.output_, self.circuit.output_) - old_m
            )
            np.testing.assert_allclose(self.circuit.lateral_, expected_m)

    def test_lateral_weights_evolution_with_whitening_no_pc_scaling(self):
        # testing algorithm 2 from Minden, Pehlevan, Chklovskii.
        n_steps = 10
        self.circuit.whiten = True

        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_m = np.array(self.circuit.lateral_)

            self.circuit.fit([x])

            expected_m = old_m + (self.alpha / self.tau) * (
                np.outer(self.circuit.output_, self.circuit.output_)
                - np.eye(self.output_dim)
            )
            np.testing.assert_allclose(self.circuit.lateral_, expected_m)


class TestNonRecurrentFitWithPCScaling(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

        self.input_dim = 4
        self.output_dim = 3

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.0005
        self.tau = 0.4
        self.pc_scalings = self.rng.uniform(size=self.output_dim)

        self.circuit = NonRecurrent(
            weights=self.w0,
            lateral=self.m0,
            tau=self.tau,
            learning_rate=self.alpha,
            scalings=self.pc_scalings,
        )

    def test_lateral_weights_evolution_no_whitening(self):
        n_steps = 10
        self.circuit.whiten = False
        lbd = np.diag(self.pc_scalings)

        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_m = np.array(self.circuit.lateral_)

            self.circuit.fit([x])

            expected_m = old_m + (self.alpha / self.tau) * (
                np.outer(self.circuit.output_, self.circuit.output_) - lbd @ old_m @ lbd
            )
            np.testing.assert_allclose(self.circuit.lateral_, expected_m)

    def test_lateral_weights_evolution_with_whitening(self):
        n_steps = 10
        self.circuit.whiten = True

        lbd = np.diag(self.pc_scalings)

        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_m = np.array(self.circuit.lateral_)

            self.circuit.fit([x])

            expected_m = old_m + (self.alpha / self.tau) * (
                np.outer(self.circuit.output_, self.circuit.output_) - lbd @ lbd
            )
            np.testing.assert_allclose(self.circuit.lateral_, expected_m)


class TestNonRecurrentFitNonnegative(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

        self.input_dim = 3
        self.output_dim = 2

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.0008
        self.tau = 0.7

        self.circuit = NonRecurrent(
            weights=self.w0,
            lateral=self.m0,
            tau=self.tau,
            learning_rate=self.alpha,
            non_negative=True,
        )

    def test_outputs_stay_non_negative(self):
        n_steps = 10
        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            self.circuit.fit([x])
            self.assertGreaterEqual(np.min(self.circuit.output_), 0)

    def test_constraint_implemented_before_weights_evolution(self):
        n_steps = 10
        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_w = np.array(self.circuit.weights_)

            self.circuit.fit([x])

            expected_w = old_w + self.alpha * (
                np.outer(self.circuit.output_, x) - old_w
            )
            np.testing.assert_allclose(self.circuit.weights_, expected_w)


class TestNonRecurrentFitNonnegativeWithPCScalings(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

        self.input_dim = 4
        self.output_dim = 2

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.001
        self.tau = 0.5
        self.pc_scalings = self.rng.uniform(size=self.output_dim)

        self.kwargs = dict(
            weights=self.w0,
            lateral=self.m0,
            learning_rate=self.alpha,
            tau=self.tau,
            non_negative=True,
            scalings=self.pc_scalings,
            whiten=False,
        )

        self.circuit = NonRecurrent(**self.kwargs)

    def test_constraint_implemented_before_lateral_evolution(self):
        n_steps = 10

        lbd = np.diag(self.pc_scalings)

        for k in range(n_steps):
            x = self.rng.normal(size=self.input_dim)

            old_m = np.array(self.circuit.lateral_)

            self.circuit.fit([x])

            expected_m = old_m + (self.alpha / self.tau) * (
                np.outer(self.circuit.output_, self.circuit.output_) - lbd @ old_m @ lbd
            )
            np.testing.assert_allclose(self.circuit.lateral_, expected_m)


class TestNonRecurrentFitInfer(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

        self.input_dim = 4
        self.output_dim = 2

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.0011
        self.tau = 0.34
        self.pc_scalings = self.rng.uniform(size=self.output_dim)

        self.kwargs = dict(
            weights=self.w0,
            lateral=self.m0,
            learning_rate=self.alpha,
            tau=self.tau,
            scalings=self.pc_scalings,
        )
        self.circuit = NonRecurrent(**self.kwargs)

        self.n_samples = 85
        self.x = self.rng.normal(size=(self.n_samples, self.input_dim))

    def test_fit_infer_returns_same_as_monitor_output(self):
        monitor = AttributeMonitor(["output_"])
        res = self.circuit.fit_infer(self.x, monitor=monitor)

        np.testing.assert_allclose(res, monitor.history_.output_)


class TestNonRecurrentClone(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)

        self.input_dim = 5
        self.output_dim = 3

        self.w0 = self.rng.normal(size=(self.output_dim, self.input_dim))
        self.sqrt_m0 = self.rng.normal(size=(self.output_dim, self.output_dim))
        self.m0 = self.sqrt_m0 @ self.sqrt_m0.T

        self.alpha = 0.0012
        self.tau = 0.35
        self.pc_scalings = self.rng.uniform(size=self.output_dim)

        self.kwargs = dict(
            weights=self.w0,
            lateral=self.m0,
            learning_rate=self.alpha,
            tau=self.tau,
            scalings=self.pc_scalings,
        )
        self.circuit = NonRecurrent(**self.kwargs)

    def test_clone_copies_meta_parameters(self):
        circuit_copy = self.circuit.clone()

        self.assertEqual(circuit_copy.n_components, self.circuit.n_components)
        self.assertEqual(circuit_copy.learning_rate, self.circuit.learning_rate)
        self.assertEqual(circuit_copy.tau, self.circuit.tau)
        np.testing.assert_allclose(circuit_copy.scalings, self.circuit.scalings)
        self.assertEqual(circuit_copy.non_negative, self.circuit.non_negative)
        self.assertEqual(circuit_copy.whiten, self.circuit.whiten)

    def test_clone_copies_last_output(self):
        self.circuit.fit(self.rng.normal(size=(1, self.input_dim)))
        circuit_copy = self.circuit.clone()
        np.testing.assert_allclose(self.circuit.output_, circuit_copy.output_)


class TestNonRecurrentRandom(unittest.TestCase):
    def setUp(self):
        self.n_features = 4
        self.n_components = 3
        self.seed = 4
        self.rng = np.random.default_rng(self.seed)
        self.kwargs = dict(n_features=self.n_features, n_components=self.n_components)
        self.circuit = NonRecurrent(rng=self.rng, **self.kwargs)

    def test_random_state_changes_weights(self):
        circuit1 = NonRecurrent(rng=np.random.default_rng(2), **self.kwargs)

        self.assertGreater(
            np.max(np.abs(self.circuit.weights_ - circuit1.weights_)), 1e-4
        )

    def test_random_state_does_not_change_lateral(self):
        circuit1 = NonRecurrent(rng=np.random.default_rng(3), **self.kwargs)
        np.testing.assert_allclose(circuit1.lateral_, self.circuit.lateral_)

    def test_rng_as_seed(self):
        circuit1 = NonRecurrent(rng=self.seed, **self.kwargs)

        np.testing.assert_allclose(self.circuit.weights_, circuit1.weights_)
        np.testing.assert_allclose(self.circuit.lateral_, circuit1.lateral_)

    def test_repeated_calls_to_same_generator_lead_to_different_results(self):
        circuit1 = NonRecurrent(**self.kwargs, rng=self.rng)

        self.assertGreater(
            np.max(np.abs(self.circuit.weights_ - circuit1.weights_)), 1e-4
        )


class MockProgress(object):
    def __init__(self):
        self.calls = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))

        return args[0]


class TestNonRecurrentMonitoring(unittest.TestCase):
    def setUp(self):
        self.n_features = 4
        self.n_components = 3
        self.seed = 4
        self.rng = np.random.default_rng(self.seed)
        self.kwargs = dict(n_features=self.n_features, n_components=self.n_components)
        self.circuit = NonRecurrent(rng=self.rng, **self.kwargs)

    def test_progress_called_in_fit(self):
        mock_progress = mock.MagicMock()

        self.circuit.fit(np.zeros((30, self.n_features)), progress=mock_progress)

        mock_progress.assert_called()

    def test_monitor_fit(self):
        names = ["weights_", "lateral_", "output_"]
        monitor = AttributeMonitor(names)

        n_samples = 100

        rng = np.random.default_rng(1)
        x = rng.normal(size=(n_samples, self.n_features))

        self.circuit.fit(x, monitor=monitor)

        circuit1 = NonRecurrent(**self.kwargs, rng=self.seed)
        weights = []
        lateral = []
        output = []
        for crt_x in x:
            weights.append(np.copy(circuit1.weights_))
            lateral.append(np.copy(circuit1.lateral_))
            output.append(np.copy(circuit1.output_))

            circuit1.fit([crt_x])

        np.testing.assert_allclose(monitor.history_.weights_, weights)
        np.testing.assert_allclose(monitor.history_.lateral_, lateral)
        np.testing.assert_allclose(monitor.history_.output_, output)

    def test_history_same_when_chunk_hint_changes(self):
        names = ["weights_", "lateral_", "output_"]
        monitor = AttributeMonitor(names)

        n_samples = 100

        rng = np.random.default_rng(1)
        x = rng.normal(size=(n_samples, self.n_features))

        self.circuit.fit(x, monitor=monitor, chunk_hint=13)

        circuit_alt = NonRecurrent(**self.kwargs, rng=self.seed)
        monitor_alt = AttributeMonitor(names)
        circuit_alt.fit(x, monitor=monitor_alt, chunk_hint=2)

        np.testing.assert_allclose(
            monitor.history_.weights_, monitor_alt.history_.weights_
        )
        np.testing.assert_allclose(
            monitor.history_.lateral_, monitor_alt.history_.lateral_
        )
        np.testing.assert_allclose(
            monitor.history_.output_, monitor_alt.history_.output_
        )


class TestNonRecurrentStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_features = 5
        self.n_components = 3
        self.seed = 3
        self.rng = np.random.default_rng(self.seed)
        self.kwargs = dict(n_features=self.n_features, n_components=self.n_components)
        self.circuit = NonRecurrent(rng=self.rng, **self.kwargs)

    def test_repr(self):
        s = repr(self.circuit)

        self.assertTrue(s.startswith("NonRecurrent("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_components=", s)
        self.assertIn("non_negative=", s)
        self.assertIn("whiten=", s)
        self.assertIn("learning_rate=", s)
        self.assertIn("tau=", s)
        self.assertIn("scalings=", s)
        self.assertIn("output_=", s)
        self.assertIn("weights_=", s)
        self.assertIn("lateral_=", s)

    def test_str(self):
        s = str(self.circuit)

        self.assertTrue(s.startswith("NonRecurrent("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_components=", s)
        self.assertIn("non_negative=", s)
        self.assertIn("whiten=", s)
        self.assertIn("learning_rate=", s)
        self.assertIn("tau=", s)
        self.assertNotIn("scalings=", s)
        self.assertNotIn("output_=", s)
        self.assertNotIn("weights_=", s)
        self.assertNotIn("lateral_=", s)


class TestNonRecurrentVectorLearningRate(unittest.TestCase):
    def setUp(self):
        self.n_features = 5
        self.n_components = 3
        self.seed = 3
        self.kwargs = dict(
            n_features=self.n_features, n_components=self.n_components, rng=self.seed
        )

        self.rng = np.random.default_rng(0)
        self.n_samples = 53
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

        self.rate = 0.005

        self.monitor_full = AttributeMonitor(["output_"])
        self.circuit_full = NonRecurrent(**self.kwargs, learning_rate=self.rate)
        self.circuit_full.fit(self.x, monitor=self.monitor_full)

        self.n_partial = self.n_samples // 2
        self.monitor_partial = AttributeMonitor(["output_"])
        self.circuit_partial = NonRecurrent(**self.kwargs, learning_rate=self.rate)
        self.circuit_partial.fit(self.x[: self.n_partial], monitor=self.monitor_partial)

    def test_final_weights_different_in_partial_and_full_run(self):
        self.assertGreater(
            np.max(np.abs(self.circuit_partial.weights_ - self.circuit_full.weights_)),
            1e-3,
        )
        self.assertGreater(
            np.max(np.abs(self.circuit_partial.lateral_ - self.circuit_full.lateral_)),
            1e-3,
        )

    def test_switching_rate_to_zero_fixes_weights(self):
        schedule = np.zeros(self.n_samples)
        schedule[: self.n_partial] = self.rate
        circuit = NonRecurrent(**self.kwargs, learning_rate=schedule)

        circuit.fit(self.x)

        np.testing.assert_allclose(circuit.weights_, self.circuit_partial.weights_)

    def test_output_history_same_if_rate_is_constant_then_switches(self):
        schedule = np.zeros(self.n_samples)
        schedule[: self.n_partial] = self.rate
        circuit = NonRecurrent(**self.kwargs, learning_rate=schedule)

        monitor = AttributeMonitor(["output_"])
        circuit.fit(self.x, monitor=monitor)
        np.testing.assert_allclose(
            monitor.history_.output_[: self.n_partial],
            self.monitor_partial.history_.output_,
        )

    def test_constructor_copies_weight_schedule(self):
        schedule = self.rate * np.ones(self.n_samples)
        circuit = NonRecurrent(**self.kwargs, learning_rate=schedule)

        schedule[:] = 0
        circuit.fit(self.x)

        np.testing.assert_allclose(circuit.weights_, self.circuit_full.weights_)


class TestNonRecurrentVectorLearningRateOther(unittest.TestCase):
    def setUp(self):
        self.n_features = 5
        self.n_components = 3
        self.seed = 3
        self.kwargs = dict(
            n_features=self.n_features, n_components=self.n_components, rng=self.seed
        )

        self.rng = np.random.default_rng(0)
        self.n_samples = 53
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

        self.rate = 0.005

    def test_last_value_of_rate_is_used_if_more_samples_than_len_rate(self):
        n = 3 * self.n_samples // 4
        schedule_short = self.rng.uniform(0, self.rate, size=n)
        schedule = np.hstack(
            (schedule_short, (self.n_samples - n) * [schedule_short[-1]])
        )

        circuit1 = NonRecurrent(**self.kwargs, learning_rate=schedule_short)
        circuit2 = NonRecurrent(**self.kwargs, learning_rate=schedule)

        circuit1.fit(self.x)
        circuit2.fit(self.x)

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)

    def test_schedule_used_in_sequence_for_multiple_calls_to_fit(self):
        schedule = self.rng.uniform(0, self.rate, size=self.n_samples)

        circuit1 = NonRecurrent(**self.kwargs, learning_rate=schedule)
        circuit2 = NonRecurrent(**self.kwargs, learning_rate=schedule)

        circuit1.fit(self.x)

        circuit2.fit(self.x[: self.n_samples // 2])
        circuit2.fit(self.x[self.n_samples // 2 :])

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)


class TestNonRecurrentCallableLearningRate(unittest.TestCase):
    def test_callable_rate_works_like_vector(self):
        n_features = 5
        n_components = 3
        seed = 3
        kwargs = dict(n_features=n_features, n_components=n_components, rng=seed)

        rng = np.random.default_rng(0)
        n_samples = 55
        x = rng.normal(size=(n_samples, n_features))

        def rate_fct(i):
            return 1 / (100 + 5 * i)

        circuit1 = NonRecurrent(**kwargs, learning_rate=rate_fct)
        circuit1.fit(x)

        schedule = [rate_fct(_) for _ in range(n_samples)]
        circuit2 = NonRecurrent(**kwargs, learning_rate=schedule)
        circuit2.fit(x)

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)
        np.testing.assert_allclose(circuit1.lateral_, circuit2.lateral_)

    def test_callable_rate_works_like_constant(self):
        n_features = 5
        n_components = 3
        seed = 3
        kwargs = dict(n_features=n_features, n_components=n_components, rng=seed)

        rng = np.random.default_rng(0)
        n_samples = 55
        x = rng.normal(size=(n_samples, n_features))

        rate = 1e-4

        def rate_fct(_):
            return rate

        circuit1 = NonRecurrent(**kwargs, learning_rate=rate_fct)
        circuit1.fit(x)

        schedule = [rate_fct(_) for _ in range(n_samples)]
        circuit2 = NonRecurrent(**kwargs, learning_rate=schedule)
        circuit2.fit(x)

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)
        np.testing.assert_allclose(circuit1.lateral_, circuit2.lateral_)

    def test_schedule_used_in_sequence_for_multiple_calls_to_fit(self):
        n_features = 6
        n_components = 4
        seed = 2
        kwargs = dict(n_features=n_features, n_components=n_components, rng=seed)

        rng = np.random.default_rng(0)
        n_samples = 50
        x = rng.normal(size=(n_samples, n_features))

        def rate_fct(i):
            return 1 / (100 + 5 * i)

        circuit1 = NonRecurrent(**kwargs, learning_rate=rate_fct)
        circuit2 = NonRecurrent(**kwargs, learning_rate=rate_fct)

        circuit1.fit(x)

        circuit2.fit(x[: n_samples // 2])
        circuit2.fit(x[n_samples // 2 :])

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)


class TestNonRecurrentChunkHintDoesNotAffectResult(unittest.TestCase):
    def setUp(self):
        self.n_features = 5
        self.n_components = 3
        self.seed = 3
        self.kwargs = dict(
            n_features=self.n_features, n_components=self.n_components, rng=self.seed
        )

        self.rng = np.random.default_rng(0)
        self.n_samples = 100
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

    def test_small_chunk_same_as_no_chunk(self):
        circuit1 = NonRecurrent(**self.kwargs,)
        circuit1.fit(self.x)

        circuit2 = NonRecurrent(**self.kwargs,)
        circuit2.fit(self.x, chunk_hint=12)

        np.testing.assert_allclose(circuit1.weights_, circuit2.weights_)
        np.testing.assert_allclose(circuit1.lateral_, circuit2.lateral_)
        np.testing.assert_allclose(circuit1.output_, circuit2.output_)
