import unittest

import numpy as np

from bioslds.variance import OnlineVariance
from bioslds.monitor import AttributeMonitor


class TestOnlineVarianceConstructor(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.rate = 0.02
        self.variance = OnlineVariance(self.n_features, rate=self.rate)

    def test_n_features_set(self):
        self.assertEqual(self.variance.n_features, self.n_features)

    def test_n_components_set_to_n_features(self):
        self.assertEqual(self.variance.n_components, self.n_features)

    def test_rate_set(self):
        self.assertEqual(self.variance.rate, self.rate)


class TestOnlineVarianceTransform(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.rate = 0.02
        self.variance = OnlineVariance(self.n_features, rate=self.rate)

        self.rng = np.random.default_rng(33)
        self.n_samples = 500
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

    def test_all_outputs_are_non_negative(self):
        norms = self.variance.transform(X=self.x)
        self.assertGreaterEqual(np.min(norms), 0)

    def test_result_does_not_depend_on_position_in_feature_dimension(self):
        norms1 = self.variance.transform(X=self.x)

        variance2 = OnlineVariance(self.n_features, rate=self.rate)
        norms2 = variance2.transform(X=np.fliplr(self.x))

        np.testing.assert_allclose(norms1, np.fliplr(norms2))

    def test_result_in_one_seq_independent_of_other_seqs(self):
        X1 = self.x
        norms1 = self.variance.transform(X=X1)

        X2 = np.copy(self.x)
        X2[:, 1] = self.x[:, 2]
        variance2 = OnlineVariance(self.n_features, rate=self.rate)
        norms2 = variance2.transform(X=X2)

        np.testing.assert_allclose(norms1[:, 0], norms2[:, 0])

    def test_initial_estimate_is_zero(self):
        norms = self.variance.transform(X=self.x)

        np.testing.assert_allclose(norms[0], self.rate * self.x[0] ** 2)

    def test_one_step_update_end(self):
        norms = self.variance.transform(X=self.x)
        diffs = np.diff(norms, axis=0)

        expected_last_diff = self.rate * (self.x[-1] ** 2 - norms[-2])
        np.testing.assert_allclose(diffs[-1], expected_last_diff)

    def test_shape_of_output_correct(self):
        norms = self.variance.transform(X=self.x)
        self.assertEqual(norms.shape, self.x.shape)

    def test_repeated_transform_same_as_single_transform_on_combined_input(self):
        norms = self.variance.transform(X=self.x)

        n1 = self.n_samples // 2
        x1 = self.x[:n1]
        x2 = self.x[n1:]

        variance_alt = OnlineVariance(self.n_features, rate=self.rate)
        norms1 = variance_alt.transform(X=x1)
        norms2 = variance_alt.transform(X=x2)

        norms_exp = np.row_stack((norms1, norms2))
        np.testing.assert_allclose(norms, norms_exp)

    def test_output_is_set_properly(self):
        norms = self.variance.transform(X=self.x)
        np.testing.assert_allclose(self.variance.output_, norms[-1])

    def test_monitor_output(self):
        monitor = AttributeMonitor(["output_"])
        norms = self.variance.transform(X=self.x, monitor=monitor)

        np.testing.assert_allclose(monitor.history_.output_, norms)


class TestOnlineVarianceNegative(unittest.TestCase):
    def setUp(self):
        self.n_features = 4
        self.order = 4
        self.rate = 0.01
        self.variance = OnlineVariance(self.n_features, rate=self.rate)

        self.rng = np.random.default_rng(33)
        self.n_samples = 500
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

        self.norms = self.variance.transform(X=self.x)

    def test_negative_returned_when_asked_for(self):
        variance_neg = OnlineVariance(self.n_features, rate=self.rate, negative=True)
        norms_neg = variance_neg.transform(X=self.x)

        np.testing.assert_allclose(norms_neg, -self.norms)


class TestOnlineVarianceStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.order = 5
        self.rate = 0.02
        self.variance = OnlineVariance(self.n_features, rate=self.rate)

    def test_repr(self):
        s = repr(self.variance)

        self.assertTrue(s.startswith("OnlineVariance("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("rate=", s)

    def test_str(self):
        s = str(self.variance)

        self.assertTrue(s.startswith("OnlineVariance("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("rate=", s)


if __name__ == "__main__":
    unittest.main()
