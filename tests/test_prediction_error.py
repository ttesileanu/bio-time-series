import unittest

import numpy as np

from bioslds.prediction_error import PredictionError
from bioslds.monitor import AttributeMonitor


class TestPredictionError(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n_samples = 55
        self.n_features = 4
        self.n_models = 3

        self.x = self.rng.normal(size=(self.n_samples, self.n_features))
        self.y = self.rng.normal(size=self.n_samples)
        self.w = self.rng.normal(size=(self.n_models, self.n_features))

        self.pred = PredictionError(self.w)

    def test_init_sets_n_models(self):
        self.assertTrue(hasattr(self.pred, "n_models"))
        self.assertEqual(self.pred.n_models, self.n_models)

    def test_init_sets_n_components(self):
        self.assertTrue(hasattr(self.pred, "n_components"))
        self.assertEqual(self.pred.n_components, self.n_models)

    def test_init_sets_n_features(self):
        self.assertTrue(hasattr(self.pred, "n_features"))
        self.assertEqual(self.pred.n_features, self.n_features)

    def test_init_creates_output(self):
        self.assertTrue(hasattr(self.pred, "output_"))

    def test_output_is_correct(self):
        err = self.pred.transform(self.x, self.y)

        err_exp = np.zeros((self.n_samples, self.n_models))
        for i in range(self.n_samples):
            crt_yk = self.w @ self.x[i]
            err_exp[i, :] = self.y[i] - crt_yk

        np.testing.assert_allclose(err, err_exp)

    def test_monitor_output_matches_retval(self):
        monitor = AttributeMonitor(["output_"])
        err = self.pred.transform(self.x, self.y, monitor=monitor)

        self.assertTrue(hasattr(monitor.history_, "output_"))
        np.testing.assert_allclose(monitor.history_.output_, err)

    def test_monitor_history_is_returned_when_return_history_is_true(self):
        monitor = AttributeMonitor(["output_"])
        _, history = self.pred.transform(
            self.x, self.y, monitor=monitor, return_history=True
        )

        self.assertIs(history, monitor.history_)

    def test_retval_is_same_when_chunk_hint_changes(self):
        err = self.pred.transform(self.x, self.y, chunk_hint=11)
        err_alt = self.pred.transform(self.x, self.y, chunk_hint=17)

        np.testing.assert_allclose(err, err_alt)

    def test_monitor_history_is_same_when_chunk_hint_changes(self):
        monitor = AttributeMonitor(["output_"])
        self.pred.transform(self.x, self.y, chunk_hint=11, monitor=monitor)

        monitor_alt = AttributeMonitor(["output_"])
        self.pred.transform(self.x, self.y, chunk_hint=11, monitor=monitor_alt)

        np.testing.assert_allclose(
            monitor.history_.output_, monitor_alt.history_.output_
        )

    def test_transform_sets_output(self):
        err = self.pred.transform(self.x, self.y)

        np.testing.assert_allclose(self.pred.output_, err[-1])


if __name__ == "__main__":
    unittest.main()
