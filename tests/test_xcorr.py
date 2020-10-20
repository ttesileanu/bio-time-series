import unittest

import numpy as np

from unittest import mock

from bioslds.monitor import AttributeMonitor
from bioslds.xcorr import OnlineCrosscorrelation


class TestOnlineCrosscorrelationTransformMonitor(unittest.TestCase):
    def setUp(self):
        self.n_components = 3
        self.rate = 0.05
        self.acorr = OnlineCrosscorrelation(self.n_components, rate=self.rate)

        self.rng = np.random.default_rng(1)
        self.n_samples = 53
        self.x = self.rng.normal(size=(self.n_samples, self.n_components))
        self.y = self.rng.normal(size=self.n_samples)

    def test_initial_variance_is_one(self):
        self.assertAlmostEqual(self.acorr.var_, 1)

    def test_works_without_monitor(self):
        try:
            self.acorr.transform(self.x, self.y)
        except AttributeError:
            raise AssertionError("monitor called when it was None.")

    def test_monitor_setup_called(self):
        monitor = mock.Mock()

        self.acorr.transform(self.x, self.y, monitor=monitor)
        monitor.setup.assert_called()

    def test_monitor_setup_called_with_correct_arguments(self):
        monitor = mock.Mock()

        self.acorr.transform(self.x, self.y, monitor=monitor)
        monitor.setup.assert_called_with(self.n_samples)

    def test_monitor_coef_same_as_return_from_transform(self):
        monitor = AttributeMonitor(["coef_"])

        res = self.acorr.transform(self.x, self.y, monitor=monitor)
        np.testing.assert_allclose(res, monitor.history_.coef_)

    def test_monitor_var_correct(self):
        monitor = AttributeMonitor(["var_"])

        self.acorr.transform(self.x, self.y, monitor=monitor)

        acorr_alt = OnlineCrosscorrelation(self.n_components, rate=self.rate)
        var_exp = np.zeros(self.n_samples)
        for i in range(self.n_samples):
            var_exp[i] = acorr_alt.var_
            acorr_alt.transform(self.x[[i]], self.y[[i]])

        np.testing.assert_allclose(monitor.history_.var_, var_exp)


class TestOnlineCrosscorrelationTransformCorrectness(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.rate = 0.03
        self.acorr = OnlineCrosscorrelation(self.n_components, rate=self.rate)

        self.rng = np.random.default_rng(3)
        self.n_samples = 49
        self.x = self.rng.normal(size=(self.n_samples, self.n_components))
        self.y = self.rng.normal(size=self.n_samples)

    def test_reference_implementation_for_variance_estimate(self):
        self.acorr.transform(self.x[:-1], self.y[:-1])

        crt_var = self.acorr.var_
        exp_var = crt_var + self.rate * (self.y[-1] ** 2 - crt_var)

        self.acorr.transform(self.x[[-1]], self.y[[-1]])

        self.assertAlmostEqual(self.acorr.var_, exp_var)

    def test_reference_implementation_for_xcorr_estimate(self):
        self.acorr.transform(self.x[:-1], self.y[:-1])

        crt_coef = self.acorr.coef_
        crt_var = self.acorr.var_
        exp_coef = crt_coef + self.rate * (
            (self.y[-1] / crt_var) * self.x[-1] - crt_coef
        )

        self.acorr.transform(self.x[[-1]], self.y[[-1]])

        np.testing.assert_allclose(self.acorr.coef_, exp_coef)

    def test_progress_called_once(self):
        mock_progress = mock.MagicMock()

        self.acorr.transform(self.x, self.y, progress=mock_progress)

        mock_progress.assert_called_once()


class TestOnlineCrosscorrelationStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.rate = 0.03
        self.acorr = OnlineCrosscorrelation(self.n_components, rate=self.rate)

    def test_repr(self):
        s = repr(self.acorr)

        self.assertTrue(s.startswith("OnlineCrosscorrelation("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_components=", s)
        self.assertIn("rate=", s)
        self.assertIn("var_=", s)
        self.assertIn("coef_=", s)

    def test_str(self):
        s = str(self.acorr)

        self.assertTrue(s.startswith("OnlineCrosscorrelation("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_components=", s)
        self.assertIn("rate=", s)
        self.assertNotIn("var_=", s)
        self.assertNotIn("coef_=", s)


if __name__ == "__main__":
    unittest.main()
