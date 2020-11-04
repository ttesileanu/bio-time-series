import unittest

import numpy as np
import bioslds

from unittest import mock

from bioslds.monitor import AttributeMonitor
from bioslds.xcorr_regressor import CrosscorrelationRegressor


class TestCrosscorrelationRegressorRegressorInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.xcorr = CrosscorrelationRegressor(self.n_models, self.n_features)

    def test_n_models_attribute(self):
        self.assertEqual(self.xcorr.n_models, self.n_models)

    def test_n_components_attribute(self):
        self.assertTrue(hasattr(self.xcorr, "n_components"))
        self.assertEqual(self.xcorr.n_components, self.n_models)

    def test_n_features_attribute(self):
        self.assertEqual(self.xcorr.n_features, self.n_features)

    def test_call_nsm_constructor(self):
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(2, 3)
        MockNSM.assert_called()

    def test_call_xcorr_constructor(self):
        with mock.patch("bioslds.xcorr_regressor.OnlineCrosscorrelation") as MockXcorr:
            CrosscorrelationRegressor(2, 3)
        MockXcorr.assert_called()

    def test_xcorr_constructor_called_with_correct_n_features(self):
        n_models = 3
        n_features = 4
        with mock.patch("bioslds.xcorr_regressor.OnlineCrosscorrelation") as MockXcorr:
            CrosscorrelationRegressor(n_models, n_features)

        self.assertEqual(MockXcorr.call_args[0][0], n_features)

    def test_xcorr_constructor_called_with_correct_rate(self):
        n_models = 3
        n_features = 4
        xcorr_rate = 0.124
        with mock.patch("bioslds.xcorr_regressor.OnlineCrosscorrelation") as MockXcorr:
            CrosscorrelationRegressor(n_models, n_features, xcorr_rate=xcorr_rate)

        self.assertIn("rate", MockXcorr.call_args[1])
        self.assertEqual(MockXcorr.call_args[1]["rate"], xcorr_rate)

    def test_nsm_constructor_called_with_correct_number_of_dimensions(self):
        n_models = 3
        n_features = 4
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(n_models, n_features)

        self.assertEqual(MockNSM.call_args[0][0], n_features)
        self.assertEqual(MockNSM.call_args[0][1], n_models)

    def test_nsm_constructor_called_with_correct_rate(self):
        n_models = 3
        n_features = 4
        nsm_rate = 0.987
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(n_models, n_features, nsm_rate=nsm_rate)

        self.assertIn("rate", MockNSM.call_args[1])
        self.assertEqual(MockNSM.call_args[1]["rate"], nsm_rate)

    def test_nsm_constructor_called_with_non_negative_true(self):
        n_models = 3
        n_features = 4
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(n_models, n_features)

        self.assertIn("non_negative", MockNSM.call_args[1])
        self.assertTrue(MockNSM.call_args[1]["non_negative"])

    def test_rng_passed_to_nsm_constructor(self):
        n_models = 3
        n_features = 4
        rng = mock.Mock()
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(n_models, n_features, rng=rng)

        self.assertIn("rng", MockNSM.call_args[1])
        self.assertIs(MockNSM.call_args[1]["rng"], rng)

    def test_nsm_kws_passed_to_nsm_constructor(self):
        n_models = 3
        n_features = 4
        foo_value = 1.3
        with mock.patch("bioslds.xcorr_regressor.NonRecurrent") as MockNSM:
            CrosscorrelationRegressor(n_models, n_features, nsm_kws={"foo": foo_value})

        self.assertIn("foo", MockNSM.call_args[1])
        self.assertEqual(MockNSM.call_args[1]["foo"], foo_value)


class TestCrosscorrelationRegressorTransform(unittest.TestCase):
    def setUp(self):
        self.n_models = 2
        self.n_features = 4

        self.n_samples = 34
        self.rng = np.random.default_rng(1)
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))
        self.y = self.rng.normal(size=self.n_samples)

        self.xcorr = CrosscorrelationRegressor(
            self.n_models, self.n_features, rng=self.rng
        )

    def test_calls_xcorr_transform(self):
        with mock.patch.object(
            bioslds.xcorr_regressor.OnlineCrosscorrelation, "transform"
        ) as mock_transform:
            mock_transform.return_value = self.x
            self.xcorr.transform(self.x, self.y)

        mock_transform.assert_called()

    def test_calls_xcorr_transform_with_proper_args(self):
        with mock.patch.object(
            bioslds.xcorr_regressor.OnlineCrosscorrelation, "transform"
        ) as mock_transform:
            mock_transform.return_value = self.x
            self.xcorr.transform(self.x, self.y)

        self.assertIn("X", mock_transform.call_args[1])
        self.assertIn("y", mock_transform.call_args[1])
        np.testing.assert_allclose(mock_transform.call_args[1]["X"], self.x)
        np.testing.assert_allclose(mock_transform.call_args[1]["y"], self.y)

    def test_calls_nsm_transform(self):
        with mock.patch.object(
            bioslds.xcorr_regressor.NonRecurrent, "transform"
        ) as mock_transform:
            mock_transform.return_value = self.rng.normal(
                size=(self.n_samples, self.n_models)
            )
            self.xcorr.transform(self.x, self.y)

        mock_transform.assert_called()

    def test_calls_nsm_transform_with_proper_out_from_xcorr_transform(self):
        with mock.patch.object(
            bioslds.xcorr_regressor.OnlineCrosscorrelation, "transform"
        ) as mock_xcorr_transform, mock.patch.object(
            bioslds.xcorr_regressor.NonRecurrent, "transform"
        ) as mock_nsm_transform:
            trafo_ret = self.rng.normal(size=(self.n_samples, self.n_features))
            mock_xcorr_transform.return_value = trafo_ret
            mock_nsm_transform.return_value = self.rng.normal(
                size=(self.n_samples, self.n_models)
            )
            self.xcorr.transform(self.x, self.y)

        self.assertIn("X", mock_nsm_transform.call_args[1])
        np.testing.assert_allclose(mock_nsm_transform.call_args[1]["X"], trafo_ret)

    def test_returns_out_from_nsm_transform(self):
        with mock.patch.object(
            bioslds.xcorr_regressor.NonRecurrent, "transform"
        ) as mock_transform:
            nsm_out = self.rng.normal(size=(self.n_samples, self.n_models))
            mock_transform.return_value = nsm_out
            res = self.xcorr.transform(self.x, self.y)

        np.testing.assert_allclose(res, nsm_out)

    def test_passes_chunk_hint_to_nsm_transform(self):
        chunk = 13
        with mock.patch.object(
            bioslds.xcorr_regressor.NonRecurrent, "transform"
        ) as mock_transform:
            mock_transform.side_effect = lambda X, **kwargs: self.rng.normal(
                size=(len(X), self.n_models)
            )
            self.xcorr.transform(self.x, self.y, chunk_hint=chunk)

        self.assertIn("chunk_hint", mock_transform.call_args[1])
        self.assertEqual(mock_transform.call_args[1]["chunk_hint"], chunk)

    def test_passes_chunk_hint_to_xcorr_transform(self):
        chunk = 7
        with mock.patch.object(
            bioslds.xcorr_regressor.OnlineCrosscorrelation, "transform"
        ) as mock_transform:
            trafo_ret = lambda X, y, **kwargs: self.rng.normal(
                size=(len(X), self.n_features)
            )
            mock_transform.side_effect = trafo_ret
            self.xcorr.transform(self.x, self.y, chunk_hint=chunk)

        self.assertIn("chunk_hint", mock_transform.call_args[1])
        self.assertEqual(mock_transform.call_args[1]["chunk_hint"], chunk)


class TestCrosscorrelationRegressorTransformMonitor(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 3

        self.n_samples = 71
        self.rng = np.random.default_rng(2)
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))
        self.y = self.rng.normal(size=self.n_samples)

        self.seed = 4
        self.xcorr = CrosscorrelationRegressor(
            self.n_models, self.n_features, rng=self.seed
        )

    def test_with_monitor_object_history_not_returned_by_default(self):
        monitor = AttributeMonitor(["xcorr.coef_"])
        res = self.xcorr.transform(self.x, self.y, monitor=monitor)

        self.assertEqual(np.shape(res), (self.n_samples, self.n_models))

    def test_history_returned_when_return_history_is_true(self):
        monitor = AttributeMonitor(["xcorr.coef_"])
        _, history = self.xcorr.transform(
            self.x, self.y, monitor=monitor, return_history=True
        )

        self.assertIs(history, monitor.history_)

    def test_result_is_the_same_with_and_without_monitor(self):
        monitor = AttributeMonitor(["xcorr.var_"])
        res = self.xcorr.transform(self.x, self.y, monitor=monitor)

        xcorr_alt = CrosscorrelationRegressor(
            self.n_models, self.n_features, rng=self.seed
        )
        res_alt = xcorr_alt.transform(self.x, self.y)

        np.testing.assert_allclose(res, res_alt)

    def test_monitor_as_sequence(self):
        names = ["xcorr.var_", "nsm.weights_"]
        _, history = self.xcorr.transform(
            self.x, self.y, monitor=names, return_history=True
        )

        monitor = AttributeMonitor(names)
        xcorr_alt = CrosscorrelationRegressor(
            self.n_models, self.n_features, rng=self.seed
        )
        _, history_alt = xcorr_alt.transform(
            self.x, self.y, monitor=monitor, return_history=True
        )

        np.testing.assert_allclose(history.xcorr.var_, history_alt.xcorr.var_)
        np.testing.assert_allclose(history.nsm.weights_, history_alt.nsm.weights_)

    def test_when_monitor_sequence_return_history_is_forced_true(self):
        names = ["xcorr.var_", "nsm.weights_"]
        res = self.xcorr.transform(self.x, self.y, monitor=names)

        self.assertEqual(len(res), 2)

    def test_out_matches_monitor_nsm_output(self):
        res, history = self.xcorr.transform(self.x, self.y, monitor=["nsm.output_"])
        np.testing.assert_allclose(res, history.nsm.output_)

    def test_results_same_regardless_of_chunk_hint(self):
        names = ["xcorr.var_"]
        res, history = self.xcorr.transform(self.x, self.y, monitor=names, chunk_hint=5)

        xcorr_alt = CrosscorrelationRegressor(
            self.n_models, self.n_features, rng=self.seed
        )
        res_alt, history_alt = xcorr_alt.transform(
            self.x, self.y, monitor=names, chunk_hint=13
        )

        np.testing.assert_allclose(res, res_alt)
        np.testing.assert_allclose(history.xcorr.var_, history_alt.xcorr.var_)

    def test_passes_chunk_hint_to_nsm_transform(self):
        chunk = 13
        with mock.patch.object(
            bioslds.xcorr_regressor.NonRecurrent, "transform"
        ) as mock_transform:
            mock_transform.side_effect = lambda X, **kwargs: (
                self.rng.normal(size=(len(X), self.n_models))
            )
            self.xcorr.transform(
                self.x, self.y, monitor=["xcorr.var_"], chunk_hint=chunk
            )

        self.assertIn("chunk_hint", mock_transform.call_args[1])
        self.assertEqual(mock_transform.call_args[1]["chunk_hint"], chunk)

    def test_passes_chunk_hint_to_xcorr_transform(self):
        chunk = 7
        with mock.patch.object(
            bioslds.xcorr_regressor.OnlineCrosscorrelation, "transform"
        ) as mock_transform:
            mock_transform.side_effect = lambda X, y, **kwargs: (
                self.rng.normal(size=(len(X), self.n_features))
            )
            self.xcorr.transform(
                self.x, self.y, monitor=["nsm.weights_"], chunk_hint=chunk
            )

        self.assertIn("chunk_hint", mock_transform.call_args[1])
        self.assertEqual(mock_transform.call_args[1]["chunk_hint"], chunk)

    def test_return_history_ignored_when_monitor_is_none(self):
        res = self.xcorr.transform(self.x, self.y, return_history=True)
        self.assertEqual(np.shape(res), (self.n_samples, self.n_models))

    def test_return_history_ignored_when_monitor_is_none_but_progress_is_not(self):
        res = self.xcorr.transform(
            self.x, self.y, progress=mock.MagicMock(), return_history=True
        )
        self.assertEqual(np.shape(res), (self.n_samples, self.n_models))

    def test_progress_called_with_monitor(self):
        mock_progress = mock.MagicMock()

        self.xcorr.transform(
            self.x, self.y, progress=mock_progress, monitor=["xcorr.var_"]
        )
        mock_progress.assert_called()

    def test_progress_called_without_monitor(self):
        mock_progress = mock.MagicMock()

        self.xcorr.transform(self.x, self.y, progress=mock_progress)
        mock_progress.assert_called()


class TestCrosscorrelationRegressorStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 3

        self.xcorr = CrosscorrelationRegressor(self.n_models, self.n_features, rng=1)

    def test_repr(self):
        s = repr(self.xcorr)

        self.assertTrue(s.startswith("CrosscorrelationRegressor("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_models=", s)
        self.assertIn("nsm=", s)
        self.assertIn("xcorr=", s)

    def test_str(self):
        s = str(self.xcorr)

        self.assertTrue(s.startswith("CrosscorrelationRegressor("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_models=", s)
        self.assertIn("nsm=", s)
        self.assertIn("xcorr=", s)


if __name__ == "__main__":
    unittest.main()
