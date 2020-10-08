import unittest

import numpy as np

from unittest import mock
from unittest.mock import Mock

from types import SimpleNamespace
from typing import Union, Sequence, Callable, Optional, Tuple

from bioslds.batch import hyper_score_ar


class MockRegressor(object):
    def __init__(
        self,
        n_models: int,
        n_features: int,
        rng: Union[int, np.random.RandomState, np.random.Generator],
    ):
        self.n_models = n_models
        self.n_features = n_features
        self.rng = rng

    # noinspection PyUnusedLocal
    def fit_infer(
        self,
        X: Sequence,
        y: Sequence,
        progress: Optional[Callable] = None,
        monitor: Optional[Sequence] = None,
    ) -> Tuple[np.ndarray, SimpleNamespace]:
        return np.ones((len(y), self.n_models)), SimpleNamespace()


class TestHyperScoreARBasic(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n_signals = 5
        self.n_samples = 42
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]

    def test_returns_tuple(self):
        res = hyper_score_ar(
            MockRegressor, self.dataset, lambda x, y: 1.0, n_models=2, n_features=2
        )
        self.assertEqual(len(res), 2)

    def test_metric_called_n_signals_times(self):
        metric = Mock(return_value=1.0)
        # noinspection PyTypeChecker
        hyper_score_ar(MockRegressor, self.dataset, metric, n_models=2, n_features=2)
        self.assertEqual(metric.call_count, self.n_signals)


class TestHyperScoreARRegressorCalls(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)
        self.n_signals = 4
        self.n_samples = 41
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]

        self.regressors = []
        for i in range(self.n_signals):
            regressor = Mock(n_features=4)
            regressor.fit_infer.return_value = (
                np.zeros((self.n_samples, self.n_models)),
                SimpleNamespace(),
            )
            self.regressors.append(regressor)

        self.regressor_class = Mock(side_effect=self.regressors)
        self.metric = lambda x, y: 1.0
        self.kwargs = {"foo": "bar"}
        self.fit_kws = {"bar": 3}

        with mock.patch("bioslds.batch.fit_infer_ar") as mock_fit_infer_ar:
            mock_fit_infer_ar.side_effect = [
                _.fit_infer.return_value for _ in self.regressors
            ]
            hyper_score_ar(
                self.regressor_class,
                self.dataset,
                self.metric,
                fit_kws=self.fit_kws,
                **self.kwargs
            )

            self.mock_fit_infer_ar = mock_fit_infer_ar

        for call in self.regressor_class.call_args_list:
            for key, value in self.kwargs.items():
                self.assertIn(key, call[1])
                self.assertEqual(call[1][key], value)

    def test_regressor_class_called_n_signals_times(self):
        self.assertEqual(self.regressor_class.call_count, self.n_signals)

    def test_additional_kwargs_passed_to_regressor_class_constructor(self):
        self.assertTrue(self.regressor_class.called)

    def test_no_positional_args_passed_to_regressor_class_constructor(self):
        for call in self.regressor_class.call_args_list:
            self.assertEqual(len(call[0]), 0)

    def test_rng_passed_to_regressor_class_constructor(self):
        self.assertTrue(self.regressor_class.called)
        for call in self.regressor_class.call_args_list:
            self.assertIn("rng", call[1])

    def test_fit_infer_ar_called_once_for_each_regressor(self):
        self.assertEqual(self.mock_fit_infer_ar.call_count, self.n_signals)
        for i, call in enumerate(self.mock_fit_infer_ar.call_args_list):
            self.assertIs(call[0][0], self.regressors[i])

    def test_fit_infer_ar_called_once_for_each_signal(self):
        self.assertEqual(self.mock_fit_infer_ar.call_count, self.n_signals)
        for i, call in enumerate(self.mock_fit_infer_ar.call_args_list):
            self.assertIs(call[0][1], self.dataset[i].y)

    def test_fit_kws_are_forwarded_to_fit_infer_ar(self):
        self.assertTrue(self.mock_fit_infer_ar.called)
        for call in self.mock_fit_infer_ar.call_args_list:
            for key, value in self.fit_kws.items():
                self.assertIn(key, call[1])
                self.assertEqual(call[1][key], value)


class TestHyperScoreARMetricCalls(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)
        self.n_signals = 4
        self.n_samples = 35
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]

        self.regressors = []
        for i in range(self.n_signals):
            regressor = Mock(n_features=4)
            regressor.fit_infer.return_value = (
                self.rng.uniform(size=(self.n_samples, self.n_models)),
                SimpleNamespace(),
            )
            self.regressors.append(regressor)

        self.regressor_class = Mock(side_effect=self.regressors)
        self.metric_outputs = np.linspace(0, 1, self.n_signals)
        self.metric = Mock(side_effect=self.metric_outputs)

        # noinspection PyTypeChecker
        self.summary, self.details = hyper_score_ar(
            self.regressor_class, self.dataset, self.metric, test_fraction=1.0
        )

    def test_metric_is_called_with_usage_seq_as_first_arg(self):
        self.assertEqual(self.metric.call_count, self.n_signals)
        for i, call in enumerate(self.metric.call_args_list):
            np.testing.assert_equal(call[0][0], self.dataset[i].usage_seq)

    def test_metric_is_called_with_argmaxed_infer_result_as_second_arg(self):
        self.assertEqual(self.metric.call_count, self.n_signals)
        for i, call in enumerate(self.metric.call_args_list):
            crt_r = self.regressors[i].fit_infer.return_value[0]
            crt_inferred_usage = crt_r.argmax(axis=1)
            np.testing.assert_equal(call[0][1], crt_inferred_usage)

    def test_first_output_is_median_of_scores_from_metric(self):
        # noinspection PyTypeChecker
        expected: float = np.median(self.metric_outputs)
        self.assertAlmostEqual(self.summary, expected)

    def test_second_output_trial_scores_are_correct(self):
        np.testing.assert_equal(self.details.trial_scores, self.metric_outputs)


class TestHyperScoreARRng(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n_signals = 4
        self.n_samples = 35
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]
        self.metric = lambda x, y: 1.0

    def check_regressor_calls(
        self, rng: Union[None, int, np.random.RandomState, np.random.Generator]
    ) -> Tuple[Mock, Tuple]:
        regressor = Mock(n_features=4)
        regressor.fit_infer.return_value = (
            self.rng.uniform(size=(self.n_samples, self.n_models)),
            SimpleNamespace(),
        )

        regressor_class = Mock(return_value=regressor)

        kwargs = {}
        if rng is not None:
            kwargs["rng"] = rng
        res = hyper_score_ar(regressor_class, self.dataset, self.metric, **kwargs)
        return regressor_class, res

    def compare_rngs_from_regressor_calls(
        self, calls1: Sequence, calls2: Sequence, n_samples: int = 10
    ) -> bool:
        """ Compare random number generators from two sets of regressor calls.

        This looks at the "rng" keyword in each of the calls, and uses that random
        number generator to generate a short sequence of numbers. If the numbers
        returned from the rng from a call in `calls1` is different from those obtained
        from the corresponding entry in `calls2`, the function returns `False`. If
        everything matches, it returns `True`.
        """
        self.assertEqual(len(calls1), len(calls2))
        for call1, call2 in zip(calls1, calls2):
            rng1 = call1[1]["rng"]
            rng2 = call2[1]["rng"]

            v1 = rng1.normal(size=n_samples)
            v2 = rng2.normal(size=n_samples)

            if not np.allclose(v1, v2):
                return False

        return True

    def test_int_seed_uses_default_rng(self):
        seed = 14
        regressors1 = self.check_regressor_calls(seed)[0]
        regressors2 = self.check_regressor_calls(np.random.default_rng(seed))[0]

        calls1 = regressors1.call_args_list
        calls2 = regressors2.call_args_list

        self.assertTrue(self.compare_rngs_from_regressor_calls(calls1, calls2))

    def test_int_default_seed_is_zero(self):
        regressors1 = self.check_regressor_calls(None)[0]
        regressors2 = self.check_regressor_calls(0)[0]

        calls1 = regressors1.call_args_list
        calls2 = regressors2.call_args_list

        self.assertTrue(self.compare_rngs_from_regressor_calls(calls1, calls2))

    def test_works_with_random_state(self):
        seed = 14
        regressors1 = self.check_regressor_calls(np.random.RandomState(seed))[0]
        regressors2 = self.check_regressor_calls(np.random.RandomState(seed))[0]

        calls1 = regressors1.call_args_list
        calls2 = regressors2.call_args_list

        self.assertTrue(self.compare_rngs_from_regressor_calls(calls1, calls2))

    def test_different_seeds_yield_different_results(self):
        regressors1 = self.check_regressor_calls(1)[0]
        regressors2 = self.check_regressor_calls(2)[0]

        calls1 = regressors1.call_args_list
        calls2 = regressors2.call_args_list

        self.assertFalse(self.compare_rngs_from_regressor_calls(calls1, calls2))

    def test_regressor_seeds_output(self):
        regressors, res = self.check_regressor_calls(0)
        calls = regressors.call_args_list

        self.assertEqual(len(res[1].regressor_seeds), self.n_signals)
        n_samples = 10
        for call, seed in zip(calls, res[1].regressor_seeds):
            rng_exp = np.random.default_rng(seed)
            rng = call[1]["rng"]

            v_exp = rng_exp.normal(size=n_samples)
            v = rng.normal(size=n_samples)

            np.testing.assert_equal(v, v_exp)


class TestHyperScoreARTestAmount(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(5)
        self.n_signals = 4
        self.n_samples = 100
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]

        self.regressors = []
        for i in range(self.n_signals):
            regressor = Mock(n_features=4)
            regressor.fit_infer.return_value = (
                self.rng.uniform(size=(self.n_samples, self.n_models)),
                SimpleNamespace(),
            )
            self.regressors.append(regressor)

        self.regressor_class = Mock(side_effect=self.regressors)
        self.metric = Mock(side_effect=lambda x, y: np.mean(x - y))

    def test_appropriate_test_fraction_is_passed_to_metric(self):
        fraction = 0.6
        # noinspection PyTypeChecker
        hyper_score_ar(
            self.regressor_class, self.dataset, self.metric, test_fraction=fraction
        )

        call = self.metric.call_args_list[0]
        usage0 = self.dataset[0].usage_seq
        r0 = self.regressors[0].fit_infer.return_value[0]
        inferred0 = r0.argmax(axis=1)

        n = int(fraction * len(r0))
        labels_true_exp = usage0[-n:]
        labels_pred_exp = inferred0[-n:]

        np.testing.assert_equal(call[0][0], labels_true_exp)
        np.testing.assert_equal(call[0][1], labels_pred_exp)

    def test_appropriate_test_samples_are_passed_to_metric(self):
        samples = 23
        # noinspection PyTypeChecker
        hyper_score_ar(
            self.regressor_class, self.dataset, self.metric, test_samples=samples
        )

        call = self.metric.call_args_list[1]
        usage1 = self.dataset[1].usage_seq
        r1 = self.regressors[1].fit_infer.return_value[0]
        inferred1 = r1.argmax(axis=1)

        labels_true_exp = usage1[-samples:]
        labels_pred_exp = inferred1[-samples:]

        np.testing.assert_equal(call[0][0], labels_true_exp)
        np.testing.assert_equal(call[0][1], labels_pred_exp)

    def test_default_test_fraction_is_twenty_percent(self):
        # noinspection PyTypeChecker
        res_def = hyper_score_ar(self.regressor_class, self.dataset, self.metric)[0]

        regressor_class = Mock(side_effect=self.regressors)
        # noinspection PyTypeChecker
        res_20 = hyper_score_ar(
            regressor_class, self.dataset, self.metric, test_fraction=0.2
        )[0]

        regressor_class = Mock(side_effect=self.regressors)
        # noinspection PyTypeChecker
        res_50 = hyper_score_ar(
            regressor_class, self.dataset, self.metric, test_fraction=0.5
        )[0]

        self.assertNotEqual(res_20, res_50)
        self.assertEqual(res_def, res_20)


class TestHyperScoreARProgress(unittest.TestCase):
    def test_progress_called(self):
        mock_progress = mock.MagicMock()

        hyper_score_ar(
            MockRegressor,
            [SimpleNamespace(y=[1, 2], usage_seq=[0, 0])],
            lambda x, y: 1.0,
            n_models=2,
            n_features=3,
            progress=mock_progress,
        )
        mock_progress.assert_called()

    def test_progress_trial_passed_to_fit_infer_ra(self):
        regressor = Mock(n_features=4)
        regressor.fit_infer.return_value = (np.zeros((10, 2)), SimpleNamespace())
        regressor_class = Mock(return_value=regressor)

        mock_progress = Mock()
        hyper_score_ar(
            regressor_class,
            [SimpleNamespace(y=[1, 2], usage_seq=[0, 0])],
            lambda x, y: 1.0,
            n_models=2,
            n_features=3,
            progress_trial=mock_progress,
        )

        regressor.fit_infer.assert_called()
        call = regressor.fit_infer.call_args_list[0]
        self.assertIn("progress", call[1])
        self.assertIs(call[1]["progress"], mock_progress)


class TestHyperScoreARMonitor(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n_signals = 4
        self.n_samples = 40
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=np.zeros(self.n_samples), usage_seq=np.zeros(self.n_samples),
            )
            for _ in range(self.n_signals)
        ]

        self.r = [
            self.rng.uniform(size=(self.n_samples, self.n_models))
            for _ in range(self.n_signals)
        ]
        self.history_out = [{"foo": "bar", "bar": _} for _ in range(self.n_signals)]
        self.regressors = []
        for i in range(self.n_signals):
            regressor = Mock(n_features=4)
            regressor.fit_infer.return_value = (
                self.r[i],
                SimpleNamespace(**self.history_out[i]),
            )
            self.regressors.append(regressor)

        self.regressor_class = Mock(side_effect=self.regressors)
        self.metric = lambda x, y: 1.0

    def test_monitor_sequence_passed_to_fit_infer_when_step_is_default(self):
        monitor = ["a", "b"]
        hyper_score_ar(self.regressor_class, self.dataset, self.metric, monitor=monitor)

        self.regressors[0].fit_infer.assert_called()
        call = self.regressors[0].fit_infer.call_args_list[0]
        self.assertIn("monitor", call[1])
        self.assertEqual(call[1]["monitor"], monitor)

    def test_when_monitor_step_is_not_one_an_attribute_monitor_is_created(self):
        monitor = ["a", "b"]
        step = 3

        with mock.patch("bioslds.batch.AttributeMonitor") as MockAttributeMonitor:
            MockAttributeMonitor.history = SimpleNamespace()
            hyper_score_ar(
                self.regressor_class,
                self.dataset,
                self.metric,
                monitor=monitor,
                monitor_step=step,
            )
            MockAttributeMonitor.assert_called()
            call = MockAttributeMonitor.call_args_list[0]

            self.assertEqual(len(call[0]), 1)
            self.assertEqual(call[0][0], monitor)

            self.assertIn("step", call[1])
            self.assertEqual(call[1]["step"], step)

    def test_fit_infer_details_has_history_for_each_signal(self):
        _, details = hyper_score_ar(self.regressor_class, self.dataset, self.metric)
        self.assertTrue(hasattr(details, "history"))
        self.assertEqual(len(details.history), self.n_signals)

    def test_fit_infer_second_output_is_returned_in_details_history(self):
        _, details = hyper_score_ar(self.regressor_class, self.dataset, self.metric)
        for i, history in enumerate(details.history):
            for key, value in self.history_out[i].items():
                self.assertTrue(hasattr(history, key))
                self.assertEqual(getattr(history, key), value)

    def test_r_is_special_value_in_monitor(self):
        _, details = hyper_score_ar(
            self.regressor_class, self.dataset, self.metric, monitor="r",
        )
        for i, history in enumerate(details.history):
            self.assertTrue(hasattr(history, "r"))
            np.testing.assert_allclose(history.r, self.r[i])

    def test_r_monitor_with_nontrivial_step(self):
        step = 3
        _, details = hyper_score_ar(
            self.regressor_class,
            self.dataset,
            self.metric,
            monitor="r",
            monitor_step=step,
        )
        for i, history in enumerate(details.history):
            self.assertTrue(hasattr(history, "r"))
            np.testing.assert_allclose(history.r, self.r[i][::step])


class TestHyperScoreARRegressorDetails(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.n_signals = 4
        self.n_samples = 35
        self.n_models = 3
        self.dataset = [
            SimpleNamespace(
                y=self.rng.normal(size=self.n_samples),
                usage_seq=self.rng.integers(0, self.n_models, size=self.n_samples),
            )
            for _ in range(self.n_signals)
        ]
        self.metric = lambda x, y: 1.0

        self.regressors = [Mock(n_features=4) for _ in range(self.n_signals)]
        for regressor in self.regressors:
            regressor.fit_infer.return_value = (
                self.rng.uniform(size=(self.n_samples, self.n_models)),
                SimpleNamespace(),
            )

        self.regressor_class = Mock(side_effect=self.regressors)
        self.res = hyper_score_ar(self.regressor_class, self.dataset, self.metric)

    def test_regressors_are_returned_in_details(self):
        self.assertTrue(hasattr(self.res[1], "regressors"))

    def test_returned_regressors_are_correct(self):
        for regressor, regressor_exp in zip(self.res[1].regressors, self.regressors):
            self.assertIs(regressor, regressor_exp)


if __name__ == "__main__":
    unittest.main()
