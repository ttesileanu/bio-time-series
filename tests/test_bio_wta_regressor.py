import unittest

import numpy as np
import warnings

from unittest import mock
from bioslds.regressors import BioWTARegressor
from bioslds.monitor import AttributeMonitor


class TestBioWTARegressorInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.wta = BioWTARegressor(self.n_models, self.n_features)

    def test_n_models_attribute(self):
        self.assertEqual(self.wta.n_models, self.n_models)

    def test_n_components_attribute(self):
        self.assertTrue(hasattr(self.wta, "n_components"))
        self.assertEqual(self.wta.n_components, self.n_models)

    def test_n_features_attribute(self):
        self.assertEqual(self.wta.n_features, self.n_features)

    def test_default_rate(self):
        np.testing.assert_allclose(self.wta.rate, 1e-3)

    def test_default_weights_shape(self):
        self.assertEqual(np.ndim(self.wta.weights_), 2)

        np.testing.assert_equal(
            np.shape(self.wta.weights_), [self.n_models, self.n_features]
        )


class TestBioWTARegressorInitSetWeights(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 2
        self.args = (self.n_models, self.n_features)

    def test_set_custom_weights(self):
        weights = [[0, 1], [0.5, 2.3], [-0.1, 0.2], [0.3, -0.5]]
        wta = BioWTARegressor(*self.args, weights=weights)

        np.testing.assert_allclose(wta.weights_, weights)

    def test_set_random_weights_change_if_done_again(self):
        rng = np.random.default_rng(1)
        wta1 = BioWTARegressor(*self.args, rng=rng)
        wta2 = BioWTARegressor(*self.args, rng=rng)

        self.assertGreater(np.max(np.abs(wta1.weights_ - wta2.weights_)), 1e-4)

    def test_set_random_weights_with_integer_seed_uses_default_rng(self):
        seed = 13
        wta1 = BioWTARegressor(*self.args, rng=seed)
        wta2 = BioWTARegressor(*self.args, rng=np.random.default_rng(seed))

        np.testing.assert_allclose(wta1.weights_, wta2.weights_)

    def test_default_random_seed_is_zero(self):
        wta1 = BioWTARegressor(*self.args, rng=0)
        wta2 = BioWTARegressor(*self.args)

        np.testing.assert_allclose(wta1.weights_, wta2.weights_)


class TestBioWTARegressorTransformDefaultInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 5
        self.wta = BioWTARegressor(self.n_models, self.n_features)

        self.rng = np.random.default_rng(1)
        self.n_samples = 85
        self.predictors = self.rng.uniform(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.uniform(size=self.n_samples)

    def test_output_shape(self):
        r = self.wta.transform(self.predictors, self.dependent)

        np.testing.assert_equal(np.shape(r), (self.n_samples, self.n_models))

    def test_every_output_row_sums_to_one(self):
        r = self.wta.transform(self.predictors, self.dependent)
        sums = np.sum(r, axis=1)

        np.testing.assert_allclose(sums, 1)

    def test_output_is_argmin_of_squared_error_first_sample(self):
        ini_weights = np.copy(self.wta.weights_)
        x = self.predictors[[0], :]
        y = self.dependent[[0]]
        r = self.wta.transform(x, y)

        err = np.zeros(self.n_models)
        for i in range(self.n_models):
            w = ini_weights[i]
            err[i] = y[0] - np.dot(w, x[0])

        r_exp = np.zeros(self.n_models)
        r_exp[np.argmin(err)] = 1.0

        np.testing.assert_allclose(r[0], r_exp)

    def test_output_is_argmin_of_squared_monitored_error(self):
        r, history = self.wta.transform(
            self.predictors, self.dependent, monitor=["error_"]
        )

        r_exp = np.zeros((self.n_samples, self.n_models))
        for k in range(self.n_samples):
            r_exp[k, np.argmin(np.abs(history.error_[k]))] = 1.0

        np.testing.assert_allclose(r, r_exp)

    def test_raises_if_size_of_X_is_not_right(self):
        n_samples = 5
        with self.assertRaises(ValueError):
            self.wta.transform(np.zeros((n_samples, 2)), np.zeros(n_samples))

    def test_output_of_repeated_calls_to_transform_equivalent_to_single_call(self):
        n1 = 4 * self.n_samples // 7
        r1 = self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        r2 = self.wta.transform(self.predictors[n1:], self.dependent[n1:])

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        r = wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r, np.vstack((r1, r2)))

    def test_weight_change_from_repeated_transform_calls_equivalent_to_one_call(self):
        n1 = 2 * self.n_samples // 7
        n2 = 3 * self.n_samples // 7
        self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        self.wta.transform(self.predictors[n1 : n1 + n2], self.dependent[n1 : n1 + n2])
        self.wta.transform(self.predictors[n1 + n2 :], self.dependent[n1 + n2 :])

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(self.wta.weights_, wta_again.weights_)

    def test_only_weights_for_best_fitting_model_change(self):
        all_ks = []
        for crt_x, crt_y in zip(self.predictors, self.dependent):
            weights0 = np.copy(self.wta.weights_)
            r = self.wta.transform([crt_x], [crt_y])[0]

            k = np.argmax(r)
            all_ks.append(k)
            dweights = self.wta.weights_ - weights0

            # check that the kth weights got updated
            self.assertGreater(np.max(np.abs(dweights[k])), 1e-7)

            # check that all other weights stayed the same
            dweights[k] = 0
            np.testing.assert_allclose(dweights, 0)

        # make sure we've seen several different models get updated
        self.assertGreater(len(np.unique(all_ks)), 0)

    def test_weight_update_is_correct(self):
        weights0 = np.copy(self.wta.weights_)
        r = self.wta.transform(self.predictors[[0]], self.dependent[[0]])
        k = np.argmax(r)

        dw = (self.wta.weights_ - weights0)[k]
        eps_k = self.dependent[0] - np.dot(weights0[k], self.predictors[0])
        dw_exp = self.wta.rate * self.predictors[0] * eps_k

        np.testing.assert_allclose(dw, dw_exp)

    def test_monitor_as_sequence(self):
        _, history = self.wta.transform(
            self.predictors,
            self.dependent,
            monitor=["weights_", "prediction_"],
            return_history=True,
        )

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        weights = []
        predictions = []
        for crt_x, crt_y in zip(self.predictors, self.dependent):
            weights.append(np.copy(wta_again.weights_))

            crt_all_pred = np.dot(wta_again.weights_, crt_x)
            crt_r = wta_again.transform([crt_x], [crt_y])
            crt_k = np.argmax(crt_r)

            crt_pred = crt_all_pred[crt_k]
            predictions.append(crt_pred)

        np.testing.assert_allclose(weights, history.weights_)
        np.testing.assert_allclose(predictions, history.prediction_)

    def test_monitor_as_object(self):
        names = ["weights_"]
        monitor = AttributeMonitor(names)
        self.wta.transform(self.predictors, self.dependent, monitor=monitor)

        wta_alt = BioWTARegressor(self.n_models, self.n_features)
        _, history_alt = wta_alt.transform(
            self.predictors, self.dependent, monitor=names
        )

        np.testing.assert_allclose(monitor.history_.weights_, history_alt.weights_)

    def test_when_monitor_is_object_history_returned_is_its_attrib(self):
        names = ["weights_"]
        monitor = AttributeMonitor(names)
        _, history = self.wta.transform(
            self.predictors, self.dependent, monitor=monitor, return_history=True
        )

        self.assertIs(history, monitor.history_)

    def test_by_default_history_is_not_returned(self):
        monitor = AttributeMonitor(["weights_"])
        res = self.wta.transform(self.predictors, self.dependent, monitor=monitor)

        self.assertEqual(np.shape(res), (self.n_samples, self.n_models))

    def test_history_is_returned_when_monitor_is_sequence(self):
        res = self.wta.transform(self.predictors, self.dependent, monitor=["weights_"])

        self.assertEqual(len(res), 2)

    def test_history_same_when_chunk_hint_changes(self):
        names = ["prediction_"]
        _, history = self.wta.transform(
            self.predictors,
            self.dependent,
            monitor=names,
            chunk_hint=1000,
            return_history=True,
        )

        wta_alt = BioWTARegressor(self.n_models, self.n_features)
        _, history_alt = wta_alt.transform(
            self.predictors,
            self.dependent,
            monitor=names,
            chunk_hint=1,
            return_history=True,
        )

        np.testing.assert_allclose(history.prediction_, history_alt.prediction_)

    def test_monitor_output_matches_transform_retval(self):
        r, history = self.wta.transform(
            self.predictors, self.dependent, monitor=["output_"], return_history=True
        )
        self.assertTrue(hasattr(history, "output_"))
        np.testing.assert_allclose(history.output_, r)

    def test_output_attribute_matches_transform_retval(self):
        r = self.wta.transform(self.predictors[:-1], self.dependent[:-1])
        np.testing.assert_allclose(r[-1], self.wta.output_)

    def test_initial_output_is_all_zeros(self):
        self.assertEqual(len(self.wta.output_), self.n_models)
        np.testing.assert_allclose(self.wta.output_, 0)

    def test_return_history_ignored_when_monitor_is_none(self):
        res = self.wta.transform(self.predictors, self.dependent, return_history=True)
        self.assertEqual(np.shape(res), (self.n_samples, self.n_models))

    def test_progress_called(self):
        mock_progress = mock.MagicMock()

        self.wta.transform(self.predictors, self.dependent, progress=mock_progress)

        mock_progress.assert_called()

    def test_prediction_is_based_on_hard_clustering_by_default(self):
        weights0 = np.copy(self.wta.weights_)
        r, history = self.wta.transform(
            self.predictors[[0]], self.dependent[[0]], monitor=["prediction_"]
        )
        k = np.argmax(r[0])

        pred = weights0 @ self.predictors[0]
        self.assertAlmostEqual(history.prediction_[0], pred[k])


class TestBioWTARegressorTransformTemperature(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 5
        self.temperature = 1.0
        self.wta = BioWTARegressor(
            self.n_models, self.n_features, temperature=self.temperature
        )

        self.rng = np.random.default_rng(1)
        self.n_samples = 85
        self.predictors = self.rng.uniform(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.uniform(size=self.n_samples)

    def test_output_is_softmax_of_minus_half_squared_error(self):
        ini_weights = np.copy(self.wta.weights_)
        x = self.predictors[[0], :]
        y = self.dependent[[0]]
        r = self.wta.transform(x, y)

        err = np.zeros(self.n_models)
        for i in range(self.n_models):
            w = ini_weights[i]
            err[i] = y[0] - np.dot(w, x[0])

        np.testing.assert_allclose(
            np.log(r[0]) - np.log(r[0][0]), -0.5 * (err ** 2 - err[0] ** 2)
        )

    def test_every_output_row_sums_to_one(self):
        r = self.wta.transform(self.predictors, self.dependent)
        sums = np.sum(r, axis=1)

        np.testing.assert_allclose(sums, 1)

    def test_output_of_repeated_calls_to_transform_equivalent_to_single_call(self):
        n1 = 4 * self.n_samples // 7
        r1 = self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        r2 = self.wta.transform(self.predictors[n1:], self.dependent[n1:])

        wta_again = BioWTARegressor(
            self.n_models, self.n_features, temperature=self.temperature
        )
        r = wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r, np.vstack((r1, r2)))

    def test_weight_change_from_repeated_transform_calls_equivalent_to_one_call(self):
        n1 = 2 * self.n_samples // 7
        n2 = 3 * self.n_samples // 7
        self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        self.wta.transform(self.predictors[n1 : n1 + n2], self.dependent[n1 : n1 + n2])
        self.wta.transform(self.predictors[n1 + n2 :], self.dependent[n1 + n2 :])

        wta_again = BioWTARegressor(
            self.n_models, self.n_features, temperature=self.temperature
        )
        wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(self.wta.weights_, wta_again.weights_)

    def test_weight_update_is_correct(self):
        weights0 = np.copy(self.wta.weights_)
        r = self.wta.transform(self.predictors[[0]], self.dependent[[0]])

        dw = self.wta.weights_ - weights0
        eps = np.zeros(self.n_models)
        for k in range(self.n_models):
            eps[k] = self.dependent[0] - np.dot(weights0[k], self.predictors[0])
        dw_exp = self.wta.rate * np.outer(r[0] * eps, self.predictors[0])

        np.testing.assert_allclose(dw, dw_exp)

    def test_log_r_prop_temperature(self):
        r = self.wta.transform(self.predictors[[0]], self.dependent[[0]])

        temperature_again = 3.2
        wta_again = BioWTARegressor(
            self.n_models, self.n_features, temperature=temperature_again
        )
        r_again = wta_again.transform(self.predictors[[0]], self.dependent[[0]])

        d_logr = np.log(r[0]) - np.log(r[0, 0])
        d_logr_again = np.log(r_again[0]) - np.log(r_again[0, 0])

        np.testing.assert_allclose(
            d_logr_again, d_logr * self.temperature / temperature_again
        )

    def test_monitor_output_matches_half_error_squared_when_temperature_is_one(self):
        r, history = self.wta.transform(
            self.predictors, self.dependent, monitor=["error_"], return_history=True
        )
        self.assertTrue(hasattr(history, "error_"))

        log_r = np.log(r)
        log_r_norm = log_r - log_r[:, 0][:, None]

        log_r_exp = -0.5 * history.error_ ** 2
        log_r_norm_exp = log_r_exp - log_r_exp[:, 0][:, None]
        np.testing.assert_allclose(log_r_norm_exp, log_r_norm)

    def test_prediction_is_weighted(self):
        weights0 = np.copy(self.wta.weights_)
        r, history = self.wta.transform(
            self.predictors[[0]], self.dependent[[0]], monitor=["prediction_"]
        )
        k = np.argmax(r[0])

        pred = weights0 @ self.predictors[0]
        self.assertNotAlmostEqual(history.prediction_[0], pred[k])
        # noinspection PyTypeChecker
        self.assertAlmostEqual(history.prediction_[0], np.dot(r[0], pred))


class TestBioWTARegressorStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 5
        self.wta = BioWTARegressor(self.n_models, self.n_features)

    def test_str(self):
        s = str(self.wta)
        s_exp = (
            f"BioWTARegressor(n_models={self.n_models}, n_features="
            f"{self.n_features})"
        )

        self.assertEqual(s, s_exp)

    def test_repr(self):
        s = repr(self.wta)
        s_exp = (
            f"BioWTARegressor(n_models={self.n_models}, n_features="
            f"{self.n_features}, rate={self.wta.rate}, weights_="
            f"{repr(self.wta.weights_)})"
        )

        self.assertEqual(s, s_exp)


class TestBioWTARegressorLatentPrior(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.wta = BioWTARegressor(self.n_models, self.n_features)

        self.rng = np.random.default_rng(10)
        self.n_samples = 79
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

    def test_default_initial_latent_state_distribution_is_uniform(self):
        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            start_prob=np.ones(self.n_models) / self.n_models,
        )

        r1 = self.wta.transform(self.predictors, self.dependent)
        r2 = wta2.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)

    def test_default_latent_state_transition_matrix_is_uniform(self):
        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            trans_mat=np.ones((self.n_models, self.n_models)) / self.n_models,
        )

        r1 = self.wta.transform(self.predictors, self.dependent)
        r2 = wta2.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)

    def test_changing_initial_latent_state_distribution_changes_output(self):
        r1 = self.wta.transform(self.predictors, self.dependent)
        k = np.argmax(r1[0])

        start_prob = self.rng.uniform(size=self.n_models)
        start_prob[k] /= 50

        start_prob = start_prob / np.sum(start_prob)
        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        r2 = wta2.transform(self.predictors, self.dependent)

        self.assertGreater(np.max(np.abs(r1 - r2)), 1e-3)

    def test_changing_latent_state_transition_matrix_changes_output(self):
        trans_mat = [
            (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]
        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        r1 = self.wta.transform(self.predictors, self.dependent)
        r2 = wta2.transform(self.predictors, self.dependent)

        self.assertGreater(np.max(np.abs(r1 - r2)), 1e-3)

    def test_when_start_prob_large_for_a_state_then_that_state_gets_high_r(self):
        r1 = self.wta.transform(self.predictors[[0]], self.dependent[[0]])
        k1 = np.argmax(r1[0])

        # make another state win out
        k = 0 if k1 != 0 else 1
        p_large = 0.999
        start_prob = (1 - p_large) * np.ones(self.n_models) / (self.n_models - 1)
        start_prob[k] = p_large
        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        r2 = wta2.transform(self.predictors[[0]], self.dependent[[0]])
        k2 = np.argmax(r2[0])

        self.assertNotEqual(k1, k2)
        self.assertEqual(k2, k)

    def test_when_trans_mat_large_for_a_target_state_then_that_state_gets_high_r(self):
        # first figure out where we're starting
        r1 = self.wta.transform(self.predictors[:2], self.dependent[:2])
        k1_start = np.argmax(r1[0])
        k1 = np.argmax(r1[1])

        # make another state win out
        k = 0 if k1 != 0 else 1
        p0 = 0.999
        trans_mat = np.ones((self.n_models, self.n_models)) / self.n_models
        trans_mat[k1_start, :] = (1 - p0) * np.ones(self.n_models) / (self.n_models - 1)
        trans_mat[k1_start, k] = p0
        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        r2 = wta2.transform(self.predictors[:2], self.dependent[:2])
        k2_start = np.argmax(r2[0])
        k2 = np.argmax(r2[1])

        self.assertEqual(k1_start, k2_start)

        self.assertNotEqual(k1, k2)
        self.assertEqual(k2, k)

    def test_float_trans_mat(self):
        r1 = self.wta.transform(self.predictors, self.dependent)

        wta2 = BioWTARegressor(
            self.n_models, self.n_features, trans_mat=1 / self.n_models
        )
        r2 = wta2.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)


class TestBioWTARegressorLatentPriorTemperatureOne(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.temperature = 1.0
        self.wta = BioWTARegressor(
            self.n_models, self.n_features, temperature=self.temperature
        )

        self.rng = np.random.default_rng(10)
        self.n_samples = 79
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

    def test_initial_r_value_changes_by_correct_amount_according_to_start_prob(self):
        r1 = self.wta.transform(self.predictors[[0]], self.dependent[[0]])

        # change initial state distribution
        start_prob = (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            temperature=self.temperature,
            start_prob=start_prob,
        )

        r2 = wta2.transform(self.predictors[[0]], self.dependent[[0]])

        diff_log_r = np.log(r2[0]) - np.log(r1[0])
        np.testing.assert_allclose(
            diff_log_r - diff_log_r[0], np.log(start_prob) - np.log(start_prob[0]),
        )

    def test_second_r_value_changes_by_correct_amount_according_to_trans_mat(self):
        r1 = self.wta.transform(self.predictors[:2], self.dependent[:2])

        # change transition matrix
        trans_mat = [
            (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]
        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            temperature=self.temperature,
            trans_mat=trans_mat,
        )

        r2 = wta2.transform(self.predictors[:2], self.dependent[:2])
        np.testing.assert_allclose(r1[0], r2[0])

        diff_log_r = np.log(r2[1]) - np.log(r1[1])
        expected_diff_log_r = r2[0] @ np.log(trans_mat)
        np.testing.assert_allclose(
            diff_log_r - diff_log_r[0], expected_diff_log_r - expected_diff_log_r[0]
        )


class TestBioWTARegressorArbitraryStartProbAndTransMat(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.rng = np.random.default_rng(9)

        def normalize_v(v: np.ndarray) -> np.ndarray:
            return v / np.sum(v)

        self.start_prob = normalize_v(self.rng.uniform(size=self.n_models))
        self.trans_mat = [
            normalize_v(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]

        self.kwargs = {
            "n_models": self.n_models,
            "n_features": self.n_features,
            "start_prob": self.start_prob,
            "trans_mat": self.trans_mat,
        }
        self.wta = BioWTARegressor(**self.kwargs)

        self.n_samples = 79
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

    def test_output_of_repeated_calls_to_transform_equivalent_to_single_call(self):
        n1 = 4 * self.n_samples // 7
        r1 = self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        r2 = self.wta.transform(self.predictors[n1:], self.dependent[n1:])

        wta_again = BioWTARegressor(**self.kwargs)
        r = wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(r, np.vstack((r1, r2)))

    def test_weight_change_from_repeated_transform_calls_equivalent_to_one_call(self):
        n1 = 2 * self.n_samples // 7
        n2 = 3 * self.n_samples // 7
        self.wta.transform(self.predictors[:n1], self.dependent[:n1])
        self.wta.transform(self.predictors[n1 : n1 + n2], self.dependent[n1 : n1 + n2])
        self.wta.transform(self.predictors[n1 + n2 :], self.dependent[n1 + n2 :])

        wta_again = BioWTARegressor(**self.kwargs)
        wta_again.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(self.wta.weights_, wta_again.weights_)


class TestBioWTARegressorDegenerateStartProbOrTransMat(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 3

        self.rng = np.random.default_rng(31)
        self.n_samples = 79
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

    def test_state_with_zero_start_prob_is_not_used_at_first_step(self):
        # first find the most likely state with uniform start_prob
        wta1 = BioWTARegressor(self.n_models, self.n_features)
        r1 = wta1.transform(self.predictors, self.dependent)
        k0 = np.argmax(r1[0])

        # next disallow the system from starting in that state
        start_prob = np.ones(self.n_models) / (self.n_models - 1)
        # we never start in state k0
        start_prob[k0] = 0

        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)
        r2 = wta2.transform(self.predictors, self.dependent)

        self.assertLess(r2[0, k0], 1e-6)

    def test_no_warning_when_some_start_prob_are_zero(self):
        start_prob = np.ones(self.n_models) / (self.n_models - 2)

        start_prob[[0, 2]] = 0
        wta = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        with warnings.catch_warnings(record=True) as warn_list:
            # ensure warnings haven't been disabled
            warnings.simplefilter("always")
            wta.transform(self.predictors, self.dependent)

            # ensure no warnings have been triggered
            self.assertEqual(len(warn_list), 0)

    def test_transition_with_zero_trans_mat_is_never_used(self):
        # first find most likely transition with uniform trans_mat
        wta1 = BioWTARegressor(self.n_models, self.n_features)
        r1 = wta1.transform(self.predictors, self.dependent)
        k1 = np.argmax(r1, axis=1)

        trans_count1 = np.zeros((self.n_models, self.n_models), dtype=int)
        for ki, kf in zip(k1, k1[1:]):
            trans_count1[ki, kf] += 1

        kmli, kmlf = np.unravel_index(trans_count1.argmax(), trans_count1.shape)
        self.assertGreater(trans_count1[kmli, kmlf], 0)

        # next disallow the system performing that transition
        trans_mat = np.ones((self.n_models, self.n_models)) / self.n_models
        trans_mat[kmli] = np.ones(self.n_models) / (self.n_models - 1)
        trans_mat[kmli, kmlf] = 0

        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)
        r2 = wta2.transform(self.predictors, self.dependent)
        k2 = np.argmax(r2, axis=1)

        trans_count2 = np.zeros((self.n_models, self.n_models), dtype=int)
        for ki, kf in zip(k2, k2[1:]):
            trans_count2[ki, kf] += 1

        self.assertEqual(trans_count2[kmli, kmlf], 0)

    def test_no_warning_when_some_trans_mat_are_zero(self):
        # disallow the system from performing some transition
        trans_mat = np.ones((self.n_models, self.n_models)) / self.n_models
        trans_mat[0] = np.ones(self.n_models) / (self.n_models - 1)
        trans_mat[0, 1] = 0
        wta = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        with warnings.catch_warnings(record=True) as warn_list:
            # ensure warnings haven't been disabled
            warnings.simplefilter("always")
            wta.transform(self.predictors, self.dependent)

            # ensure no warnings have been triggered
            self.assertEqual(len(warn_list), 0)


class TestBioWTARegressorChunkHintDoesNotAffectResult(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 3

        self.rng = np.random.default_rng(30)
        self.n_samples = 125
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

        self.start_prob = (lambda v: v / np.sum(v))(
            self.rng.uniform(size=self.n_models)
        )
        self.trans_mat = [
            (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]

    def test_small_chunk_same_as_no_chunk(self):
        wta1 = BioWTARegressor(
            self.n_models,
            self.n_features,
            start_prob=self.start_prob,
            trans_mat=self.trans_mat,
        )
        r1 = wta1.transform(self.predictors, self.dependent)

        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            start_prob=self.start_prob,
            trans_mat=self.trans_mat,
        )
        r2 = wta2.transform(self.predictors, self.dependent, chunk_hint=12)

        np.testing.assert_allclose(r1, r2)


class TestBioWTARegressorVectorLearningRate(unittest.TestCase):
    def setUp(self):
        self.n_models = 2
        self.n_features = 3

        self.rng = np.random.default_rng(0)
        self.n_samples = 53
        self.predictors = self.rng.normal(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.normal(size=self.n_samples)

        self.rate = 0.005

        self.wta_full = BioWTARegressor(self.n_models, self.n_features, rate=self.rate)
        self.r_full = self.wta_full.transform(self.predictors, self.dependent)

        self.n_partial = self.n_samples // 2
        self.wta_partial = BioWTARegressor(
            self.n_models, self.n_features, rate=self.rate
        )
        self.r_partial = self.wta_partial.transform(
            self.predictors[: self.n_partial], self.dependent[: self.n_partial]
        )

    def test_weights_different_in_partial_and_full_run(self):
        self.assertGreater(
            np.max(np.abs(self.wta_partial.weights_ - self.wta_full.weights_)), 1e-3
        )

    def test_switching_rate_to_zero_fixes_weights(self):
        schedule = np.zeros(self.n_samples)
        schedule[: self.n_partial] = self.rate
        wta = BioWTARegressor(self.n_models, self.n_features, rate=schedule)

        wta.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(wta.weights_, self.wta_partial.weights_)

    def test_resulting_r_same_if_rate_is_constant_then_switches(self):
        schedule = np.zeros(self.n_samples)
        schedule[: self.n_partial] = self.rate
        wta = BioWTARegressor(self.n_models, self.n_features, rate=schedule)

        r = wta.transform(self.predictors, self.dependent)
        np.testing.assert_allclose(r[: self.n_partial], self.r_partial)

    def test_last_value_of_rate_is_used_if_more_samples_than_len_rate(self):
        n = 3 * self.n_samples // 4
        schedule_short = self.rng.uniform(0, self.rate, size=n)
        schedule = np.hstack(
            (schedule_short, (self.n_samples - n) * [schedule_short[-1]])
        )

        wta1 = BioWTARegressor(self.n_models, self.n_features, rate=schedule_short)
        wta2 = BioWTARegressor(self.n_models, self.n_features, rate=schedule)

        wta1.transform(self.predictors, self.dependent)
        wta2.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(wta1.weights_, wta2.weights_)

    def test_constructor_copies_weight_schedule(self):
        schedule = self.rate * np.ones(self.n_samples)
        wta = BioWTARegressor(self.n_models, self.n_features, rate=schedule)

        schedule[:] = 0
        wta.transform(self.predictors, self.dependent)

        np.testing.assert_allclose(wta.weights_, self.wta_full.weights_)


class TestBioWTARegressorCallableLearningRate(unittest.TestCase):
    def test_callable_rate_works_like_vector(self):
        n_models = 3
        n_features = 4

        rng = np.random.default_rng(1)
        n_samples = 55
        predictors = rng.normal(size=(n_samples, n_features))
        dependent = rng.normal(size=n_samples)

        def rate_fct(i):
            return 1 / (1 + 0.5 * i)

        wta1 = BioWTARegressor(n_models, n_features, rate=rate_fct)

        schedule = [rate_fct(_) for _ in range(n_samples)]
        wta2 = BioWTARegressor(n_models, n_features, rate=schedule)

        wta1.transform(predictors, dependent)
        wta2.transform(predictors, dependent)

        np.testing.assert_allclose(wta1.weights_, wta2.weights_)

    def test_callable_rate_works_like_constant(self):
        n_models = 3
        n_features = 4

        rng = np.random.default_rng(2)
        n_samples = 55
        predictors = rng.normal(size=(n_samples, n_features))
        dependent = rng.normal(size=n_samples)

        rate = 1e-4

        def rate_fct(_):
            return rate

        wta1 = BioWTARegressor(n_models, n_features, rate=rate_fct)
        wta2 = BioWTARegressor(n_models, n_features, rate=rate)

        wta1.transform(predictors, dependent)
        wta2.transform(predictors, dependent)

        np.testing.assert_allclose(wta1.weights_, wta2.weights_)


if __name__ == "__main__":
    unittest.main()
