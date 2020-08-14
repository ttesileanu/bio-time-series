import unittest

import numpy as np

from unittest import mock
from bioslds.regressors import BioWTARegressor


class TestBioWTARegressorInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.wta = BioWTARegressor(self.n_models, self.n_features)

    def test_n_models_attribute(self):
        self.assertEqual(self.wta.n_models, self.n_models)

    def test_n_features_attribute(self):
        self.assertEqual(self.wta.n_features, self.n_features)

    def test_default_rate_weights(self):
        np.testing.assert_allclose(self.wta.rate_weights, 1e-3)

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


class TestBioWTARegressorFitInferDefaultInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 5
        self.wta = BioWTARegressor(self.n_models, self.n_features)

        self.rng = np.random.default_rng(1)
        self.n_samples = 85
        self.predictors = self.rng.uniform(size=(self.n_samples, self.n_features))
        self.dependent = self.rng.uniform(size=self.n_samples)

    def test_output_shape(self):
        r = self.wta.fit_infer(self.predictors, self.dependent)

        np.testing.assert_equal(np.shape(r), (self.n_samples, self.n_models))

    def test_every_output_row_sums_to_one(self):
        r = self.wta.fit_infer(self.predictors, self.dependent)
        sums = np.sum(r, axis=1)

        np.testing.assert_allclose(sums, 1)

    def test_output_is_softmax_of_minus_half_squared_error(self):
        ini_weights = np.copy(self.wta.weights_)
        x = self.predictors[[0], :]
        y = self.dependent[[0]]
        r = self.wta.fit_infer(x, y)

        err = np.zeros(self.n_models)
        for i in range(self.n_models):
            w = ini_weights[i]
            err[i] = y[0] - np.dot(w, x[0])

        np.testing.assert_allclose(
            np.log(r[0]) - np.log(r[0][0]), -0.5 * (err ** 2 - err[0] ** 2)
        )

    def test_raises_if_size_of_X_is_not_right(self):
        n_samples = 5
        with self.assertRaises(ValueError):
            self.wta.fit_infer(np.zeros((n_samples, 2)), np.zeros(n_samples))

    def test_output_of_repeated_calls_to_fit_infer_equivalent_to_single_call(self):
        n1 = 4 * self.n_samples // 7
        r1 = self.wta.fit_infer(self.predictors[:n1], self.dependent[:n1])
        r2 = self.wta.fit_infer(self.predictors[n1:], self.dependent[n1:])

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        r = wta_again.fit_infer(self.predictors, self.dependent)

        np.testing.assert_allclose(r, np.vstack((r1, r2)))

    def test_weight_change_from_repeated_fit_infer_calls_equivalent_to_one_call(self):
        n1 = 2 * self.n_samples // 7
        n2 = 3 * self.n_samples // 7
        self.wta.fit_infer(self.predictors[:n1], self.dependent[:n1])
        self.wta.fit_infer(self.predictors[n1 : n1 + n2], self.dependent[n1 : n1 + n2])
        self.wta.fit_infer(self.predictors[n1 + n2 :], self.dependent[n1 + n2 :])

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        wta_again.fit_infer(self.predictors, self.dependent)

        np.testing.assert_allclose(self.wta.weights_, wta_again.weights_)

    def test_only_weights_for_best_fitting_model_change(self):
        all_ks = []
        for crt_x, crt_y in zip(self.predictors, self.dependent):
            weights0 = np.copy(self.wta.weights_)
            r = self.wta.fit_infer([crt_x], [crt_y])[0]

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
        r = self.wta.fit_infer(self.predictors[[0]], self.dependent[[0]])
        k = np.argmax(r)

        dw = (self.wta.weights_ - weights0)[k]
        eps_k = self.dependent[0] - np.dot(weights0[k], self.predictors[0])
        dw_exp = self.wta.rate_weights * self.predictors[0] * eps_k

        np.testing.assert_allclose(dw, dw_exp)

    def test_history(self):
        _, history = self.wta.fit_infer(
            self.predictors, self.dependent, return_history=True
        )

        wta_again = BioWTARegressor(self.n_models, self.n_features)
        weights = []
        predictions = []
        for crt_x, crt_y in zip(self.predictors, self.dependent):
            weights.append(np.copy(wta_again.weights_))

            crt_all_pred = np.dot(wta_again.weights_, crt_x)
            crt_r = wta_again.fit_infer([crt_x], [crt_y])
            crt_k = np.argmax(crt_r)

            crt_pred = crt_all_pred[crt_k]
            predictions.append(crt_pred)

        np.testing.assert_allclose(weights, history.weights)
        np.testing.assert_allclose(predictions, history.predictions)

    def test_progress_called(self):
        mock_progress = mock.MagicMock()

        self.wta.fit_infer(self.predictors, self.dependent, progress=mock_progress)

        mock_progress.assert_called()


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
            f"{self.n_features}, rate_weights={self.wta.rate_weights}, weights_="
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

        r1 = self.wta.fit_infer(self.predictors, self.dependent)
        r2 = wta2.fit_infer(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)

    def test_default_latent_state_transition_matrix_is_uniform(self):
        wta2 = BioWTARegressor(
            self.n_models,
            self.n_features,
            trans_mat=np.ones((self.n_models, self.n_models)) / self.n_models,
        )

        r1 = self.wta.fit_infer(self.predictors, self.dependent)
        r2 = wta2.fit_infer(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)

    def test_changing_initial_latent_state_distribution_changes_output(self):
        start_prob = (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        r1 = self.wta.fit_infer(self.predictors, self.dependent)
        r2 = wta2.fit_infer(self.predictors, self.dependent)

        self.assertGreater(np.max(np.abs(r1 - r2)), 1e-3)

    def test_changing_latent_state_transition_matrix_changes_output(self):
        trans_mat = [
            (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]
        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        r1 = self.wta.fit_infer(self.predictors, self.dependent)
        r2 = wta2.fit_infer(self.predictors, self.dependent)

        self.assertGreater(np.max(np.abs(r1 - r2)), 1e-3)

    def test_when_start_prob_large_for_a_state_then_that_state_gets_high_r(self):
        r1 = self.wta.fit_infer(self.predictors[[0]], self.dependent[[0]])
        k1 = np.argmax(r1[0])

        # make another state win out
        k = 0 if k1 != 0 else 1
        p_large = 0.999
        start_prob = (1 - p_large) * np.ones(self.n_models) / (self.n_models - 1)
        start_prob[k] = p_large
        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        r2 = wta2.fit_infer(self.predictors[[0]], self.dependent[[0]])
        k2 = np.argmax(r2[0])

        self.assertNotEqual(k1, k2)
        self.assertEqual(k2, k)

    def test_when_trans_mat_large_for_a_target_state_then_that_state_gets_high_r(self):
        # first figure out where we're starting
        r1 = self.wta.fit_infer(self.predictors[:2], self.dependent[:2])
        k1_start = np.argmax(r1[0])
        k1 = np.argmax(r1[1])

        # make another state win out
        k = 0 if k1 != 0 else 1
        p0 = 0.999
        trans_mat = np.ones((self.n_models, self.n_models)) / self.n_models
        trans_mat[k1_start, :] = (1 - p0) * np.ones(self.n_models) / (self.n_models - 1)
        trans_mat[k1_start, k] = p0
        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        r2 = wta2.fit_infer(self.predictors[:2], self.dependent[:2])
        k2_start = np.argmax(r2[0])
        k2 = np.argmax(r2[1])

        self.assertEqual(k1_start, k2_start)

        self.assertNotEqual(k1, k2)
        self.assertEqual(k2, k)

    def test_initial_r_value_changes_by_correct_amount_according_to_start_prob(self):
        r1 = self.wta.fit_infer(self.predictors[[0]], self.dependent[[0]])

        # change initial state distribution
        start_prob = (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
        wta2 = BioWTARegressor(self.n_models, self.n_features, start_prob=start_prob)

        r2 = wta2.fit_infer(self.predictors[[0]], self.dependent[[0]])

        diff_log_r = np.log(r2[0]) - np.log(r1[0])
        np.testing.assert_allclose(
            diff_log_r - diff_log_r[0], np.log(start_prob) - np.log(start_prob[0]),
        )

    def test_second_r_value_changes_by_correct_amount_according_to_trans_mat(self):
        r1 = self.wta.fit_infer(self.predictors[:2], self.dependent[:2])

        # change transition matrix
        trans_mat = [
            (lambda v: v / np.sum(v))(self.rng.uniform(size=self.n_models))
            for _ in range(self.n_models)
        ]
        wta2 = BioWTARegressor(self.n_models, self.n_features, trans_mat=trans_mat)

        r2 = wta2.fit_infer(self.predictors[:2], self.dependent[:2])
        np.testing.assert_allclose(r1[0], r2[0])

        diff_log_r = np.log(r2[1]) - np.log(r1[1])
        expected_diff_log_r = r2[0] @ np.log(trans_mat)
        np.testing.assert_allclose(
            diff_log_r - diff_log_r[0], expected_diff_log_r - expected_diff_log_r[0]
        )

    def test_float_trans_mat(self):
        r1 = self.wta.fit_infer(self.predictors, self.dependent)

        wta2 = BioWTARegressor(
            self.n_models, self.n_features, trans_mat=1 / self.n_models
        )
        r2 = wta2.fit_infer(self.predictors, self.dependent)

        np.testing.assert_allclose(r1, r2)


if __name__ == "__main__":
    unittest.main()
