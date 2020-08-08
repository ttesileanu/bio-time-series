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


if __name__ == "__main__":
    unittest.main()
