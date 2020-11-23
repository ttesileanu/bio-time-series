import unittest

import numpy as np

from unittest import mock

from bioslds.cepstral_regressor import CepstralRegressor


class TestCepstralRegressorInit(unittest.TestCase):
    def setUp(self):
        self.n_models = 3
        self.n_features = 4
        self.order = 3
        self.rng = np.random.default_rng(1)
        self.weights = self.rng.normal(size=(self.n_models, self.n_features))

        self.cepstral = CepstralRegressor(self.weights, self.order)

    def test_n_models_attribute(self):
        self.assertEqual(self.cepstral.n_models, self.n_models)

    def test_n_features_attribute(self):
        self.assertEqual(self.cepstral.n_features, self.n_features)

    def test_n_components_attribute(self):
        self.assertEqual(self.cepstral.n_components, self.n_models)

    def test_has_prediction_error(self):
        self.assertTrue(hasattr(self.cepstral, "prediction_error"))

    def test_has_cepstral_norm(self):
        self.assertTrue(hasattr(self.cepstral, "cepstral_norm"))

    def test_cepstral_norm_negative_by_default(self):
        self.assertTrue(self.cepstral.cepstral_norm.negative)


class TestCepstralRegressorExceptions(unittest.TestCase):
    def test_raises_if_n_models_does_not_match_weights_size(self):
        with self.assertRaises(ValueError):
            CepstralRegressor(n_models=2, weights=np.zeros((3, 4)), cepstral_order=2)

    def test_raises_if_n_features_does_not_match_weights_size(self):
        with self.assertRaises(ValueError):
            CepstralRegressor(n_features=2, weights=np.zeros((3, 4)), cepstral_order=3)


class TestCepstralRegressorRng(unittest.TestCase):
    def test_weights_initialized_using_rng_normal(self):
        n_models = 2
        n_features = 3

        rng = mock.Mock()
        weights = np.reshape(np.arange(n_models * n_features), (n_models, n_features))
        rng.normal = mock.Mock(return_value=weights)

        cepstral = CepstralRegressor(
            n_models=n_models, n_features=n_features, rng=rng, cepstral_order=3
        )
        np.testing.assert_allclose(cepstral.prediction_error.weights_, weights)

    def test_rng_can_be_seed(self):
        n_models = 2
        n_features = 3
        seed = 43

        cepstral1 = CepstralRegressor(
            n_models=n_models, n_features=n_features, rng=seed, cepstral_order=3
        )
        cepstral2 = CepstralRegressor(
            n_models=n_models,
            n_features=n_features,
            rng=np.random.default_rng(seed),
            cepstral_order=3,
        )
        np.testing.assert_allclose(
            cepstral1.prediction_error.weights_, cepstral2.prediction_error.weights_
        )

    def test_default_seed_is_zero(self):
        n_models = 2
        n_features = 3

        cepstral1 = CepstralRegressor(
            n_models=n_models, n_features=n_features, cepstral_order=3
        )
        cepstral2 = CepstralRegressor(
            n_models=n_models, n_features=n_features, rng=0, cepstral_order=3
        )
        np.testing.assert_allclose(
            cepstral1.prediction_error.weights_, cepstral2.prediction_error.weights_
        )


class TestCepstralRegressorStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_models = 4
        self.n_features = 3
        self.order = 4

        self.rng = np.random.default_rng(1)
        self.weights = self.rng.normal(size=(self.n_models, self.n_features))

        self.cepstral = CepstralRegressor(self.weights, self.order)

    def test_repr(self):
        s = repr(self.cepstral)

        self.assertTrue(s.startswith("CepstralRegressor("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_models=", s)
        self.assertIn("prediction_error=", s)
        self.assertIn("cepstral_norm=", s)

    def test_str(self):
        s = str(self.cepstral)

        self.assertTrue(s.startswith("CepstralRegressor("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("n_models=", s)
        self.assertIn("prediction_error=", s)
        self.assertIn("cepstral_norm=", s)


if __name__ == "__main__":
    unittest.main()
