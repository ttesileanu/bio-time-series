import unittest

import numpy as np

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
