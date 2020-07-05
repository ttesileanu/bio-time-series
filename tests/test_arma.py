import unittest

import numpy as np
from numpy.random import default_rng

from bioslds.arma import Arma


class TestArmaInit1d(unittest.TestCase):
    def setUp(self):
        self.a = [1, 0.5]
        self.b = [0.5, -0.3, 0.2]
        self.arma = Arma(self.a, self.b)

    def test_creates_coefficients(self):
        np.testing.assert_allclose(self.arma.a, self.a)
        np.testing.assert_allclose(self.arma.b, self.b)

    def test_creates_orders(self):
        self.assertEqual(len(self.a), self.arma.p)
        self.assertEqual(len(self.b), self.arma.q)

    def test_creates_n_components(self):
        self.assertEqual(self.arma.n_components, 1)

    def test_creates_n_features(self):
        self.assertEqual(self.arma.n_features, 1)


class TestArmaInitNd(unittest.TestCase):
    def setUp(self):
        rng = default_rng(1)
        self.m = 4
        self.a = rng.normal(size=(3, self.m, self.m))
        self.b = rng.normal(size=(2, self.m, self.m))
        self.arma = Arma(self.a, self.b)

    def test_creates_n_components(self):
        self.assertEqual(self.arma.n_components, self.m)

    def test_creates_n_features(self):
        self.assertEqual(self.arma.n_features, self.m)


class TestArmaInitZeroOrdersBias1d(unittest.TestCase):
    def setUp(self):
        self.bias = 3.5
        self.arma = Arma([], [], bias=self.bias)

    def test_sets_bias(self):
        self.assertAlmostEqual(self.bias, self.arma.bias)


class TestArmaInitZeroOrdersBiasNd(unittest.TestCase):
    def setUp(self):
        self.bias = [1.5, -0.5]
        self.arma = Arma([], [], bias=self.bias)

    def test_sets_bias_nd(self):
        np.testing.assert_allclose(self.bias, self.arma.bias)

    def test_n_components(self):
        self.assertEqual(self.arma.n_components, len(self.bias))

    def test_n_features(self):
        self.assertEqual(self.arma.n_features, len(self.bias))


class TestArmaInitDefaultBias1d(unittest.TestCase):
    def setUp(self):
        self.arma = Arma([1, 2], [0.5, -0.3])

    def test_default_bias_is_zero(self):
        self.assertEqual(self.arma.bias, 0)


class TestArmaInitDefaultBiasNd(unittest.TestCase):
    def setUp(self):
        rng = default_rng(2)
        self.m = 3
        self.a = rng.normal(size=(2, self.m, self.m))
        self.b = rng.normal(size=(2, self.m, self.m))
        self.arma = Arma(self.a, self.b)

    def test_default_bias_is_zero(self):
        np.testing.assert_allclose(self.arma.bias, 0)


class TestArmaInitRaisesOnSizeMismatch(unittest.TestCase):
    def test_raises_if_a_and_b_sizes_do_not_match(self):
        rng = default_rng(3)
        a = rng.normal(size=(2, 3, 3))
        b = rng.normal(size=(2, 4, 4))
        with self.assertRaises(ValueError):
            Arma(a, b)

    def test_raises_if_coeffs_and_bias_sizes_do_not_match(self):
        rng = default_rng(4)
        a = rng.normal(size=(2, 4, 4))
        b = rng.normal(size=(2, 4, 4))
        bias = rng.normal(size=3)
        with self.assertRaises(ValueError):
            Arma(a, b, bias=bias)


class TestArmaTransformZeroSamples(unittest.TestCase):
    def setUp(self):
        # this also implicitly tests that this works without default_source
        self.arma = Arma([1, 0.5], [-0.3])

    def test_zero_samples_returns_empty(self):
        y, u = self.arma.transform(0)

        self.assertEqual(len(y), 0)
        self.assertEqual(len(u), 0)

    def test_empty_u_returns_empty(self):
        y, u = self.arma.transform(U=[])

        self.assertEqual(len(y), 0)
        self.assertEqual(len(u), 0)


class TestArmaTransform(unittest.TestCase):
    def setUp(self):
        rng = default_rng(4)
        self.n = 123
        # initial conditions (y, u)
        self.ic = (np.array([0.2, -0.3, 0.1]), np.array([-0.1, 0.5]))
        self.arma = self.create_initial_arma()
        self.u = rng.normal(size=self.n)

    def create_initial_arma(self) -> Arma:
        return Arma([-1.1, -0.6, -0.1], [0.8, 0.2], initial_conditions=self.ic)

    def test_result_length_matches_input_length(self):
        y, _ = self.arma.transform(U=self.u)

        self.assertEqual(self.n, len(y))

    def test_output_u_matches_input(self):
        _, u_out = self.arma.transform(U=self.u)

        np.testing.assert_allclose(self.u, u_out)

    def test_single_sample(self):
        sample = 0.4
        y, _ = self.arma.transform(U=[sample])

        ar_part = np.dot(np.flip(self.ic[0]), self.arma.a)
        ma_part = sample + np.dot(np.flip(self.ic[1]), self.arma.b)

        self.assertAlmostEqual(y[0], ar_part + ma_part)

    def test_transform_n1_then_n2_samples_same_as_transform_n1_plus_n2(self):
        arma_copy = self.create_initial_arma()

        # transform in two parts
        n1 = 2 * self.n // 9
        u1 = self.u[:n1]
        u2 = self.u[n1:]
        y1, _ = self.arma.transform(U=u1)
        y2, _ = self.arma.transform(U=u2)

        y = np.hstack((y1, y2))

        # transform all at once
        y_exp, _ = arma_copy.transform(U=self.u)

        np.testing.assert_allclose(y, y_exp)


if __name__ == "__main__":
    unittest.main()
