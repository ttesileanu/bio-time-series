import unittest

import numpy as np
from numpy.random import default_rng

from bioslds.arma import Arma


class TestArmaCommon(unittest.TestCase):
    def test_init_creates_coefficients(self):
        a = [1, 0.5]
        b = [0.5, -0.3, 0.2]
        arma = Arma(a, b)

        np.testing.assert_allclose(arma.a, a)
        np.testing.assert_allclose(arma.b, b)

    def test_init_creates_orders(self):
        a = [0.5, -0.3, 0.2]
        b = [0.3, 0.2]
        arma = Arma(a, b)

        self.assertEqual(len(a), arma.p)
        self.assertEqual(len(b), arma.q)

    def test_init_creates_n_dimensions_1d(self):
        a = [1]
        b = [2, -3]
        arma = Arma(a, b)

        self.assertEqual(arma.n_components, 1)
        self.assertEqual(arma.n_features, 1)

    def test_init_creates_n_dimensions_nd(self):
        rng = default_rng(1)
        m = 4
        a = rng.normal(size=(3, m, m))
        b = rng.normal(size=(2, m, m))
        arma = Arma(a, b)

        self.assertEqual(arma.n_components, m)
        self.assertEqual(arma.n_features, m)

    def test_init_sets_bias(self):
        bias = 3.5
        arma = Arma([], [], bias=bias)
        self.assertAlmostEqual(bias, arma.bias)

    def test_init_sets_bias_nd(self):
        bias = [1.5, -0.5]
        arma = Arma([], [], bias=bias)
        np.testing.assert_allclose(bias, arma.bias)

    def test_init_n_dimensions_from_bias_when_zero_orders(self):
        bias = [1, 2, -0.5]
        arma = Arma([], [], bias=bias)

        self.assertEqual(arma.n_components, len(bias))
        self.assertEqual(arma.n_features, len(bias))

    def test_init_default_bias_is_zero_1d(self):
        arma = Arma([1, 2], [0.5, -0.3])
        self.assertEqual(arma.bias, 0)

    def test_init_default_bias_is_zero_nd(self):
        rng = default_rng(2)
        m = 3
        a = rng.normal(size=(2, m, m))
        b = rng.normal(size=(2, m, m))
        arma = Arma(a, b)
        np.testing.assert_allclose(arma.bias, 0)

    def test_init_raises_if_a_and_b_sizes_do_not_match(self):
        rng = default_rng(3)
        a = rng.normal(size=(2, 3, 3))
        b = rng.normal(size=(2, 4, 4))
        with self.assertRaises(ValueError):
            Arma(a, b)

    def test_init_raises_if_coeffs_and_bias_sizes_do_not_match(self):
        rng = default_rng(4)
        a = rng.normal(size=(2, 4, 4))
        b = rng.normal(size=(2, 4, 4))
        bias = rng.normal(size=3)
        with self.assertRaises(ValueError):
            Arma(a, b, bias=bias)


if __name__ == '__main__':
    unittest.main()
