import unittest

import numpy as np
from numpy.random import default_rng

from bioslds.arma import Arma


class TestArmaInit(unittest.TestCase):
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

    def test_default_bias_is_zero(self):
        self.assertEqual(self.arma.bias, 0)


class TestArmaInitZeroOrders(unittest.TestCase):
    def setUp(self):
        self.bias = 3.5
        self.arma = Arma([], [], bias=self.bias)

    def test_sets_bias(self):
        self.assertAlmostEqual(self.bias, self.arma.bias)


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

    def test_raises_value_error_if_u_callable_and_n_samples_missing(self):
        with self.assertRaises(ValueError):
            self.arma.transform(U=lambda size: np.zeros(size))

    def test_works_with_callable_source(self):
        arma2 = self.create_initial_arma()

        y1, u1 = self.arma.transform(U=self.u)
        y2, u2 = arma2.transform(self.n, U=lambda size: self.u)

        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(u1, u2)

    def test_works_with_callable_source_different_n_samples(self):
        arma2 = self.create_initial_arma()

        n2 = self.n // 3

        y1, u1 = self.arma.transform(U=self.u[:n2])
        y2, u2 = arma2.transform(n2, U=lambda size: self.u[:size])

        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(u1, u2)


if __name__ == "__main__":
    unittest.main()
