import unittest

import numpy as np
from numpy.random import default_rng
from typing import Callable, Optional

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

    def test_raises_value_error_if_no_u_and_no_n_samples(self):
        with self.assertRaises(ValueError):
            self.arma.transform()

    def test_raises_value_error_if_no_u_and_no_default_source(self):
        with self.assertRaises(ValueError):
            self.arma.transform(5)

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


class TestTransformDefaultSource(unittest.TestCase):
    def setUp(self):
        rng = default_rng(5)
        self.n = 15
        self.u = rng.normal(size=self.n)
        self.callable = lambda size: self.u[:size]
        self.arma = self.create_initial_arma(self.callable)

    @staticmethod
    def create_initial_arma(default_source: Optional[Callable]) -> Arma:
        return Arma(
            [-1.1, -0.6, -0.1], [0.8, 0.2], default_source=default_source
        )

    def test_default_source_same_as_u_callable(self):
        y1, u1 = self.arma.transform(self.n)

        arma2 = self.create_initial_arma(None)
        y2, u2 = arma2.transform(self.n, U=self.callable)

        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(u1, u2)

    def test_default_source_same_as_u_callable_partial_n(self):
        n = self.n // 2
        y1, u1 = self.arma.transform(n)

        arma2 = self.create_initial_arma(None)
        y2, u2 = arma2.transform(n, U=self.callable)

        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(u1, u2)

    def test_default_source_same_as_u_matrix_partial_n(self):
        n = self.n // 3
        y1, u1 = self.arma.transform(n)

        arma2 = self.create_initial_arma(None)
        y2, u2 = arma2.transform(U=self.u[:n])

        np.testing.assert_allclose(y1, y2)
        np.testing.assert_allclose(u1, u2)


class TestArmaStrAndRepr(unittest.TestCase):
    def setUp(self):
        a = np.asarray([1, 0.5])
        b = np.asarray([0.5, -0.3, 0.2])
        bias = 3.5
        self.arma = Arma(a, b, bias=bias)

        self.a_str = str(a)
        self.a_repr = repr(a)

        self.b_str = str(b)
        self.b_repr = repr(b)

        self.bias_str = str(bias)
        self.bias_repr = repr(bias)

    def test_str(self):
        s = str(self.arma)
        s_exp = "Arma(a={}, b={}, bias={})".format(
            self.a_str, self.b_str, self.bias_str
        )

        self.assertEqual(s, s_exp)

    def test_repr(self):
        r = repr(self.arma)
        r_exp = "Arma(a={}, b={}, bias={}, default_source=None)".format(
            self.a_repr, self.b_repr, self.bias_repr
        )

        self.assertEqual(r, r_exp)


if __name__ == "__main__":
    unittest.main()
