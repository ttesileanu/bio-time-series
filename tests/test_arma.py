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


class TestArmaTransformPureAr(unittest.TestCase):
    def test_ar1_zero_input(self):
        alpha = 0.78
        y0 = 1.0
        ar = Arma([alpha], [], initial_conditions=([y0], []))

        n = 10
        y, _ = ar.transform(n, U=lambda size: np.zeros(size))

        y_exp = y0 * alpha ** np.arange(1, n + 1)
        np.testing.assert_allclose(y, y_exp)


class TestArmaTransformPureMa(unittest.TestCase):
    def test_ma_is_convolution(self):
        rng = default_rng(1)
        q = 3
        b = rng.normal(size=q)
        ma = Arma([], b)

        n = 52
        u = rng.normal(size=n)

        y, _ = ma.transform(U=u)

        u_padded = np.hstack((np.zeros(q), u))
        b_ext = np.hstack(([1], b))
        y_exp = np.convolve(u_padded, b_ext, mode="valid")

        np.testing.assert_allclose(y, y_exp)


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


class TestArmaIsStable(unittest.TestCase):
    def test_pure_ma_is_stable(self):
        rng = default_rng(1)
        arma = Arma([], rng.normal(size=4))

        self.assertTrue(arma.is_stable())

    def test_stable_pure_ar(self):
        arma = Arma([-1.1, -0.6, -0.1], [])
        self.assertTrue(arma.is_stable())

    def test_unstable_pure_ar(self):
        arma = Arma([1.1], [])
        self.assertFalse(arma.is_stable())

    def test_critical_ar_counts_as_not_stable(self):
        arma = Arma([1.0], [])
        self.assertFalse(arma.is_stable())

    def test_unstable_arma(self):
        arma = Arma([0.6, 0.55], [0.5, -0.5])
        self.assertFalse(arma.is_stable())

    def test_stable_arma(self):
        arma = Arma([-1.1, -0.6, -0.1], [0.5, 0.3])
        self.assertTrue(arma.is_stable())


class TestArmaIsInvertible(unittest.TestCase):
    def test_pure_ar_is_invertible(self):
        rng = default_rng(1)
        arma = Arma(rng.normal(size=4), [])

        self.assertTrue(arma.is_invertible())

    def test_invertible_pure_ma(self):
        arma = Arma([], [1.1, 0.6, 0.1])
        self.assertTrue(arma.is_invertible())

    def test_noninvertible_pure_ma(self):
        arma = Arma([], [-1.1])
        self.assertFalse(arma.is_invertible())

    def test_critical_ma_counts_as_not_invertible(self):
        arma = Arma([], [1.0])
        self.assertFalse(arma.is_invertible())

    def test_noninvertible_arma(self):
        arma = Arma([1.1], [-0.6, -0.55])
        self.assertFalse(arma.is_invertible())

    def test_invertible_arma(self):
        arma = Arma([0.5, 0.3], [1.1, 0.6, 0.1])
        self.assertTrue(arma.is_invertible())


class TestArmaInverseBasic(unittest.TestCase):
    def test_inverse_pure_ar_is_pure_ma(self):
        arma = Arma([0.5, 0.3], [])
        inv_arma = arma.inverse()

        self.assertEqual(inv_arma.p, 0)
        self.assertEqual(len(inv_arma.a), 0)

    def test_inverse_pure_ma_is_pure_ar(self):
        arma = Arma([], [-1.1, -0.6, -0.1])
        inv_arma = arma.inverse()

        self.assertEqual(inv_arma.q, 0)
        self.assertEqual(len(inv_arma.b), 0)

    def test_inverse_arma_switches_orders(self):
        arma = Arma([-1.1, -0.6, -0.1], [0.5, 0.3])
        inv_arma = arma.inverse()

        self.assertEqual(inv_arma.p, arma.q)
        self.assertEqual(inv_arma.q, arma.p)

        self.assertEqual(len(inv_arma.a), inv_arma.p)
        self.assertEqual(len(inv_arma.b), inv_arma.q)


class TestArmaInverse(unittest.TestCase):
    def setUp(self):
        rng = default_rng(5)
        self.n = 101
        self.u = rng.normal(size=self.n)

    def test_inverse_recovers_input_with_zero_bias(self):
        arma = Arma([-1.1, -0.6, -0.1], [0.5, 0.3])
        inv_arma = arma.inverse()

        y, _ = arma.transform(U=self.u)
        u_again, _ = inv_arma.transform(U=y)

        np.testing.assert_allclose(self.u, u_again)


class TestArmaBias(unittest.TestCase):
    def test_trivial_arma_with_bias_no_noise(self):
        bias = 0.75
        arma = Arma([], [], bias=bias)

        y, _ = arma.transform(U=np.zeros(15))

        np.testing.assert_allclose(y, bias)

    def test_asymptotic_value_constant_noise(self):
        bias = 0.32
        arma = Arma([-1.1, -0.6, -0.1], [0.5, 0.3], bias=bias)

        # the asymptotic value y0 should be given by
        # y0 * (1 - sum(a)) = bias + u0 * (1 + sum(b))
        # so: y0 = (bias + u0 * (1 + sum(b))) / (1 - sum(a))
        u0 = -0.5
        # give it enough time to converge
        n = 1000
        y, _ = arma.transform(U=u0 * np.ones(n))

        y0 = (bias + u0 * (1 + np.sum(arma.b))) / (1 - np.sum(arma.a))
        self.assertAlmostEqual(y[-1], y0)


if __name__ == "__main__":
    unittest.main()
