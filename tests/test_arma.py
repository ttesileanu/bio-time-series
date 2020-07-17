import unittest

import numpy as np
from numpy.random import default_rng
from unittest import mock
from typing import Callable, Optional

from bioslds.arma import Arma, make_random_arma


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

    def test_inverse_recovers_input_with_nonzero_bias(self):
        arma = Arma([1.3, -0.8, 0.2, -0.1], [-0.3, 0.5], bias=0.6)
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


class TestArmaCopy(unittest.TestCase):
    @staticmethod
    def createArma(default_source: Optional[Callable] = None):
        return Arma(
            [-1.1, -0.6, -0.1],
            [0.5, 0.3],
            bias=-0.41,
            default_source=default_source,
        )

    def test_sample_n1_then_copy_then_sample_n2_same_as_sample_n1_plus_n2(self):
        n = 90
        rng = default_rng(2)
        u = rng.normal(size=n)

        n1 = 3 * n // 5
        arma1 = self.createArma()
        y_exp, _ = arma1.transform(U=u)

        arma2 = self.createArma()
        y1, _ = arma2.transform(U=u[:n1])
        arma2_copy = arma2.copy()
        y2, _ = arma2_copy.transform(U=u[n1:])

        np.testing.assert_allclose(y_exp, np.hstack((y1, y2)))

    def test_deep_copy_of_pseudorandom_state(self):
        n = 85

        n1 = 2 * n // 5
        seed = 51
        arma1 = self.createArma(default_source=default_rng(seed).normal)
        y_exp, _ = arma1.transform(n)

        arma2 = self.createArma(default_source=default_rng(seed).normal)
        y1, _ = arma2.transform(n1)
        arma2_copy = arma2.copy()
        y2, _ = arma2_copy.transform(n - n1)

        np.testing.assert_allclose(y_exp, np.hstack((y1, y2)))


class TestMakeRandomArma(unittest.TestCase):
    def setUp(self):
        self.rng = default_rng(101)

    def test_generated_ar_has_correct_orders(self):
        p = 3
        ar = make_random_arma(p, 0, self.rng)

        self.assertEqual(ar.p, p)
        self.assertEqual(ar.q, 0)

        self.assertEqual(len(ar.a), ar.p)
        self.assertEqual(len(ar.b), ar.q)

    def test_generated_ma_has_correct_orders(self):
        q = 2
        ar = make_random_arma(0, q, self.rng)

        self.assertEqual(ar.p, 0)
        self.assertEqual(ar.q, q)

        self.assertEqual(len(ar.a), ar.p)
        self.assertEqual(len(ar.b), ar.q)

    def test_additional_kws_passed_to_arma_init(self):
        kws = {"bias": 5, "default_source": lambda size: np.zeros(size)}
        with mock.patch("bioslds.arma.Arma") as MockInit:
            make_random_arma(2, 3, self.rng, **kws)

            self.assertEqual(MockInit.call_args_list[0][1], kws)

    def test_generated_arma_ar_coeffs_are_random(self):
        arma1 = make_random_arma(3, 4, self.rng)
        arma2 = make_random_arma(3, 4, self.rng)

        self.assertFalse(np.allclose(arma1.a, arma2.a))

    def test_generated_arma_ma_coeffs_are_random(self):
        arma1 = make_random_arma(3, 4, self.rng)
        arma2 = make_random_arma(3, 4, self.rng)

        self.assertFalse(np.allclose(arma1.b, arma2.b))

    def assert_if_poles_or_zeros_not_within_given_radius(
        self, n: int, kind: str, order: int, radius: float
    ):
        """ Generate many AR or MA processes and check that their poles or zeros
        are within the given radius.

        Parameters
        ----------
        n
            Number of processes to generate.
        kind
            Kind of process, either "ar" or "ma".
        order
            Order of the AR or MA processes.
        radius
            Radius to request (and check) for the poles / zeros.
        """
        if kind == "ar":
            p = order
            q = 0
        elif kind == "ma":
            p = 0
            q = order
        else:
            raise NotImplementedError("Unknown kind.")

        all_roots = []
        for i in range(n):
            arma = make_random_arma(
                p, q, self.rng, max_pole_radius=radius, max_zero_radius=radius
            )

            coeffs = np.ones(order + 1)
            if q == 0:
                coeffs[1:] = -arma.a
            else:
                coeffs[1:] = arma.b

            roots = np.roots(coeffs)
            all_roots.extend(roots)

        np.testing.assert_array_less(np.abs(all_roots), radius)

    def test_generated_poles_within_unit_radius_by_default(self):
        self.assert_if_poles_or_zeros_not_within_given_radius(30, "ar", 6, 1.0)

    def test_generated_zeros_within_unit_radius_by_default(self):
        self.assert_if_poles_or_zeros_not_within_given_radius(30, "ma", 6, 1.0)

    def test_generated_poles_within_given_radius(self):
        self.assert_if_poles_or_zeros_not_within_given_radius(30, "ar", 5, 0.5)

    def test_generated_zeros_within_given_radius(self):
        self.assert_if_poles_or_zeros_not_within_given_radius(30, "ma", 5, 0.7)

    def test_generated_process_has_zero_bias_by_default(self):
        arma = make_random_arma(5, 3, self.rng)
        self.assertEqual(arma.bias, 0)

    def test_generated_process_has_bias_in_given_range(self):
        n = 20
        bias_range = (-0.5, 0.7)
        biases = []
        for i in range(n):
            arma = make_random_arma(5, 3, self.rng, bias_range=bias_range)
            biases.append(arma.bias)

        np.testing.assert_array_less(bias_range[0], biases)
        np.testing.assert_array_less(biases, bias_range[1])

    def test_generated_process_has_random_biases_when_bias_range_given(self):
        n = 20
        bias_range = (-0.5, 0.7)
        biases = []
        for i in range(n):
            arma = make_random_arma(5, 3, self.rng, bias_range=bias_range)
            biases.append(arma.bias)

        self.assertGreater(np.std(biases), 0.0)

    def test_raises_if_both_bias_range_and_bias_keyword(self):
        with self.assertRaises(TypeError):
            make_random_arma(3, 2, default_rng(1), bias_range=(2, 3), bias=2.5)


if __name__ == "__main__":
    unittest.main()
