import unittest

import numpy as np

from bioslds.monitor import AttributeMonitor
from bioslds.cepstral import givens, add_lq_column, OnlineCepstralNorm


class TestGivens(unittest.TestCase):
    def test_squared_c_plus_squared_s_is_one(self):
        c, s = givens(1.3, -0.4)
        self.assertAlmostEqual(c * c + s * s, 1)

    def test_rotation_zeros_second_element(self):
        abs_small = 0.3
        abs_large = 1.7

        for signs_str in ["++", "+-", "-+", "--"]:
            for abs_order_str in ["ab", "ba"]:
                with self.subTest(signs_str=signs_str, abs_order_str=abs_order_str):
                    if abs_order_str == "ab":
                        crt_abs_a, crt_abs_b = abs_small, abs_large
                    else:
                        crt_abs_a, crt_abs_b = abs_large, abs_small

                    crt_a = crt_abs_a if signs_str[0] == "+" else -crt_abs_a
                    crt_b = crt_abs_b if signs_str[1] == "+" else -crt_abs_b

                    crt_c, crt_s = givens(crt_a, crt_b)
                    crt_r = np.asarray([[crt_c, crt_s], [-crt_s, crt_c]])
                    crt_v = np.asarray([crt_a, crt_b])
                    crt_vp = crt_v @ crt_r

                    self.assertLess(np.abs(crt_vp[1]), 1e-10)

    def test_rotation_sets_first_element_appropriately(self):
        abs_small = 0.8
        abs_large = 2.1

        d = np.sqrt(abs_small ** 2 + abs_large ** 2)

        for signs_str in ["++", "+-", "-+", "--"]:
            for abs_order_str in ["ab", "ba"]:
                with self.subTest(signs_str=signs_str, abs_order_str=abs_order_str):
                    if abs_order_str == "ab":
                        crt_abs_a, crt_abs_b = abs_small, abs_large
                    else:
                        crt_abs_a, crt_abs_b = abs_large, abs_small

                    crt_a = crt_abs_a if signs_str[0] == "+" else -crt_abs_a
                    crt_b = crt_abs_b if signs_str[1] == "+" else -crt_abs_b

                    crt_c, crt_s = givens(crt_a, crt_b)
                    crt_r = np.asarray([[crt_c, crt_s], [-crt_s, crt_c]])
                    crt_v = np.asarray([crt_a, crt_b])
                    crt_vp = crt_v @ crt_r

                    self.assertAlmostEqual(crt_vp[0], d)

    def test_when_second_element_is_already_zero(self):
        abs_a = 1.5
        for sign in [-1, +1]:
            with self.subTest(sign=sign):
                crt_a = sign * abs_a
                crt_c, crt_s = givens(crt_a, 0)

                self.assertAlmostEqual(crt_c, sign)
                self.assertLess(np.abs(crt_s), 1e-10)

    def test_when_first_element_is_zero(self):
        abs_b = 1.5
        for sign in [-1, +1]:
            with self.subTest(sign=sign):
                crt_b = sign * abs_b
                crt_c, crt_s = givens(0, crt_b)

                self.assertAlmostEqual(crt_s, -sign)
                self.assertLess(np.abs(crt_c), 1e-10)


def _get_lq_l(m: np.ndarray) -> np.ndarray:
    """ Calculate L term from LQ decomposition, ensuring the diagonal is non-negative.

    Parameters
    ----------
    m
        Matrix to process.

    Returns the L term in the LQ decomposition, using the convention that all diagonal
    elements are non-negative. This is the same as getting the Cholesky decomposition of
    `m @ m.T`.
    """
    ll = np.linalg.cholesky(m @ m.T)

    return ll


class TestAddLQColumn(unittest.TestCase):
    def test_add_to_square_matrix(self):
        rng = np.random.default_rng(1)
        n = 5

        m = rng.normal(size=(n, n))
        v = rng.normal(size=n)

        ll = _get_lq_l(m)

        m_alt = np.column_stack((m, v))
        ll_alt = _get_lq_l(m_alt)

        add_lq_column(ll, v)

        np.testing.assert_allclose(ll, ll_alt)

    def test_add_to_rectangular_matrix(self):
        rng = np.random.default_rng(1)
        shape = (4, 7)

        m = rng.normal(size=shape)
        v = rng.normal(size=shape[0])

        ll = _get_lq_l(m)

        m_alt = np.column_stack((m, v))
        ll_alt = _get_lq_l(m_alt)

        add_lq_column(ll, v)

        np.testing.assert_allclose(ll, ll_alt)

    def test_l_stays_diagonal(self):
        rng = np.random.default_rng(2)
        shape = (4, 5)

        m = rng.normal(size=shape)
        v = rng.normal(size=shape[0])

        ll = _get_lq_l(m)

        add_lq_column(ll, v)

        ll_triu = np.triu(ll, k=1)
        np.testing.assert_allclose(ll_triu, 0)


class TestOnlineCesptralNormConstructor(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.order = 5
        self.rate = 0.02
        self.cepstral = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )

    def test_n_features_set(self):
        self.assertEqual(self.cepstral.n_features, self.n_features)

    def test_n_components_set_to_n_features(self):
        self.assertEqual(self.cepstral.n_components, self.n_features)

    def test_order_set(self):
        self.assertEqual(self.cepstral.order, self.order)

    def test_rate_set(self):
        self.assertEqual(self.cepstral.rate, self.rate)


class TestOnlineCesptralNormTransform(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.order = 5
        self.rate = 0.02
        self.cepstral = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )

        self.rng = np.random.default_rng(33)
        self.n_samples = 500
        self.x = self.rng.normal(size=(self.n_samples, self.n_features))

    def test_all_outputs_are_non_negative(self):
        norms = self.cepstral.transform(X=self.x)
        self.assertGreaterEqual(np.min(norms), 0)

    def test_result_does_not_depend_on_position_in_feature_dimension(self):
        norms1 = self.cepstral.transform(X=self.x)

        cepstral2 = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )
        norms2 = cepstral2.transform(X=np.fliplr(self.x))

        np.testing.assert_allclose(norms1, np.fliplr(norms2))

    def test_result_in_one_seq_independent_of_other_seqs(self):
        X1 = self.x
        norms1 = self.cepstral.transform(X=X1)

        X2 = np.copy(self.x)
        X2[:, 1] = self.x[:, 2]
        cepstral2 = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )
        norms2 = cepstral2.transform(X=X2)

        np.testing.assert_allclose(norms1[:, 0], norms2[:, 0])

    def test_norm_generally_changes_at_every_step_after_an_initial_period(self):
        norms = self.cepstral.transform(X=self.x)
        diffs = np.diff(norms, axis=0)

        self.assertGreaterEqual(np.min(np.abs(diffs[2 * self.order :])), 1e-6)

    def test_norm_constant_after_long_enough_streak_of_zeros(self):
        streak_length = 2 * self.order
        extra_length = 35
        X = np.row_stack(
            (self.x, np.zeros((streak_length + extra_length, self.n_features)))
        )
        norms = self.cepstral.transform(X=X)
        diffs = np.diff(norms, axis=0)

        self.assertLess(np.max(np.abs(diffs[-extra_length:])), 1e-10)

    def test_shape_of_output_correct(self):
        norms = self.cepstral.transform(X=self.x)
        self.assertEqual(norms.shape, self.x.shape)

    def test_repeated_transform_same_as_single_transform_on_combined_input(self):
        norms = self.cepstral.transform(X=self.x)

        n1 = self.n_samples // 2
        x1 = self.x[:n1]
        x2 = self.x[n1:]

        cepstral_alt = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )
        norms1 = cepstral_alt.transform(X=x1)
        norms2 = cepstral_alt.transform(X=x2)

        norms_exp = np.row_stack((norms1, norms2))
        np.testing.assert_allclose(norms, norms_exp)

    def test_output_is_set_properly(self):
        norms = self.cepstral.transform(X=self.x)

        np.testing.assert_allclose(self.cepstral.output_, norms[-1])

    def test_history_is_correct(self):
        self.cepstral.transform(X=self.x)
        np.testing.assert_allclose(self.cepstral.history_, self.x[-2 * self.order :])

    def test_monitor_output(self):
        monitor = AttributeMonitor(["output_"])
        norms = self.cepstral.transform(X=self.x, monitor=monitor)

        np.testing.assert_allclose(monitor.history_.output_, norms)


class TestOnlineCesptralNormStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_features = 3
        self.order = 5
        self.rate = 0.02
        self.cepstral = OnlineCepstralNorm(
            self.n_features, order=self.order, rate=self.rate
        )

    def test_repr(self):
        s = repr(self.cepstral)

        self.assertTrue(s.startswith("OnlineCesptralNorm("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("order=", s)
        self.assertIn("rate=", s)

    def test_str(self):
        s = str(self.cepstral)

        self.assertTrue(s.startswith("OnlineCesptralNorm("))
        self.assertTrue(s.endswith(")"))

        self.assertIn("n_features=", s)
        self.assertIn("order=", s)
        self.assertIn("rate=", s)


if __name__ == "__main__":
    unittest.main()
