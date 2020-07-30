import unittest

import numpy as np

import bioslds.sources as sources
from bioslds.arma import make_random_arma


class TestSourcesConstant(unittest.TestCase):
    def test_all_returned_values_are_equal_to_given_constant(self):
        x = 3.5
        src = sources.Constant(x)

        y = src(size=15)

        np.testing.assert_allclose(y, x)

    def test_returns_requested_number_of_values(self):
        src = sources.Constant(0.5)

        n = 23
        y = src(size=n)
        self.assertEqual(len(y), n)


class TestSourcesStreamBasic(unittest.TestCase):
    def test_empty_result_if_zero_samples_requested_from_empty_store(self):
        src = sources.Stream([])
        y = src(size=0)

        self.assertEqual(len(y), 0)

    def test_reading_samples_from_empty_store_raises_index_error(self):
        src = sources.Stream([])
        with self.assertRaises(IndexError):
            src(size=1)


class TestSourcesStream(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.n = 100
        self.data = rng.normal(size=self.n)
        self.src = sources.Stream(self.data)

    def test_empty_result_if_zero_samples_requested(self):
        y = self.src(size=0)

        self.assertEqual(len(y), 0)

    def test_returned_samples_match_data(self):
        y = self.src(size=self.n)
        np.testing.assert_allclose(y, self.data)

    def test_generate_n1_then_n2_returns_first_n1_plus_n2(self):
        n1 = self.n // 3
        n2 = self.n // 2
        y1 = self.src(size=n1)
        y2 = self.src(size=n2)
        y = np.hstack((y1, y2))

        data_exp = self.data[: n1 + n2]
        np.testing.assert_allclose(y, data_exp)

    def test_requesting_too_many_items_raises_index_error(self):
        with self.assertRaises(IndexError):
            self.src(size=self.n + 1)


class TestSourcesGaussianNoise(unittest.TestCase):
    def test_returned_number_of_samples_matches_size(self):
        src = sources.GaussianNoise(1)
        n = 50
        y = src(size=n)
        self.assertEqual(len(y), n)

    def test_returned_samples_are_random(self):
        src = sources.GaussianNoise(1)

        n = 10
        y1 = src(size=n)
        y2 = src(size=n)

        self.assertGreater(np.max(np.abs(y1 - y2)), 1e-3)

    def test_init_can_take_default_rng_seed_or_rng(self):
        seed = 123
        src1 = sources.GaussianNoise(seed)
        src2 = sources.GaussianNoise(np.random.default_rng(seed))

        n = 13
        y1 = src1(size=n)
        y2 = src2(size=n)

        np.testing.assert_allclose(y1, y2)

    def test_loc_matters(self):
        seed = 123
        src1 = sources.GaussianNoise(seed, loc=0)
        src2 = sources.GaussianNoise(seed, loc=1)

        n = 13
        y1 = src1(size=n)
        y2 = src2(size=n)

        self.assertGreater(np.max(np.abs(y1 - y2)), 1e-3)

    def test_scale_matters(self):
        seed = 123
        src1 = sources.GaussianNoise(seed, scale=1)
        src2 = sources.GaussianNoise(seed, scale=2)

        n = 13
        y1 = src1(size=n)
        y2 = src2(size=n)

        self.assertGreater(np.max(np.abs(y1 - y2)), 1e-3)

    def test_when_scale_is_zero_all_values_equal_loc(self):
        loc = 0.35
        src = sources.GaussianNoise(1, loc=loc, scale=0)

        n = 10
        y = src(size=n)

        np.testing.assert_allclose(y, loc)

    def test_default_loc_is_zero(self):
        seed = 123
        src1 = sources.GaussianNoise(seed)
        src2 = sources.GaussianNoise(seed, loc=0)

        n = 13
        y1 = src1(size=n)
        y2 = src2(size=n)

        np.testing.assert_allclose(y1, y2)

    def test_default_scale_is_one(self):
        seed = 1
        src1 = sources.GaussianNoise(seed)
        src2 = sources.GaussianNoise(seed, scale=1)

        n = 12
        y1 = src1(size=n)
        y2 = src2(size=n)

        np.testing.assert_allclose(y1, y2)


class TestSourcesFixSourceScale(unittest.TestCase):
    def test_fix_source_scale_does_not_affect_sources_rng_by_default(self):
        seed = 123
        src1 = sources.GaussianNoise(seed)

        n = 12
        u1 = src1(size=n)

        rng = np.random.default_rng(30)
        src2 = sources.GaussianNoise(seed)
        arma = make_random_arma(3, 2, rng, default_source=src2)

        sources.fix_source_scale(arma)

        # reset scale
        src2.scale = 1
        u2 = src2(size=n)

        np.testing.assert_allclose(u1, u2)

    def test_fix_source_scale_affects_sources_rng_when_use_copy_is_false(self):
        seed = 123
        src1 = sources.GaussianNoise(seed)

        n = 12
        u1 = src1(size=n)

        rng = np.random.default_rng(30)
        src2 = sources.GaussianNoise(seed)
        arma = make_random_arma(3, 2, rng, default_source=src2)

        sources.fix_source_scale(arma, use_copy=False)

        # reset scale
        src2.scale = 1
        u2 = src2(size=n)

        self.assertGreater(np.max(np.abs(u1 - u2)), 1e-3)

    def test_ar1_output_variance_is_fixed_to_one_by_default(self):
        seed = 10
        src = sources.GaussianNoise(seed)

        rng = np.random.default_rng(30)
        arma = make_random_arma(1, 0, rng, default_source=src)

        sources.fix_source_scale(arma, n_samples=5000)
        ma_var = src.scale ** 2
        arma_var = ma_var / (1 - arma.a[0] ** 2)

        self.assertAlmostEqual(arma_var, 1, places=2)

    def test_ar2_output_variance_is_fixed_to_one_by_default(self):
        seed = 10
        src = sources.GaussianNoise(seed)

        rng = np.random.default_rng(30)
        arma = make_random_arma(2, 0, rng, default_source=src)

        sources.fix_source_scale(arma, n_samples=10000)
        ma_var = src.scale ** 2
        a_diff = 1 - arma.a[1]
        a_sum = 1 + arma.a[1]
        arma_var = a_diff * ma_var / (a_sum * (a_diff ** 2 - arma.a[0] ** 2))

        self.assertAlmostEqual(arma_var, 1, places=2)

    def test_scale_varies_in_proportion_to_output_variance(self):
        seed = 10
        src = sources.GaussianNoise(seed)

        rng = np.random.default_rng(30)
        arma = make_random_arma(2, 0, rng, default_source=src)

        var1 = 0.5
        sources.fix_source_scale(arma, output_std=var1)
        scale1 = src.scale

        var2 = 1.5
        sources.fix_source_scale(arma, output_std=var2)
        scale2 = src.scale

        self.assertAlmostEqual(var1 / var2, scale1 / scale2, places=2)

    def test_fix_source_scale_returns_final_scale(self):
        seed = 10
        src = sources.GaussianNoise(seed)

        rng = np.random.default_rng(30)
        arma = make_random_arma(2, 0, rng, default_source=src)

        scale = sources.fix_source_scale(arma)

        self.assertAlmostEqual(scale, src.scale)


if __name__ == "__main__":
    unittest.main()
