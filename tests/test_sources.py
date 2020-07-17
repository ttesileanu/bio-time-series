import unittest

import numpy as np

import bioslds.sources as sources


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
            y = src(size=1)


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


if __name__ == '__main__':
    unittest.main()
