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


if __name__ == '__main__':
    unittest.main()
