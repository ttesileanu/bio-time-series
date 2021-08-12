import unittest

import numpy as np

from bioslds.utils import rle_encode, rle_decode, to_hankel


class TestRleEncode(unittest.TestCase):
    def test_empty(self):
        seq = rle_encode([])

        self.assertEqual(len(seq), 0)

    def test_example(self):
        seq_rle = rle_encode([0, 0, 1, 2, 2, 2, 1, 1, 0])
        exp_rle = [(0, 2), (1, 1), (2, 3), (1, 2), (0, 1)]

        self.assertEqual(exp_rle, seq_rle)


class TestRleDecode(unittest.TestCase):
    def test_empty(self):
        seq = rle_decode([])

        self.assertEqual(len(seq), 0)

    def test_example(self):
        seq = rle_decode([(1, 3), (0, 1), (2, 2)])
        exp = [1, 1, 1, 0, 2, 2]

        np.testing.assert_equal(seq, exp)


class TestRleRoundtrip(unittest.TestCase):
    def test_encode_then_decode(self):
        ns = 5
        n = 23
        rng = np.random.default_rng(1)
        seq = rng.integers(0, ns, size=n)

        seq_rle = rle_encode(seq)

        np.testing.assert_equal(rle_decode(seq_rle), seq)

    def test_decode_then_encode(self):
        ns = 4
        n_rle = 12
        max_run = 8

        rng = np.random.default_rng(2)
        seq_rle0 = [
            (rng.integers(0, ns), rng.integers(1, max_run + 1)) for _ in range(n_rle)
        ]
        # make sure we don't have repeated elements
        last_elem = None
        seq_rle = []
        for elem, n_rep in seq_rle0:
            if elem != last_elem:
                seq_rle.append((elem, n_rep))

            last_elem = elem

        self.assertGreater(len(seq_rle), 0)

        seq = rle_decode(seq_rle)

        self.assertEqual(rle_encode(seq), seq_rle)


class TestToHankel(unittest.TestCase):
    def test_returns_empty_for_empty_y(self):
        H = to_hankel([], 3)
        self.assertEqual(np.size(H), 0)

    def test_returns_empty_if_p_smaller_than_one(self):
        H = to_hankel([1, 2, 3], 0)
        self.assertEqual(np.size(H), 0)

    def test_output_has_len_y_rows(self):
        y = [1, 0, 1, 2, 3, 4, 2, 3, 0.5]
        p = 3
        H = to_hankel(y, p)

        self.assertEqual(len(H), len(y))

    def test_output_matches_hankel_definition(self):
        rng = np.random.default_rng(0)

        n = 50
        p = 4
        y = rng.normal(size=n)
        H = to_hankel(y, p)

        for i in range(n):
            for j in range(p):
                if i >= j:
                    self.assertAlmostEqual(H[i, j], y[i - j])
                else:
                    self.assertEqual(H[i, j], 0)

    def test_output_is_different_when_step_is_nontrivial(self):
        y = [1, 0, 1, 2, 3, 4, 2, 3, 0.5]
        p = 3

        H1 = to_hankel(y, p, step=1)
        H2 = to_hankel(y, p, step=2)

        self.assertFalse(np.allclose(H1, H2))

    def test_default_step_is_one(self):
        y = [1, 0, 1, 2, 3, 4, 2, 3, 0.5]
        p = 3

        H1 = to_hankel(y, p)
        H2 = to_hankel(y, p, step=1)

        np.testing.assert_equal(H1, H2)

    def test_output_has_p_cols_even_for_nontrivial_step(self):
        y = [1, 0, 1, 2, 3, 4, 2, 3, 0.5]
        p = 3
        H = to_hankel(y, p, step=3)

        self.assertEqual(np.shape(H)[1], p)

    def test_output_matches_definition_for_nontrivial_step(self):
        rng = np.random.default_rng(1)

        n = 50
        p = 4
        step = 3
        y = rng.normal(size=n)
        H = to_hankel(y, p, step=step)

        for i in range(n):
            for j in range(p):
                js = j * step
                if i >= js:
                    self.assertAlmostEqual(H[i, j], y[i - js])
                else:
                    self.assertEqual(H[i, j], 0)


if __name__ == "__main__":
    unittest.main()
