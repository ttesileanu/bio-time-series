import unittest

import numpy as np

from bioslds.utils import rle_encode, rle_decode


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
            (rng.integers(0, ns), rng.integers(1, max_run + 1))
            for _ in range(n_rle)
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


if __name__ == "__main__":
    unittest.main()
