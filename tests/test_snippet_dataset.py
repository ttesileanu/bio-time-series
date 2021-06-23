import unittest

import numpy as np

from bioslds.snippet_dataset import RandomSnippetDataset
from bioslds.utils import rle_encode


class TestRandomSnippetDatasetAccess(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n_signals = 5
        self.n_samples = 25
        self.snippets = [
            rng.normal(size=rng.integers(10, 30)) for _ in range(self.n_signals)
        ]
        self.dataset = RandomSnippetDataset(
            self.n_signals, self.n_samples, self.snippets
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), self.n_signals)

    def test_iterate_matches_random_access(self):
        for i, crt_sig in enumerate(self.dataset):
            np.testing.assert_equal(crt_sig.y, self.dataset[i].y)
            np.testing.assert_equal(crt_sig.usage_seq, self.dataset[i].usage_seq)

    def test_iteration_stops_after_correct_number_of_steps(self):
        n = 0
        for _ in self.dataset:
            n += 1
        self.assertEqual(n, self.n_signals)

    def test_raises_index_error_if_subscript_too_large(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[self.n_signals]

    def test_negative_index_counts_from_end(self):
        for i in range(1, self.n_signals + 1):
            ir = self.n_signals - i
            np.testing.assert_equal(self.dataset[ir].y, self.dataset[-i].y)
            np.testing.assert_equal(
                self.dataset[ir].usage_seq, self.dataset[-i].usage_seq
            )

    def test_raises_index_error_if_subscript_too_negative(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[-self.n_signals - 1]


class TestRandomSnippetDatasetOutput(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 256
        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(3, 5)) for _ in range(self.n_signals)
        ]
        self.dataset = RandomSnippetDataset(
            self.n_signals, self.n_samples, self.snippets, dwell_times=3,
        )

    def test_output_contains_y_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "y"))

    def test_output_contains_usage_seq_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "usage_seq"))

    def test_y_output_has_correct_n_samples(self):
        for crt_sig in self.dataset:
            self.assertEqual(len(crt_sig.y), self.n_samples)

    def test_usage_seq_output_has_correct_n_samples(self):
        for crt_sig in self.dataset:
            self.assertEqual(len(crt_sig.usage_seq), self.n_samples)

    def test_usage_seq_is_int(self):
        for crt_sig in self.dataset:
            self.assertTrue(np.issubdtype(crt_sig.usage_seq.dtype, np.integer))

    def test_repeated_access_yields_same_output(self):
        idx = 1
        crt_sig1 = self.dataset[idx]
        crt_sig2 = self.dataset[idx]

        np.testing.assert_equal(crt_sig1.y, crt_sig2.y)
        np.testing.assert_equal(crt_sig1.usage_seq, crt_sig2.usage_seq)

    def test_different_signals_are_different(self):
        crt_sig1 = self.dataset[0]
        crt_sig2 = self.dataset[1]
        self.assertGreater(np.max(np.abs(crt_sig1.y - crt_sig2.y)), 1e-3)

    def test_different_usage_seqs_are_different(self):
        crt_sig1 = self.dataset[0]
        crt_sig2 = self.dataset[1]
        self.assertGreater(
            np.max(np.abs(crt_sig1.usage_seq - crt_sig2.usage_seq)), 1e-3
        )

    def test_ensure_all_streaks_are_no_longer_than_respective_snippets(self):
        for crt_sig in self.dataset:
            encoded = rle_encode(crt_sig.usage_seq)
            for (elem, n) in encoded:
                self.assertLessEqual(n, len(self.snippets[elem]))


if __name__ == "__main__":
    unittest.main()
