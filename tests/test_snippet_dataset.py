import unittest

import numpy as np

from bioslds.snippet_dataset import RandomSnippetDataset


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


if __name__ == "__main__":
    unittest.main()
