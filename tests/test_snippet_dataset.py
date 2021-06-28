import unittest

import numpy as np

from bioslds.snippet_dataset import RandomSnippetDataset
from bioslds.utils import rle_encode


class TestRandomSnippetDatasetAccess(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.n_signals = 5
        self.n_snippets = 3
        self.n_samples = 25
        self.snippets = [
            rng.normal(size=rng.integers(10, 30)) for _ in range(self.n_snippets)
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


class TestRandomSnippetDatasetOutputBasic(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_snippets = 3
        self.n_samples = 256
        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(3, 5)) for _ in range(self.n_snippets)
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


class TestRandomSnippetDatasetOutputDetails(unittest.TestCase):
    def setUp(self):
        self.n_signals = 1
        self.n_snippets = 3
        self.n_samples = 256
        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(10, 25)) for _ in range(self.n_snippets)
        ]
        self.dataset = RandomSnippetDataset(
            self.n_signals, self.n_samples, self.snippets, dwell_times=7
        )

    def test_signal_snippets_match_usage_seq(self):
        sig = self.dataset[0]
        usage_rle = rle_encode(sig.usage_seq)
        idx = 0
        for elem, n in usage_rle:
            full_snippet = self.snippets[elem]
            sub_y = sig.y[idx : idx + n]

            n_pos = len(full_snippet) - n + 1
            errors = np.zeros(n_pos)
            for i in range(n_pos):
                errors[i] = np.max(np.abs(full_snippet[i : i + n] - sub_y))

            self.assertLess(np.min(errors), 1e-6)

            idx += n

    def test_used_snippets_are_not_constant_distance_from_the_ends(self):
        sig = self.dataset[0]
        usage_rle = rle_encode(sig.usage_seq)
        idx = 0
        min_error_idxs = np.zeros(len(usage_rle), dtype=int)
        min_error_idxs_rev = np.zeros(len(usage_rle), dtype=int)
        for k, (elem, n) in enumerate(usage_rle):
            full_snippet = self.snippets[elem]
            sub_y = sig.y[idx : idx + n]

            n_pos = len(full_snippet) - n + 1
            errors = np.zeros(n_pos)
            for i in range(n_pos):
                errors[i] = np.max(np.abs(full_snippet[i : i + n] - sub_y))

            min_error_idxs[k] = np.argmin(errors)
            min_error_idxs_rev[k] = n_pos - min_error_idxs[k]
            idx += n

        self.assertGreater(len(np.unique(min_error_idxs)), 1)
        self.assertGreater(len(np.unique(min_error_idxs_rev)), 1)


class TestRandomSnippetDatasetRng(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 51
        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(5, 10)) for _ in range(self.n_signals)
        ]

    def create_dataset(self, rng) -> RandomSnippetDataset:
        dataset = RandomSnippetDataset(
            self.n_signals, self.n_samples, self.snippets, rng=rng
        )
        return dataset

    def test_int_seed_rng_same_as_calling_default_rng(self):
        seed = 13
        dataset1 = self.create_dataset(seed)
        dataset2 = self.create_dataset(np.random.default_rng(seed))

        for crt_sig1, crt_sig2 in zip(dataset1, dataset2):
            np.testing.assert_equal(crt_sig1.y, crt_sig2.y)
            np.testing.assert_equal(crt_sig1.usage_seq, crt_sig2.usage_seq)

    def test_with_random_state(self):
        dataset = self.create_dataset(np.random.RandomState(1))
        self.assertEqual(len(dataset), self.n_signals)


class TestRandomSnippetDatasetKeywordForwarding(unittest.TestCase):
    def test_init_kwargs_forwarded_to_semi_markov(self):
        dataset = RandomSnippetDataset(
            3, 20, snippets=([0, 1, 2], [-0.5, 0.5, 1]), min_dwell=2, dwell_times=2.5,
        )
        for crt_sig in dataset:
            rle = rle_encode(crt_sig.usage_seq)
            # note that last streak could be chopped, so might be shorter than min_dwell
            self.assertGreaterEqual(min(_[1] for _ in rle[:-1]), 2)


class TestRandomSnippetDatasetStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_signals = 4
        self.n_samples = 10

        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(5, 10)) for _ in range(self.n_signals)
        ]
        self.semi_markov_kws = {"min_dwell": 2}
        self.dataset = RandomSnippetDataset(
            self.n_signals, self.n_samples, self.snippets, **self.semi_markov_kws
        )

    def test_str(self):
        s = str(self.dataset)
        s_exp = (
            f"RandomSnippetDataset(n_signals={self.n_signals}, "
            + f"n_samples={self.n_samples})"
        )

        self.assertEqual(s, s_exp)

    def test_repr(self):
        r = repr(self.dataset)

        self.assertTrue(r.startswith("RandomSnippetDataset("))
        self.assertTrue(r.endswith(")"))
        self.assertNotEqual(r.find("n_signals="), -1)
        self.assertNotEqual(r.find("n_samples="), -1)
        self.assertNotEqual(r.find("signal_seeds="), -1)
        self.assertNotEqual(r.find("semi_markov_kws="), -1)


class TestRandomSnippetDatasetNormalize(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 100
        rng = np.random.default_rng(0)
        self.snippets = [
            rng.normal(size=rng.integers(5, 10)) for _ in range(self.n_signals)
        ]
        self.kwargs = dict(
            n_signals=self.n_signals, n_samples=self.n_samples, snippets=self.snippets
        )
        self.dataset = RandomSnippetDataset(**self.kwargs)

    def test_normalize_returns_same_signals_with_unit_variance(self):
        dataset_norm = RandomSnippetDataset(**self.kwargs, normalize=True)
        for sig, sig_norm in zip(self.dataset, dataset_norm):
            # noinspection PyTypeChecker
            self.assertAlmostEqual(np.std(sig_norm.y), 1)
            np.testing.assert_allclose(sig_norm.y, sig.y / np.std(sig.y))
            np.testing.assert_equal(sig_norm.usage_seq, sig.usage_seq)

            self.assertAlmostEqual(1.0, sig.scale)
            self.assertAlmostEqual(1.0 / np.std(sig.y), sig_norm.scale)


if __name__ == "__main__":
    unittest.main()
