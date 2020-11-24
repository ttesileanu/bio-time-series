import unittest

import numpy as np
from unittest import mock
from typing import Optional

from bioslds.dataset import RandomArmaDataset
from bioslds.arma import Arma
from bioslds import sources


class TestRandomArmaDatasetAccess(unittest.TestCase):
    def setUp(self):
        self.n_signals = 5
        self.n_samples = 25
        self.dataset = RandomArmaDataset(
            self.n_signals, self.n_samples, [(1, 1), (2, 1)]
        )

    def test_len(self):
        self.assertEqual(len(self.dataset), self.n_signals)

    def test_iterate_matches_random_access(self):
        for i, crt_sig in enumerate(self.dataset):
            np.testing.assert_equal(crt_sig.y, self.dataset[i].y)
            np.testing.assert_equal(crt_sig.u, self.dataset[i].u)
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
            np.testing.assert_equal(self.dataset[ir].u, self.dataset[-i].u)
            np.testing.assert_equal(
                self.dataset[ir].usage_seq, self.dataset[-i].usage_seq
            )

    def test_raises_index_error_if_subscript_too_negative(self):
        with self.assertRaises(IndexError):
            _ = self.dataset[-self.n_signals - 1]


class TestRandomArmaDatasetOutput(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 32
        self.dataset = RandomArmaDataset(
            self.n_signals, self.n_samples, [(3, 0), (1, 1)]
        )

    def test_output_contains_u_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "u"))

    def test_output_contains_y_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "y"))

    def test_output_contains_usage_seq_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "usage_seq"))

    def test_output_contains_armas_field(self):
        for crt_sig in self.dataset:
            self.assertTrue(hasattr(crt_sig, "armas"))

    def test_y_output_has_correct_n_samples(self):
        for crt_sig in self.dataset:
            self.assertEqual(len(crt_sig.y), self.n_samples)

    def test_u_output_has_correct_n_samples(self):
        for crt_sig in self.dataset:
            self.assertEqual(len(crt_sig.u), self.n_samples)

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
        np.testing.assert_equal(crt_sig1.u, crt_sig2.u)
        np.testing.assert_equal(crt_sig1.usage_seq, crt_sig2.usage_seq)

    def test_signal_armas_matches_armas_sequence(self):
        for i, crt_sig in enumerate(self.dataset):
            crt_armas = crt_sig.armas
            crt_armas_exp = self.dataset.armas[i]
            self.assertEqual(len(crt_armas), len(crt_armas_exp))
            for arma1, arma2 in zip(crt_armas, crt_armas_exp):
                self.assertIs(arma1, arma2)

    def test_different_signals_are_different(self):
        crt_sig1 = self.dataset[0]
        crt_sig2 = self.dataset[1]
        self.assertGreater(np.max(np.abs(crt_sig1.y - crt_sig2.y)), 1e-3)

    def test_different_sources_are_different(self):
        crt_sig1 = self.dataset[0]
        crt_sig2 = self.dataset[1]
        self.assertGreater(np.max(np.abs(crt_sig1.u - crt_sig2.u)), 1e-3)

    def test_different_usage_seqs_are_different(self):
        crt_sig1 = self.dataset[0]
        crt_sig2 = self.dataset[1]
        self.assertGreater(
            np.max(np.abs(crt_sig1.usage_seq - crt_sig2.usage_seq)), 1e-3
        )


class TestRandomArmaDatasetOutputWithTrivialNoise(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 100

        self.alphas = [0.99, 0.95]
        self.armas = [Arma([_], []) for _ in self.alphas]

        self.y0 = 1.0
        self.dataset = RandomArmaDataset(
            self.n_signals,
            self.n_samples,
            armas=self.armas,
            source=sources.Constant(0),
            initial_conditions=([self.y0], []),
        )

    def test_output_correct_when_armas_given_by_hand_and_noise_is_absent(self):
        for crt_sig in self.dataset:
            last_y = self.y0
            for crt_y, crt_id in zip(crt_sig.y, crt_sig.usage_seq):
                expected_y = self.alphas[crt_id] * last_y
                self.assertAlmostEqual(crt_y, expected_y)

                last_y = crt_y

            np.testing.assert_equal(crt_sig.u, 0)

    def test_armas_attribute_has_length_equal_to_n_signals(self):
        self.assertEqual(len(self.dataset.armas), self.n_signals)

    def test_armas_attribute_mirrors_armas_init_arg(self):
        for crt_armas in self.dataset.armas:
            for crt_arma, crt_exp_arma in zip(crt_armas, self.armas):
                self.assertIs(crt_arma, crt_exp_arma)


class TestRandomArmaDatasetArmasAtInit(unittest.TestCase):
    def setUp(self):
        self.n_signals = 4
        self.n_samples = 10

        self.arma_orders = [(3, 2), (1, 5)]
        self.dataset = RandomArmaDataset(
            self.n_signals, self.n_samples, self.arma_orders
        )

    def test_armas_are_created(self):
        self.assertTrue(hasattr(self.dataset, "armas"))

    def test_proper_number_of_sets_of_armas_are_created(self):
        self.assertEqual(len(self.dataset.armas), self.n_signals)

    def test_each_signal_is_assigned_proper_number_of_armas(self):
        for crt_armas in self.dataset.armas:
            self.assertEqual(len(crt_armas), len(self.arma_orders))

    def test_armas_have_correct_orders(self):
        for crt_armas in self.dataset.armas:
            for crt_arma, crt_orders in zip(crt_armas, self.arma_orders):
                self.assertEqual(crt_arma.p, crt_orders[0])
                self.assertEqual(crt_arma.q, crt_orders[1])

    def test_generated_armas_are_different(self):
        arma1 = self.dataset.armas[0][0]
        arma2 = self.dataset.armas[1][0]
        self.assertGreater(np.max(np.abs(arma1.a - arma2.a)), 1e-4)
        self.assertGreater(np.max(np.abs(arma1.b - arma2.b)), 1e-4)

    def test_noise_scales_is_none_by_default(self):
        self.assertIsNone(self.dataset.noise_scales)


class TestRandomArmaDatasetMissingArmas(unittest.TestCase):
    def test_raises_if_no_armas_and_no_arma_orders(self):
        with self.assertRaises(ValueError):
            RandomArmaDataset(3, 3)


class TestRandomArmaDatasetRng(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 51
        self.arma_orders = [(2, 1), (3, 2), (1, 3), (4, 3)]

    def create_dataset(self, rng) -> RandomArmaDataset:
        dataset = RandomArmaDataset(
            self.n_signals, self.n_samples, self.arma_orders, rng=rng
        )
        return dataset

    def test_int_seed_rng_same_as_calling_default_rng(self):
        seed = 13
        dataset1 = self.create_dataset(seed)
        dataset2 = self.create_dataset(np.random.default_rng(seed))

        for crt_sig1, crt_sig2 in zip(dataset1, dataset2):
            np.testing.assert_equal(crt_sig1.y, crt_sig2.y)
            np.testing.assert_equal(crt_sig1.u, crt_sig2.u)
            np.testing.assert_equal(crt_sig1.usage_seq, crt_sig2.usage_seq)

    def test_with_random_state(self):
        dataset = self.create_dataset(np.random.RandomState(1))
        self.assertEqual(len(dataset), self.n_signals)


class TestRandomArmaDatasetKeywordForwarding(unittest.TestCase):
    def test_arma_kws_forwarded_to_make_random_arma(self):
        dataset = RandomArmaDataset(
            3, 20, arma_orders=[(1, 1), (2, 2)], arma_kws={"bias_range": (0, 1)}
        )
        for crt_armas in dataset.armas:
            for crt_arma in crt_armas:
                self.assertGreater(crt_arma.bias, 1e-3)

    def test_init_kwargs_forwarded_to_arma_hsmm(self):
        start_prob = [1, 0, 0]
        trans_mat = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]

        n_samples = 25
        dataset = RandomArmaDataset(
            2,
            n_samples,
            arma_orders=[(3, 0), (2, 1), (1, 3)],
            start_prob=start_prob,
            trans_mat=trans_mat,
        )
        for crt_sig in dataset:
            np.testing.assert_equal(crt_sig.usage_seq, np.arange(n_samples) % 3)


class TestRandomArmaDatasetCustomSource(unittest.TestCase):
    def test_source_is_copied_before_using(self):
        dataset = RandomArmaDataset(
            1, 23, [(1, 2), (3, 0)], source=sources.GaussianNoise()
        )

        sig1 = dataset[0]
        sig2 = dataset[0]

        np.testing.assert_equal(sig1.y, sig2.y)
        np.testing.assert_equal(sig1.u, sig2.u)
        np.testing.assert_equal(sig1.usage_seq, sig2.usage_seq)


class TestRandomArmaDatasetStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.n_signals = 4
        self.n_samples = 10

        self.arma_orders = [(3, 2), (1, 5)]
        self.arma_hsmm_kws = {"min_dwell_time": 5}
        self.dataset = RandomArmaDataset(
            self.n_signals, self.n_samples, self.arma_orders, **self.arma_hsmm_kws
        )

    def test_str(self):
        s = str(self.dataset)
        s_exp = (
            f"RandomArmaDataset(n_signals={self.n_signals}, "
            + f"n_samples={self.n_samples})"
        )

        self.assertEqual(s, s_exp)

    def test_repr(self):
        r = repr(self.dataset)
        r_exp = (
            f"RandomArmaDataset(n_signals={self.n_signals}, "
            + f"n_samples={self.n_samples}, "
            + f"signal_seeds={repr(self.dataset.signal_seeds)}, "
            + f"arma_hsmm_kws={repr(self.arma_hsmm_kws)})"
        )

        self.assertEqual(r, r_exp)


class TestRandomArmaDatasetFixScaleInit(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 100
        self.arma_orders = [(3, 3), (2, 3)]
        self.fix_scale = 1.5
        self.scales = [2.3, 0.2, 1.2, 1.5, 0.2, 1.0]

        with mock.patch("bioslds.dataset.sources.fix_source_scale") as MockFixScale:
            MockFixScale.side_effect = self.scales
            self.dataset = RandomArmaDataset(
                self.n_signals,
                self.n_samples,
                arma_orders=self.arma_orders,
                fix_scale=self.fix_scale,
            )
            self.mock_fix_scale = MockFixScale

    def test_fix_source_scale_called_the_right_number_of_times(self):
        self.assertEqual(
            self.mock_fix_scale.call_count, self.n_signals * len(self.arma_orders)
        )

    def test_fix_source_scale_called_with_correct_output_std(self):
        self.mock_fix_scale.assert_called()
        for call in self.mock_fix_scale.call_args_list:
            self.assertIn("output_std", call[1])
            self.assertAlmostEqual(call[1]["output_std"], self.fix_scale)

    def test_fix_source_scale_outputs_stored_in_noise_scales(self):
        m = len(self.arma_orders)
        for i, scales_per_sig in enumerate(self.dataset.noise_scales):
            np.testing.assert_allclose(scales_per_sig, self.scales[m * i : m * (i + 1)])


class TestRandomArmaDatasetFixScaleNoiseScaledPerProcess(unittest.TestCase):
    def setUp(self):
        self.n_signals = 3
        self.n_samples = 100
        self.scales = np.asarray([2.3, 0.2, 1.2, 1.5, 0.2, 1.0])
        self.dataset = self.create_dataset(1.0)

    def create_dataset(self, fix_scale: Optional[float]):
        with mock.patch("bioslds.dataset.sources.fix_source_scale") as MockFixScale:
            MockFixScale.side_effect = self.scales
            dataset = RandomArmaDataset(
                self.n_signals,
                self.n_samples,
                arma_orders=[(0, 0), (0, 0)],  # ARMAs simply return their inputs
                fix_scale=fix_scale,
            )

        return dataset

    def test_noise_values_are_appropriately_scaled(self):
        scales_i = 0
        for i, sig_fix in enumerate(self.dataset):
            scale_seq = self.scales[scales_i + sig_fix.usage_seq]
            expected_y = sig_fix.u * scale_seq
            np.testing.assert_allclose(sig_fix.y, expected_y)

            scales_i += len(self.dataset.armas[i])

    def test_noise_values_do_not_change_upon_using_fix_scale(self):
        dataset_alt = self.create_dataset(None)

        for sig, sig_fix in zip(dataset_alt, self.dataset):
            np.testing.assert_allclose(sig.u, sig_fix.u)

    def test_usage_seq_does_not_change_upon_using_fix_scale(self):
        dataset_alt = self.create_dataset(None)

        for sig, sig_fix in zip(dataset_alt, self.dataset):
            np.testing.assert_allclose(sig.usage_seq, sig_fix.usage_seq)


if __name__ == "__main__":
    unittest.main()
