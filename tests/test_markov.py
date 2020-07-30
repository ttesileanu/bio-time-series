import unittest

import numpy as np

from bioslds.markov import SemiMarkov


class TestSemiMarkovWithPseudorandom(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.smm = SemiMarkov(self.n_components)

    def test_repeated_runs_yield_different_results(self):
        n = 100
        seq1 = self.smm.sample(n)
        seq2 = self.smm.sample(n)

        self.assertGreater(np.max(np.abs(seq1 - seq2)), 0)

    def test_default_seed_is_zero(self):
        smm2 = SemiMarkov(self.n_components, rng=0)

        n = 50
        seq1 = self.smm.sample(n)
        seq2 = smm2.sample(n)

        np.testing.assert_equal(seq1, seq2)

    def test_output_is_int(self):
        n = 23
        seq = self.smm.sample(n)

        dtype = np.asarray(seq).dtype
        self.assertTrue(np.issubdtype(dtype, np.integer))

    def test_int_seed_uses_numpy_default_rng(self):
        smm2 = SemiMarkov(self.n_components, rng=np.random.default_rng(0))

        n = 43
        seq1 = self.smm.sample(n)
        seq2 = smm2.sample(n)

        np.testing.assert_equal(seq1, seq2)


class TestSemiMarkovWorksWithRandomState(unittest.TestCase):
    def test_no_error(self):
        smm = SemiMarkov(5, rng=np.random.RandomState(1))
        n = 13
        seq = smm.sample(n)

        self.assertEqual(len(seq), n)


class TestSemiMarkovStateContentOfOutputIsRight(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.smm = SemiMarkov(self.n_components, rng=1)

    def test_all_states_are_between_zero_and_n_components(self):
        n = 100
        seq = self.smm.sample(n)

        self.assertGreaterEqual(np.min(seq), 0)
        self.assertLess(np.max(seq), self.n_components)

    def test_lowest_state_in_long_sequence_is_zero(self):
        n = 200
        seq = self.smm.sample(n)

        self.assertEqual(np.min(seq), 0)

    def test_highest_state_in_long_sequence_is_n_components_minus_one(self):
        n = 200
        seq = self.smm.sample(n)

        self.assertEqual(np.max(seq), self.n_components - 1)


class TestSemiMarkovZeroInitialProbabilityStates(unittest.TestCase):
    def test_single_non_zero_initial_prob(self):
        n_components = 6
        start_prob = np.zeros(n_components)
        state_idx = 3
        start_prob[state_idx] = 1.0
        smm = SemiMarkov(n_components, start_prob=start_prob)

        n_seq = 40
        n = 4
        for i in range(n_seq):
            seq = smm.sample(n)
            self.assertEqual(seq[0], state_idx, f"at iteration {i}")

    def test_several_non_zero_initial_prob(self):
        n_components = 6
        start_prob = np.zeros(n_components)
        state_idxs = [1, 3, 5]
        start_prob[state_idxs] = 1.0 / len(state_idxs)
        smm = SemiMarkov(n_components, start_prob=start_prob, rng=1)

        n_seq = 40
        n = 4
        all_ini_states = []
        for i in range(n_seq):
            seq = smm.sample(n)
            self.assertIn(seq[0], state_idxs, f"at iteration {i}")

            all_ini_states.append(seq[0])

        self.assertEqual(set(all_ini_states), set(state_idxs))

    def test_all_states_occur_by_default(self):
        n_components = 5
        smm = SemiMarkov(n_components, rng=2)

        n_seq = 50
        n = 3
        all_ini_states = []
        for i in range(n_seq):
            seq = smm.sample(n)
            all_ini_states.append(seq[0])

        self.assertEqual(set(all_ini_states), set(range(n_components)))


class TestSemiMarkovZeroProbabilityTransitions(unittest.TestCase):
    def test_deterministic_cyclic_transitions(self):
        n_components = 4
        cycle = [2, 3, 1, 0]
        trans_mat = np.zeros((n_components, n_components))
        allowed_pairs = set()
        for i in range(len(cycle)):
            s1 = cycle[i]
            s2 = cycle[(i + 1) % len(cycle)]
            trans_mat[s1, s2] = 1.0
            allowed_pairs.add((s1, s2))
        smm = SemiMarkov(n_components, trans_mat=trans_mat)

        n_seq = 6
        n = 40
        for i in range(n_seq):
            seq = smm.sample(n)
            pairs = set(zip(seq, seq[1:]))

            for s1, s2 in pairs:
                self.assertIn((s1, s2), allowed_pairs, f"iteration {i}")

    def test_absorbing_state(self):
        n_components = 3
        abs_idx = 2
        dwell_times = np.ones(n_components)
        dwell_times[abs_idx] = np.inf
        smm = SemiMarkov(n_components, dwell_times=dwell_times)

        n_seq = 10
        n = 16
        for i in range(n_seq):
            seq = smm.sample(n)
            where_abs = (seq == abs_idx).nonzero()[0]

            # make sure we reach absorbing state
            self.assertGreater(len(where_abs), 0)

            # and stay there once we've reached it
            np.testing.assert_equal(seq[where_abs[0] :], abs_idx)

    def test_all_non_dwelling_transitions_occur_by_default(self):
        n_components = 3
        smm = SemiMarkov(n_components, rng=4)

        n_seq = 5
        n = 200
        for i in range(n_seq):
            seq = smm.sample(n)
            pairs = set(zip(seq, seq[1:]))

            self.assertEqual(len(pairs), n_components * (n_components - 1))

    def test_all_transitions_occur_if_dwell_time_larger_than_one(self):
        n_components = 3
        smm = SemiMarkov(n_components, rng=5, dwell_times=2.0)

        n_seq = 5
        n = 200
        for i in range(n_seq):
            seq = smm.sample(n)
            pairs = set(zip(seq, seq[1:]))

            self.assertEqual(len(pairs), n_components ** 2)


class TestSemiMarkovSampleOutputHasCorrectLength(unittest.TestCase):
    def test_pure_markov(self):
        smm = SemiMarkov(4)

        n = 12
        seq = smm.sample(n)

        self.assertEqual(len(seq), n)

    def test_semi_markov(self):
        smm = SemiMarkov(5, min_dwell=3, max_dwell=13, dwell_times=4.3)

        n = 28
        seq = smm.sample(n)

        self.assertEqual(len(seq), n)


class TestSemiMarkovDwellTimeConstraintsObeyed(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.min_dwell = [2, 3, 1, 0]
        self.max_dwell = [5, 3, np.inf, 4]
        self.dwell_times = [2.5, 3.0, 3.5, 2.0]
        self.smm = SemiMarkov(
            self.n_components,
            min_dwell=self.min_dwell,
            max_dwell=self.max_dwell,
            dwell_times=self.dwell_times,
        )

    @staticmethod
    def to_rle(seq: np.ndarray) -> list:
        starts = np.hstack(([0], np.diff(seq).nonzero()[0] + 1, len(seq)))
        rle = [
            (seq[tmp1], tmp2 - tmp1) for tmp1, tmp2 in zip(starts, starts[1:])
        ]

        return rle

    def test_dwell_times_reach_minimum_but_do_not_go_below(self):
        seq = self.smm.sample(300)
        seq_rle = self.to_rle(seq)

        for i in range(self.n_components):
            # last element of RLE might be truncated due to n_samples
            dwell_times = [_[1] for _ in seq_rle[:-1] if _[0] == i]

            self.assertGreater(len(dwell_times), 0, f"State {i} does not occur")
            self.assertEqual(
                np.min(dwell_times), max(1, self.min_dwell[i]), f"State {i}"
            )

    def test_dwell_times_reach_maximum_but_do_not_go_above(self):
        seq = self.smm.sample(2000)
        seq_rle = self.to_rle(seq)

        for i in range(self.n_components):
            # last element of RLE might be truncated due to n_samples
            dwell_times = [_[1] for _ in seq_rle[:-1] if _[0] == i]

            self.assertGreater(len(dwell_times), 0, f"State {i} does not occur")
            if np.isfinite(self.max_dwell[i]):
                self.assertEqual(
                    np.max(dwell_times), self.max_dwell[i], f"State {i}"
                )
            else:
                self.assertGreater(len(np.unique(dwell_times)), 0, f"State {i}")


class TestSemiMarkovTransMatHandling(unittest.TestCase):
    def test_diagonal_trans_mat_elements_are_ignored(self):
        ns = 5
        rng = np.random.default_rng(1)
        trans_mat1 = rng.uniform(size=(ns, ns))
        trans_mat2 = np.copy(trans_mat1)
        trans_mat1 += np.diag(rng.uniform(low=-1, high=1, size=ns))
        smm1 = SemiMarkov(ns, trans_mat=trans_mat1)

        n = 32
        seq1 = smm1.sample(n)

        smm2 = SemiMarkov(ns, trans_mat=trans_mat2)
        seq2 = smm2.sample(n)

        np.testing.assert_equal(seq1, seq2)

    def test_offdiagonal_trans_mat_elements_are_normalized(self):
        ns = 4
        rng = np.random.default_rng(2)
        trans_mat1 = rng.uniform(size=(ns, ns))
        trans_mat1 -= np.diag(np.diag(trans_mat1))

        trans_mat2 = 3.2 * trans_mat1

        smm1 = SemiMarkov(ns, trans_mat=trans_mat1)

        n = 27
        seq1 = smm1.sample(n)

        smm2 = SemiMarkov(ns, trans_mat=trans_mat2)
        seq2 = smm2.sample(n)

        np.testing.assert_equal(seq1, seq2)

    def test_dwell_times_affect_generated_sequence(self):
        ns = 3
        rng = np.random.default_rng(4)
        trans_mat = rng.uniform(size=(ns, ns))

        dwell_times1 = rng.uniform(low=0, high=10, size=ns)
        smm1 = SemiMarkov(ns, trans_mat=trans_mat, dwell_times=dwell_times1)

        n = 100
        seq1 = smm1.sample(n)

        dwell_times2 = rng.uniform(low=0, high=10, size=ns)
        smm2 = SemiMarkov(ns, trans_mat=trans_mat, dwell_times=dwell_times2)
        seq2 = smm2.sample(n)

        self.assertGreater(np.max(np.abs(seq1 - seq2)), 0)

    def test_default_dwell_times_are_equal_to_one(self):
        ns = 3
        rng = np.random.default_rng(3)
        trans_mat = rng.uniform(size=(ns, ns))

        smm1 = SemiMarkov(ns, trans_mat=trans_mat)

        n = 10
        seq1 = smm1.sample(n)

        smm2 = SemiMarkov(ns, trans_mat=trans_mat, dwell_times=1)
        seq2 = smm2.sample(n)

        np.testing.assert_equal(seq1, seq2)


class TestSemiMarkovInitRaisesOnInvalidDwellTimes(unittest.TestCase):
    def test_raises_value_error_if_dwell_times_not_above_dwell_min(self):
        with self.assertRaises(ValueError):
            SemiMarkov(2, dwell_times=[100, 10], min_dwell=20)

    def test_raises_value_error_if_dwell_times_not_below_dwell_max(self):
        with self.assertRaises(ValueError):
            SemiMarkov(2, dwell_times=[100, 10], max_dwell=20)


if __name__ == "__main__":
    unittest.main()
