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
        trans_mat = np.ones((n_components, n_components)) / n_components
        abs_idx = 2
        trans_mat[abs_idx, :] = np.zeros(n_components)
        trans_mat[abs_idx, abs_idx] = 1
        smm = SemiMarkov(n_components, trans_mat=trans_mat)

        n_seq = 10
        n = 16
        for i in range(n_seq):
            seq = smm.sample(n)
            where_abs = (seq == abs_idx).nonzero()[0]

            # make sure we reach absorbing state
            self.assertGreater(len(where_abs), 0)

            # and stay there once we've reached it
            np.testing.assert_equal(seq[where_abs[0] :], abs_idx)

    def test_all_transitions_occur_by_default(self):
        n_components = 3
        smm = SemiMarkov(n_components, rng=4)

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


class TestSemiMarkovRepeatedCallsToSample(unittest.TestCase):
    def setUp(self):
        self.n_components = 4
        self.seed = 5

    def create_pure_markov(self):
        return SemiMarkov(self.n_components, rng=self.seed)

    def test_pure_markov_sample_n1_then_n2_same_as_sample_n1_plus_n2(self):
        smm1 = self.create_pure_markov()
        smm2 = self.create_pure_markov()

        n1 = 32
        n2 = 54

        seq1 = smm1.sample(n1 + n2)

        seq2a = smm2.sample(n1)
        seq2b = smm2.sample(n2)
        seq2 = np.hstack((seq2a, seq2b))

        np.testing.assert_equal(seq1, seq2)


if __name__ == "__main__":
    unittest.main()
