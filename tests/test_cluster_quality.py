import unittest

import itertools
import numpy as np

from unittest.mock import Mock

from bioslds.cluster_quality import calculate_sliding_score, unordered_accuracy_score


class TestCalculateSlidingScoreBasics(unittest.TestCase):
    def test_returns_empty_if_window_size_larger_than_labels_length(self):
        window_loc, score = calculate_sliding_score(
            lambda x, y: 1, [1, 2, 3], [0, 1, 2], window_size=5
        )

        self.assertEqual(len(window_loc), 0)
        self.assertEqual(len(score), 0)

    def test_returns_empty_for_empty_inputs(self):
        window_loc, score = calculate_sliding_score(lambda x, y: 1, [], [])
        self.assertEqual(len(window_loc), 0)
        self.assertEqual(len(score), 0)


class TestCalculateSlidingScoreLengthMismatchPolicyWhenPredIsShorter(unittest.TestCase):
    def setUp(self):
        self.n_samples_true = 50
        self.n_samples_pred = 45
        self.window_size = 8

        self.rng = np.random.default_rng(0)

        self.labels_true = self.rng.normal(size=self.n_samples_true)
        self.labels_pred = self.rng.normal(size=self.n_samples_pred)

        self.args = (lambda x, y: np.mean(x - y), self.labels_true, self.labels_pred)
        self.kwargs = {"window_size": self.window_size}

    def test_raises_value_error_if_policy_is_raise(self):
        with self.assertRaises(ValueError):
            calculate_sliding_score(
                *self.args, **self.kwargs, length_mismatch_policy="raise",
            )

    def test_align_at_start(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_start"
        )

        n = min(self.n_samples_true, self.n_samples_pred)
        window_loc2, score2 = calculate_sliding_score(
            self.args[0], self.labels_true[:n], self.labels_pred, **self.kwargs
        )

        np.testing.assert_equal(window_loc1, window_loc2)
        np.testing.assert_equal(score1, score2)

    def test_align_at_end(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_end"
        )

        n = min(self.n_samples_true, self.n_samples_pred)
        shift = self.n_samples_true - n
        window_loc2, score2 = calculate_sliding_score(
            self.args[0], self.labels_true[-n:], self.labels_pred, **self.kwargs
        )

        np.testing.assert_equal(window_loc1, window_loc2 + shift)
        np.testing.assert_equal(score1, score2)

    def test_default_is_to_align_at_end(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_end"
        )

        window_loc2, score2 = calculate_sliding_score(*self.args, **self.kwargs)

        np.testing.assert_equal(window_loc1, window_loc2)
        np.testing.assert_equal(score1, score2)


class TestCalculateSlidingScoreLengthMismatchPolicyWhenPredIsLonger(unittest.TestCase):
    def setUp(self):
        self.n_samples_true = 44
        self.n_samples_pred = 51
        self.window_size = 8

        self.rng = np.random.default_rng(1)

        self.labels_true = self.rng.normal(size=self.n_samples_true)
        self.labels_pred = self.rng.normal(size=self.n_samples_pred)

        self.args = (lambda x, y: np.mean(x - y), self.labels_true, self.labels_pred)
        self.kwargs = {"window_size": self.window_size}

    def test_raises_value_error_if_policy_is_raise(self):
        with self.assertRaises(ValueError):
            calculate_sliding_score(
                *self.args, **self.kwargs, length_mismatch_policy="raise",
            )

    def test_align_at_start(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_start"
        )

        n = min(self.n_samples_true, self.n_samples_pred)
        window_loc2, score2 = calculate_sliding_score(
            self.args[0], self.labels_true, self.labels_pred[:n], **self.kwargs
        )

        np.testing.assert_equal(window_loc1, window_loc2)
        np.testing.assert_equal(score1, score2)

    def test_align_at_end(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_end"
        )

        n = min(self.n_samples_true, self.n_samples_pred)
        window_loc2, score2 = calculate_sliding_score(
            self.args[0], self.labels_true, self.labels_pred[-n:], **self.kwargs
        )

        np.testing.assert_equal(window_loc1, window_loc2)
        np.testing.assert_equal(score1, score2)

    def test_default_is_to_align_at_end(self):
        window_loc1, score1 = calculate_sliding_score(
            *self.args, **self.kwargs, length_mismatch_policy="align_end"
        )

        window_loc2, score2 = calculate_sliding_score(*self.args, **self.kwargs)

        np.testing.assert_equal(window_loc1, window_loc2)
        np.testing.assert_equal(score1, score2)


class TestCalculateSlidingScoreOutput(unittest.TestCase):
    def setUp(self):
        self.n_samples = 200
        self.window_size = 13
        self.overlap_fraction = 0.3

        self.rng = np.random.default_rng(0)
        self.metric_output = self.rng.normal(size=self.n_samples)
        self.metric = Mock(side_effect=self.metric_output)

        self.labels_true = self.rng.normal(size=self.n_samples)
        self.labels_pred = self.rng.normal(size=self.n_samples)

        self.window_loc, self.score = calculate_sliding_score(
            self.metric,
            self.labels_true,
            self.labels_pred,
            window_size=self.window_size,
            overlap_fraction=self.overlap_fraction,
        )

    def test_window_loc_step_is_correct(self):
        exp_overlap_steps = int(self.overlap_fraction * self.window_size)
        exp_step = self.window_size - exp_overlap_steps

        steps = np.diff(self.window_loc)
        np.testing.assert_equal(steps, exp_step)

    def test_first_window_loc_is_zero(self):
        self.assertEqual(self.window_loc[0], 0)

    def test_last_window_loc_is_last_possible_within_n_samples(self):
        self.assertLess(self.window_loc[-1], self.n_samples)
        self.assertLessEqual(self.window_loc[-1] + self.window_size, self.n_samples)

        step = self.window_loc[1] - self.window_loc[0]
        self.assertGreater(
            self.window_loc[-1] + step + self.window_size, self.n_samples
        )

    def test_score_output_matches_what_metric_returned(self):
        n = len(self.score)
        np.testing.assert_equal(self.score, self.metric_output[:n])

    def test_metric_called_the_right_number_of_times(self):
        self.assertEqual(self.metric.call_count, len(self.window_loc))

    def test_metric_called_with_appropriate_inputs(self):
        for crt_loc, crt_call in zip(self.window_loc, self.metric.call_args_list):
            self.assertEqual(len(crt_call[1]), 0)
            self.assertEqual(len(crt_call[0]), 2)

            exp_true = self.labels_true[crt_loc : crt_loc + self.window_size]
            exp_pred = self.labels_pred[crt_loc : crt_loc + self.window_size]
            np.testing.assert_equal(crt_call[0][0], exp_true)
            np.testing.assert_equal(crt_call[0][1], exp_pred)


class TestCalculateSlidingScoreMisc(unittest.TestCase):
    def setUp(self):
        self.n_samples = 200
        self.window_size = 13

        self.rng = np.random.default_rng(0)
        self.metric_output = self.rng.normal(size=self.n_samples)
        self.metric = Mock(side_effect=self.metric_output)

        self.labels_true = self.rng.normal(size=self.n_samples)
        self.labels_pred = self.rng.normal(size=self.n_samples)

    def test_step_does_not_go_below_one(self):
        window_loc, _ = calculate_sliding_score(
            self.metric,
            self.labels_true,
            self.labels_pred,
            window_size=self.window_size,
            overlap_fraction=1,
        )

        self.assertGreater(len(window_loc), 0)
        np.testing.assert_equal(np.diff(window_loc), 1)

    def test_custom_step_overrides_fraction(self):
        step = 2
        window_loc, _ = calculate_sliding_score(
            self.metric,
            self.labels_true,
            self.labels_pred,
            window_size=self.window_size,
            step=step,
        )

        self.assertGreater(len(window_loc), 0)
        np.testing.assert_equal(np.diff(window_loc), step)

    def test_progress_called(self):
        mock_progress = Mock(side_effect=lambda it: it)

        calculate_sliding_score(
            self.metric, self.labels_true, self.labels_pred, progress=mock_progress
        )

        mock_progress.assert_called()


class TestUnorderedAccuracyScoreOptimalAssignment(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(12)
        self.n_clusters = 5
        self.n_samples = 43
        self.seq1 = self.rng.integers(0, self.n_clusters, size=self.n_samples)
        self.seq2 = self.rng.integers(0, self.n_clusters, size=self.n_samples)

    def test_identical_returns_identity(self):
        _, best_assignment = unordered_accuracy_score(
            self.seq1, self.seq1, return_assignment=True
        )

        np.testing.assert_equal(best_assignment, np.arange(self.n_clusters))

    def test_arbitrary_permutation_correctly_identified(self):
        perm = self.rng.permutation(self.n_clusters)
        seq1_perm = perm[self.seq1]
        _, best_assignment = unordered_accuracy_score(
            seq1_perm, self.seq1, return_assignment=True
        )

        np.testing.assert_equal(best_assignment, perm)

    def test_score_matches_shuffle(self):
        score, best_assignment = unordered_accuracy_score(
            self.seq1, self.seq2, return_assignment=True
        )
        seq2_shuf = best_assignment[self.seq2]
        # noinspection PyTypeChecker
        score_exp: float = np.mean(self.seq1 == seq2_shuf)

        self.assertAlmostEqual(score_exp, score)

    def test_graceful_fail_if_inputs_are_empty(self):
        _, best_assignment = unordered_accuracy_score([], [], return_assignment=True)
        self.assertEqual(0, len(best_assignment))


class TestUnorderedAccuracyScoreBasic(unittest.TestCase):
    def test_raises_value_error_if_lengths_mismatched(self):
        with self.assertRaises(ValueError):
            unordered_accuracy_score([1], [1, 2])

    def test_graceful_fail_if_inputs_are_empty(self):
        score = unordered_accuracy_score([], [])
        self.assertEqual(score, 1)


class TestUnorderedAccuracyScore(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(13)
        self.n_clusters = 4
        self.n_samples = 46
        self.seq1 = self.rng.integers(0, self.n_clusters, size=self.n_samples)
        self.seq2 = self.rng.integers(0, self.n_clusters, size=self.n_samples)

    def test_identical_returns_score_one(self):
        score = unordered_accuracy_score(self.seq1, self.seq1)

        self.assertAlmostEqual(score, score)

    def test_symmetry(self):
        score1 = unordered_accuracy_score(self.seq1, self.seq2)
        score2 = unordered_accuracy_score(self.seq2, self.seq1)
        self.assertAlmostEqual(score1, score2)

    def test_arbitrary_permutation_yields_score_one(self):
        perm = self.rng.permutation(self.n_clusters)
        seq1_perm = perm[self.seq1]
        score = unordered_accuracy_score(self.seq1, seq1_perm)
        self.assertAlmostEqual(score, 1)

    def test_normalize_divides_by_total_length(self):
        score_count = unordered_accuracy_score(self.seq1, self.seq2, normalize=False)
        score_frac = unordered_accuracy_score(self.seq1, self.seq2, normalize=True)

        self.assertAlmostEqual(score_count, score_frac * self.n_samples)

    def test_normalize_defaults_to_true(self):
        score = unordered_accuracy_score(self.seq1, self.seq2)
        score_frac = unordered_accuracy_score(self.seq1, self.seq2, normalize=True)

        self.assertAlmostEqual(score, score_frac)

    def test_score_matches_count_and_shuffle_when_normalize_is_false(self):
        score_count, best_assignment = unordered_accuracy_score(
            self.seq1, self.seq2, return_assignment=True, normalize=False
        )
        seq2_shuf = best_assignment[self.seq2]
        # noinspection PyTypeChecker
        score_exp: int = np.sum(self.seq1 == seq2_shuf)

        self.assertAlmostEqual(score_exp, score_count)

    def test_score_matches_best_permutation(self):
        score = unordered_accuracy_score(self.seq1, self.seq2)

        perm_scores = []
        for crt_perm in itertools.permutations(np.arange(self.n_clusters)):
            crt_seq2 = np.asarray(crt_perm)[self.seq2]
            perm_scores.append(np.mean(self.seq1 == crt_seq2))

        self.assertAlmostEqual(np.max(perm_scores), score)


if __name__ == "__main__":
    unittest.main()
