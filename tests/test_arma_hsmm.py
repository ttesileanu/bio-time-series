import unittest

import numpy as np

from bioslds.arma_hsmm import sample_switching_models, ArmaHSMM
from bioslds.markov import SemiMarkov
from bioslds.arma import Arma
from bioslds import sources


class TestSampleSwitchingModelsBasic(unittest.TestCase):
    def test_raises_if_no_models(self):
        with self.assertRaises(ValueError):
            sample_switching_models([], [])

    def test_raises_if_invalid_model_indices(self):
        with self.assertRaises(ValueError):
            sample_switching_models([object()], [0, 1, 1])

    def test_raises_if_length_x_shorter_than_usage_seq(self):
        with self.assertRaises(ValueError):
            sample_switching_models([object()], [0, 0, 0], X=[0.0, 0.3])


class TestSampleSwitchingModelsY(unittest.TestCase):
    def test_single_model(self):
        n = 20
        a = [0.9]
        b = [0.1, -0.2]

        arma1 = Arma(a, b, default_source=sources.GaussianNoise())
        seq_exp = arma1.transform(n)

        arma2 = Arma(a, b, default_source=sources.GaussianNoise())
        seq = sample_switching_models([arma2], np.zeros(n, dtype=int))

        np.testing.assert_allclose(seq, seq_exp)

    def test_second_model_history_is_not_default(self):
        n1 = 13
        n2 = 23

        # these will get overridden
        seq = [0]
        seq1_exp = [1]
        seq2_exp = [2]
        for i in range(2):
            arma1 = Arma([0.9], [0.1, -0.2], default_source=sources.GaussianNoise())
            arma2 = Arma([0.1, -0.2], [0.3], default_source=sources.GaussianNoise())

            if i == 0:
                seq1_exp = arma1.transform(n1)
                seq2_exp = arma2.transform(n2)
            else:
                seq = sample_switching_models(
                    [arma1, arma2],
                    np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
                )

        np.testing.assert_allclose(seq[:n1], seq1_exp)
        self.assertGreater(np.max(np.abs(seq[n1:] - seq2_exp)), 0)

    def test_correct_history_used_at_switch_point(self):
        n1 = 13
        n2 = 5

        arma1 = Arma([0.9], [0.1, -0.2], default_source=sources.GaussianNoise())

        a2 = [0.1, -0.2]
        b2 = [0.3]
        arma2 = Arma(a2, b2, default_source=sources.GaussianNoise())

        seq, X = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
            return_input=True,
        )

        exp_ar = np.dot(np.flip(a2), seq[n1 - len(a2) : n1])
        exp_ma = np.dot(np.flip(b2), X[n1 - len(b2) : n1]) + X[n1]

        self.assertAlmostEqual(exp_ar + exp_ma, seq[n1])

    def test_model_histories_do_not_matter(self):
        n1 = 5
        n2 = 8

        usage_seq = np.hstack((np.ones(n2, dtype=int), np.zeros(n1, dtype=int)))

        seqs = []
        for i in range(2):
            if i == 0:
                ic1 = ([0.3], [-0.2, 0.2])
                ic2 = ([0.2, 0.1], [-0.1, 0.2, 0.0])
            else:
                ic1 = None
                ic2 = None

            arma1 = Arma(
                [0.9],
                [0.1, -0.2],
                default_source=sources.GaussianNoise(),
                initial_conditions=ic1,
            )
            arma2 = Arma(
                [0.1, -0.2],
                [0.3, 0.4, 0.5],
                default_source=sources.GaussianNoise(),
                initial_conditions=ic2,
            )

            seq = sample_switching_models([arma1, arma2], usage_seq)
            seqs.append(seq)

        np.testing.assert_allclose(seqs[0], seqs[1])

    def test_initial_conditions_parameter_is_obeyed(self):
        a = [0.8]
        b = [0.1, 0.2]

        n = 32

        ic = ([-0.5], [-0.5, 0.3])
        arma = Arma(a, b, default_source=sources.GaussianNoise(), initial_conditions=ic)

        seq_exp = arma.transform(n)

        arma = Arma(a, b, default_source=sources.GaussianNoise())
        seq = sample_switching_models(
            [arma], np.zeros(n, dtype=int), initial_conditions=ic
        )

        np.testing.assert_allclose(seq, seq_exp)

    def test_default_initial_conditions_are_zero(self):
        n1 = 5
        n2 = 8

        usage_seq = np.hstack((np.ones(n2, dtype=int), np.zeros(n1, dtype=int)))

        seqs = []
        for i in range(2):
            if i == 0:
                ic = ([0.0], [0.0, 0.0])
            else:
                ic = None

            arma1 = Arma([0.9], [0.1, -0.2], default_source=sources.GaussianNoise(),)
            arma2 = Arma(
                [0.1, -0.2], [0.3, 0.4, 0.5], default_source=sources.GaussianNoise(),
            )

            seq = sample_switching_models(
                [arma1, arma2], usage_seq, initial_conditions=ic
            )
            seqs.append(seq)

        np.testing.assert_allclose(seqs[0], seqs[1])


class TestSampleSwitchingModelsX(unittest.TestCase):
    def test_single_model(self):
        n = 20
        rng = np.random.default_rng(1)
        X = rng.normal(size=n)

        a = [0.9]
        b = [0.1, -0.2]

        arma = Arma(a, b)
        _, X_ret = sample_switching_models(
            [arma], np.zeros(n, dtype=int), X=X, return_input=True
        )

        np.testing.assert_allclose(X, X_ret)

    def test_multiple_models(self):
        n1 = 13
        n2 = 23

        rng = np.random.default_rng(2)
        X = rng.normal(size=n1 + n2)

        arma1 = Arma([0.9], [0.1, -0.2])
        arma2 = Arma([0.1, -0.2], [0.3])
        _, X_ret = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
            X=X,
            return_input=True,
        )

        np.testing.assert_allclose(X, X_ret)

    def test_multiple_models_callable_source(self):
        n1 = 13
        n2 = 23

        seed = 12
        src = sources.GaussianNoise(seed)

        X_exp = src(size=n1 + n2)

        arma1 = Arma([0.9], [0.1, -0.2])
        arma2 = Arma([0.1, -0.2], [0.3])
        _, X_ret = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
            X=sources.GaussianNoise(seed),
            return_input=True,
        )

        np.testing.assert_allclose(X_exp, X_ret)


class TestArmaHSMM(unittest.TestCase):
    def setUp(self):
        self.arma1 = Arma([0.9], [], default_source=sources.GaussianNoise())
        self.arma2 = Arma([0.2, -0.1], [0.3], default_source=sources.GaussianNoise())

        self.armas = [self.arma1, self.arma2]

    def test_transform_returns_triple_when_both_input_and_usage_seq_requested(self):
        arma_hsmm = ArmaHSMM(self.armas)
        res = arma_hsmm.transform(100, return_input=True, return_usage_seq=True)

        self.assertEqual(len(res), 3)

    def test_transform_returns_pair_when_just_input_requested(self):
        arma_hsmm = ArmaHSMM(self.armas)
        res = arma_hsmm.transform(100, return_input=True)

        self.assertEqual(len(res), 2)

    def test_transform_returns_pair_when_just_usage_seq_requested(self):
        arma_hsmm = ArmaHSMM(self.armas)
        res = arma_hsmm.transform(100, return_usage_seq=True)

        self.assertEqual(len(res), 2)

    def test_transform_second_return_value_is_copy_of_input_when_return_input(self):
        arma_hsmm = ArmaHSMM(self.armas)

        rng = np.random.default_rng(2)
        n = 23

        x = rng.normal(size=n)
        _, x_ret = arma_hsmm.transform(X=x, return_input=True)

        np.testing.assert_allclose(x, x_ret)

    def test_transform_second_retval_is_usage_seq_from_semi_markov_when_ret_usseq(self):
        arma_hsmm = ArmaHSMM(self.armas)

        n = 15
        _, usage_seq = arma_hsmm.transform(n, return_usage_seq=True)

        smm = SemiMarkov(2)
        usage_seq_exp = smm.sample(n)

        np.testing.assert_allclose(usage_seq, usage_seq_exp)

    def test_transform_third_retval_is_usage_seq_when_ret_input_and_ret_usseq(self):
        arma_hsmm = ArmaHSMM(self.armas)

        n = 15
        _, _, usage_seq = arma_hsmm.transform(
            n, return_input=True, return_usage_seq=True
        )

        smm = SemiMarkov(2)
        usage_seq_exp = smm.sample(n)

        np.testing.assert_allclose(usage_seq, usage_seq_exp)

    def test_transform_only_one_retval_by_default(self):
        arma_hsmm = ArmaHSMM(self.armas)

        n = 16
        y_exp, _, _ = arma_hsmm.transform(n, return_input=True, return_usage_seq=True)

        arma1 = Arma([0.9], [], default_source=sources.GaussianNoise())
        arma2 = Arma([0.2, -0.1], [0.3], default_source=sources.GaussianNoise())
        arma_hsmm2 = ArmaHSMM([arma1, arma2])

        y = arma_hsmm2.transform(n)

        np.testing.assert_allclose(y, y_exp)

    def test_transform_first_return_value_matches_sample_switching_models(self):
        arma_hsmm = ArmaHSMM(self.armas)

        n = 15
        y, x, usage_seq = arma_hsmm.transform(
            n, return_input=True, return_usage_seq=True
        )

        y_exp = sample_switching_models(self.armas, usage_seq, X=x)

        np.testing.assert_allclose(y, y_exp)

    def test_additional_init_kwargs_passed_to_semi_markov(self):
        kwargs = dict(max_dwell=20, rng=5)
        arma_hsmm = ArmaHSMM(self.armas, **kwargs)

        n = 15
        _, _, usage_seq = arma_hsmm.transform(
            n, return_input=True, return_usage_seq=True
        )

        smm = SemiMarkov(2, **kwargs)
        usage_seq_exp = smm.sample(n)

        np.testing.assert_allclose(usage_seq, usage_seq_exp)

    def test_initial_conditions_obeyed(self):
        arma_hsmm = ArmaHSMM(self.armas)

        n = 15
        ic = ([-0.3, 0.2, 0.8], [0.5, 0.7, 0.1])
        y, x, usage_seq = arma_hsmm.transform(
            n, initial_conditions=ic, return_input=True, return_usage_seq=True
        )

        y_exp = sample_switching_models(
            self.armas, usage_seq, X=x, initial_conditions=ic
        )

        np.testing.assert_allclose(y, y_exp)


class TestArmaHSMMStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.arma1 = Arma([0.9], [], default_source=sources.GaussianNoise())
        self.arma2 = Arma([0.2, -0.1], [0.3], default_source=sources.GaussianNoise())
        self.armas = [self.arma1, self.arma2]
        self.arma_hsmm = ArmaHSMM(self.armas, min_dwell=3, dwell_times=5)

    def test_str(self):
        s = str(self.arma_hsmm)
        s_exp = "ArmaHSMM(models={}, smm={})".format(
            str(self.armas), str(self.arma_hsmm.smm),
        )

        self.assertEqual(s, s_exp)

    def test_repr(self):
        r = repr(self.arma_hsmm)
        r_exp = "ArmaHSMM(models={}, smm={})".format(
            repr(self.armas), repr(self.arma_hsmm.smm),
        )

        self.assertEqual(r, r_exp)


if __name__ == "__main__":
    unittest.main()
