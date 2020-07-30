import unittest

import numpy as np

from bioslds.arma_hsmm import sample_switching_models
from bioslds.arma import Arma
from bioslds import sources


class TestSampleSwitchingModelsBasic(unittest.TestCase):
    def test_raises_if_no_models(self):
        with self.assertRaises(ValueError):
            sample_switching_models([], [])

    def test_raises_if_invalid_model_indices(self):
        with self.assertRaises(ValueError):
            sample_switching_models([object()], [0, 1, 1])

    def test_raises_if_length_u_shorter_than_usage_seq(self):
        with self.assertRaises(ValueError):
            sample_switching_models([object()], [0, 0, 0], U=[0.0, 0.3])


class TestSampleSwitchingModelsY(unittest.TestCase):
    def test_single_model(self):
        n = 20
        a = [0.9]
        b = [0.1, -0.2]

        arma1 = Arma(a, b, default_source=sources.GaussianNoise())
        seq_exp, _ = arma1.transform(n)

        arma2 = Arma(a, b, default_source=sources.GaussianNoise())
        seq, _ = sample_switching_models([arma2], np.zeros(n, dtype=int))

        np.testing.assert_allclose(seq, seq_exp)

    def test_second_model_history_is_not_default(self):
        n1 = 13
        n2 = 23

        # these will get overridden
        seq = [0]
        seq1_exp = [1]
        seq2_exp = [2]
        for i in range(2):
            arma1 = Arma(
                [0.9], [0.1, -0.2], default_source=sources.GaussianNoise()
            )
            arma2 = Arma(
                [0.1, -0.2], [0.3], default_source=sources.GaussianNoise()
            )

            if i == 0:
                seq1_exp, _ = arma1.transform(n1)
                seq2_exp, _ = arma2.transform(n2)
            else:
                seq, _ = sample_switching_models(
                    [arma1, arma2],
                    np.hstack(
                        (np.zeros(n1, dtype=int), np.ones(n2, dtype=int))
                    ),
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

        seq, U = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
        )

        exp_ar = np.dot(np.flip(a2), seq[n1 - len(a2) : n1])
        exp_ma = np.dot(np.flip(b2), U[n1 - len(b2) : n1]) + U[n1]

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

            seq, _ = sample_switching_models([arma1, arma2], usage_seq)
            seqs.append(seq)

        np.testing.assert_allclose(seqs[0], seqs[1])

    def test_initial_conditions_parameter_is_obeyed(self):
        a = [0.8]
        b = [0.1, 0.2]

        n = 32

        ic = ([-0.5], [-0.5, 0.3])
        arma = Arma(
            a, b, default_source=sources.GaussianNoise(), initial_conditions=ic
        )

        seq_exp, _ = arma.transform(n)

        arma = Arma(a, b, default_source=sources.GaussianNoise())
        seq, _ = sample_switching_models(
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

            arma1 = Arma(
                [0.9], [0.1, -0.2], default_source=sources.GaussianNoise(),
            )
            arma2 = Arma(
                [0.1, -0.2],
                [0.3, 0.4, 0.5],
                default_source=sources.GaussianNoise(),
            )

            seq, _ = sample_switching_models(
                [arma1, arma2], usage_seq, initial_conditions=ic
            )
            seqs.append(seq)

        np.testing.assert_allclose(seqs[0], seqs[1])


class TestSampleSwitchingModelsU(unittest.TestCase):
    def test_single_model(self):
        n = 20
        rng = np.random.default_rng(1)
        U = rng.normal(size=n)

        a = [0.9]
        b = [0.1, -0.2]

        arma = Arma(a, b)
        _, U_ret = sample_switching_models([arma], np.zeros(n, dtype=int), U=U)

        np.testing.assert_allclose(U, U_ret)

    def test_multiple_models(self):
        n1 = 13
        n2 = 23

        rng = np.random.default_rng(2)
        U = rng.normal(size=n1 + n2)

        arma1 = Arma([0.9], [0.1, -0.2])
        arma2 = Arma([0.1, -0.2], [0.3])
        _, U_ret = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
            U=U,
        )

        np.testing.assert_allclose(U, U_ret)

    def test_multiple_models_callable_source(self):
        n1 = 13
        n2 = 23

        seed = 12
        src = sources.GaussianNoise(seed)

        U_exp = src(size=n1 + n2)

        arma1 = Arma([0.9], [0.1, -0.2])
        arma2 = Arma([0.1, -0.2], [0.3])
        _, U_ret = sample_switching_models(
            [arma1, arma2],
            np.hstack((np.zeros(n1, dtype=int), np.ones(n2, dtype=int))),
            U=sources.GaussianNoise(seed),
        )

        np.testing.assert_allclose(U_exp, U_ret)


if __name__ == "__main__":
    unittest.main()
