import unittest

import numpy as np

from unittest import mock
from types import SimpleNamespace

from bioslds.chain import ChainMixin
from bioslds.monitor import AttributeMonitor
from bioslds.sources import Stream


class TestChainSetupNDimensions(unittest.TestCase):
    def setUp(self):
        class MockTransformerChainSimple(ChainMixin):
            def __init__(self, n1: int, n2: int, n3: int):
                self.trafo1 = mock.Mock(n_features=n1, n_components=n2)
                self.trafo2 = mock.Mock(n_features=n2, n_components=n3)

                self._transformer_names = ["trafo1", "trafo2"]
                super().__init__()

        self.ns = [2, 4, 3]
        self.chain = MockTransformerChainSimple(*self.ns)

    def test_n_features_set_from_first_transformer(self):
        self.assertEqual(self.chain.n_features, self.ns[0])

    def test_n_components_set_from_last_transformer(self):
        self.assertEqual(self.chain.n_components, self.ns[-1])

    def test_transformers_is_created(self):
        self.assertTrue(hasattr(self.chain, "_transformers"))
        self.assertEqual(len(self.chain._transformers), 2)

    def test_transformers_is_correct(self):
        self.assertIs(self.chain._transformers[0], self.chain.trafo1)
        self.assertIs(self.chain._transformers[1], self.chain.trafo2)


class MockTransformerChain(ChainMixin):
    def __init__(self, *trafos):
        n = len(trafos)
        self._transformer_names = [f"trafo{_}" for _ in range(n)]

        for i, name in enumerate(self._transformer_names):
            setattr(self, name, trafos[i])

        super().__init__()


class TestChainTransform(unittest.TestCase):
    def setUp(self):
        self.n_trafos = 3
        self.rng = np.random.default_rng(1)

        self.ns = [2, 4, 5, 3]
        self.Ws = [
            self.rng.normal(size=(n1, n2)) for n1, n2 in zip(self.ns, self.ns[1:])
        ]

        # noinspection PyUnusedLocal
        def trafo_transform(X, W, chunk_hint=None, monitor=None):
            y = X @ W
            if monitor is not None:
                monitor.setup(len(X))
                obj = SimpleNamespace(input_=X, output_=y)
                monitor.record_batch(obj)

            return y

        self.trafos = [
            mock.Mock(
                n_features=W.shape[0],
                n_components=W.shape[1],
                transform=mock.Mock(
                    side_effect=lambda W=W, **kwargs: trafo_transform(W=W, **kwargs),
                ),
            )
            for W in self.Ws
        ]

        self.chain = MockTransformerChain(*self.trafos)

        self.n_samples = 53
        self.x0 = self.rng.normal(size=(self.n_samples, self.ns[0]))

    def test_all_transforms_called(self):
        self.chain.transform(X=self.x0)
        for trafo in self.chain._transformers:
            trafo.transform.assert_called()

    def test_output_correct(self):
        xf = self.chain.transform(X=self.x0)

        xf_exp = np.copy(self.x0)
        for W in self.Ws:
            xf_exp = xf_exp @ W

        np.testing.assert_allclose(xf, xf_exp)

    def test_passes_chunk_hint_to_transformers(self):
        chunk_hint = 13
        self.chain.transform(X=self.x0, chunk_hint=chunk_hint)
        for trafo in self.chain._transformers:
            self.assertIn("chunk_hint", trafo.transform.call_args[1])
            self.assertEqual(trafo.transform.call_args[1]["chunk_hint"], chunk_hint)

    def test_monitor_dispatches_attributes_according_to_trafo_name(self):
        monitor = AttributeMonitor(["trafo0.input_", "trafo1.input_"])
        self.chain.transform(X=self.x0, monitor=monitor)

        for trafo in self.chain._transformers[:-1]:
            self.assertEqual(1, len(trafo.transform.call_args[1]["monitor"].names))
            self.assertIn("input_", trafo.transform.call_args[1]["monitor"].names)

        self.assertNotIn("monitor", self.chain._transformers[-1].transform.call_args[1])

    def test_monitor_as_sequence(self):
        names = ["trafo0.output_", "trafo2.input_"]
        monitor = AttributeMonitor(names)
        self.chain.transform(X=self.x0, monitor=monitor)

        _, history = self.chain.transform(X=self.x0, monitor=names)

        np.testing.assert_allclose(
            monitor.history_.trafo0.output_, history.trafo0.output_
        )
        np.testing.assert_allclose(
            monitor.history_.trafo2.input_, history.trafo2.input_
        )

    def test_chunk_hint_sets_size_of_chunks_sent_to_transformers(self):
        chunk_hint = 12
        self.chain.transform(X=self.x0, chunk_hint=chunk_hint)

        n_total = 0
        for call in self.chain._transformers[0].transform.call_args_list[:-1]:
            crt_n = len(call[1]["X"])
            self.assertEqual(chunk_hint, crt_n)
            n_total += crt_n

        n_total += len(
            self.chain._transformers[-1].transform.call_args_list[-1][1]["X"]
        )
        self.assertEqual(self.n_samples, n_total)

    def test_chunk_hint_does_not_change_result(self):
        res1 = self.chain.transform(X=self.x0, chunk_hint=11)
        res2 = self.chain.transform(X=self.x0, chunk_hint=23)

        np.testing.assert_allclose(res1, res2)

    def test_progress_called(self):
        mock_progress = mock.MagicMock()

        self.chain.transform(X=self.x0, progress=mock_progress)
        mock_progress.assert_called()

    def test_with_regressor_first_transformer(self):
        reg_trafo = mock.Mock(
            n_features=self.ns[0],
            n_components=self.ns[0],
            transform=mock.Mock(side_effect=lambda X, y, **kwargs: X + y),
        )
        chain_alt = MockTransformerChain(reg_trafo, *self.trafos)

        res = self.chain.transform(X=self.x0)
        res_alt = chain_alt.transform(X=-self.x0, y=2 * self.x0)

        np.testing.assert_allclose(res, res_alt)

    def test_with_generator_first_transformer(self):
        stream = Stream(self.x0)
        gen_trafo = mock.Mock(
            n_features=0,
            n_components=self.ns[0],
            transform=mock.Mock(
                side_effect=lambda n_samples, **kwargs: stream(n_samples)
            ),
        )
        chain_alt = MockTransformerChain(gen_trafo, *self.trafos)

        res = self.chain.transform(X=self.x0)
        res_alt = chain_alt.transform(n_samples=self.n_samples, chunk_hint=10)

        np.testing.assert_allclose(res, res_alt)


if __name__ == "__main__":
    unittest.main()
