import unittest

import numpy as np

from unittest.mock import Mock

from bioslds.hyperopt import random_maximize


class TestRandomMaximizeObjectiveFunctionCalls(unittest.TestCase):
    def setUp(self):
        self.fct = Mock(return_value=0)
        self.n_iter = 40
        self.param_ranges = {"foo": (1, 3), "bar": (-1.0, 5.3)}
        self.param_types = {"foo": np.integer, "bar": np.inexact}
        random_maximize(self.fct, self.param_ranges, self.n_iter)

    def test_fct_called_n_iter_times(self):
        self.assertEqual(self.fct.call_count, self.n_iter)

    def test_fct_called_with_no_positional_args(self):
        for call in self.fct.call_args_list:
            self.assertEqual(len(call.args), 0)

    def test_fct_called_with_all_necessary_kwargs(self):
        for call in self.fct.call_args_list:
            for key in self.param_ranges:
                self.assertIn(key, call.kwargs)

    def test_fct_not_called_with_extraneous_kwargs(self):
        for call in self.fct.call_args_list:
            for key in call.kwargs:
                self.assertIn(key, self.param_ranges)

    def test_generated_param_values_are_in_correct_range(self):
        for call in self.fct.call_args_list:
            for key, (lo, hi) in self.param_ranges.items():
                value = call.kwargs[key]
                self.assertGreaterEqual(value, lo)
                self.assertLess(value, hi)

    def test_generated_param_values_have_correct_type(self):
        for call in self.fct.call_args_list:
            for key, value_type in self.param_types.items():
                value = np.asarray([call.kwargs[key]])
                self.assertTrue(np.issubdtype(value.dtype, value_type))


class TestRandomMaximizeRngCalls(unittest.TestCase):
    def setUp(self):
        self.n_iter = 40
        self.param_ranges = {"foo": (1, 3), "bar": (-1.0, 5.3)}
        self.param_types = {"foo": np.integer, "bar": np.inexact}

    def test_rng_seed_uses_default_rng(self):
        seed = 10

        fct1 = Mock(return_value=0)
        random_maximize(fct1, self.param_ranges, self.n_iter, rng=seed)

        fct2 = Mock(return_value=0)
        random_maximize(
            fct2, self.param_ranges, self.n_iter, rng=np.random.default_rng(seed)
        )

        for call1, call2 in zip(fct1.call_args_list, fct2.call_args_list):
            for key in self.param_ranges:
                value1 = call1.kwargs[key]
                value2 = call2.kwargs[key]

                self.assertAlmostEqual(value1, value2)

    def test_rng_as_numpy_random_state_works(self):
        fct = Mock(return_value=0)
        random_maximize(
            fct, self.param_ranges, self.n_iter, rng=np.random.RandomState(0)
        )

        fct.assert_called()


class TestRandomMaximizeProgress(unittest.TestCase):
    def test_progress_called(self):
        mock_progress = Mock(side_effect=lambda it: it)
        fct = Mock(return_value=0)

        random_maximize(fct, {"foo": (-1.2, 3.4)}, 52, progress=mock_progress)

        mock_progress.assert_called()


class TestRandomMaximizeReturnValues(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n_iter = 50
        self.fct_values = self.rng.normal(size=self.n_iter)

        self.fct = Mock()
        self.fct.side_effect = [
            (fct_value, {"a": 2 * fct_value, "b": 3.0}) for fct_value in self.fct_values
        ]

        self.param_ranges = {"foo": (0, 3), "bar": (-1.3, 5.2)}
        self.out = random_maximize(self.fct, self.param_ranges, self.n_iter)

    def test_first_output_is_max(self):
        expected_max = np.max(self.fct_values)
        self.assertAlmostEqual(self.out[0], expected_max)

    def test_second_output_is_argmax(self):
        # noinspection PyTypeChecker
        i: int = np.argmax(self.fct_values)
        call_i = self.fct.call_args_list[i]

        for key in self.param_ranges:
            value = self.out[1][key]
            value_exp = call_i.kwargs[key]

            self.assertAlmostEqual(value, value_exp)

    def test_third_output_params_contains_used_parameter_sets(self):
        for i, call in enumerate(self.fct.call_args_list):
            param_iter = self.out[2]["params"][i]
            for key in self.param_ranges:
                value = param_iter[key]
                value_exp = call.kwargs[key]

                self.assertAlmostEqual(value, value_exp)

    def test_third_output_fct_scalar_contains_scalar_fct_values(self):
        np.testing.assert_allclose(self.out[2]["fct_scalar"], self.fct_values)

    def test_third_output_fct_full_contains_full_fct_outputs(self):
        for i, fct_exp in enumerate(self.fct.side_effect):
            fct_full = self.out[2]["fct_full"][i]
            self.assertEqual(fct_full, fct_exp)


class TestRandomMaximizeWithScalarFunction(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.n_iter = 50
        self.fct_values = self.rng.normal(size=self.n_iter)

        self.fct = Mock()
        self.fct.side_effect = self.fct_values

        self.param_ranges = {"foo": (0, 3), "bar": (-1.3, 5.2)}
        self.out = random_maximize(self.fct, self.param_ranges, self.n_iter)

    def test_first_output_is_max(self):
        expected_max = np.max(self.fct_values)
        self.assertAlmostEqual(self.out[0], expected_max)

    def test_second_output_is_argmax(self):
        # noinspection PyTypeChecker
        i: int = np.argmax(self.fct_values)
        call_i = self.fct.call_args_list[i]

        for key in self.param_ranges:
            value = self.out[1][key]
            value_exp = call_i.kwargs[key]

            self.assertAlmostEqual(value, value_exp)

    def test_third_output_fct_scalar_contains_scalar_fct_values(self):
        np.testing.assert_allclose(self.out[2]["fct_scalar"], self.fct_values)

    def test_third_output_fct_full_contains_full_fct_outputs(self):
        for i, fct_exp in enumerate(self.fct.side_effect):
            fct_full = self.out[2]["fct_full"][i]
            self.assertEqual(fct_full, fct_exp)


if __name__ == "__main__":
    unittest.main()
