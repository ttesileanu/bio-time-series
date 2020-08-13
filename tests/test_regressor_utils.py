import unittest

import numpy as np

from unittest import mock
from bioslds.regressor_utils import fit_infer_ar


class TestFitInferAR(unittest.TestCase):
    def test_calls_regressor_fit_infer_once(self):
        regressor = mock.Mock(n_features=3)
        fit_infer_ar(regressor, [])

        regressor.fit_infer.assert_called_once()

    def test_returns_regressor_fit_infer_output(self):
        regressor = mock.Mock(n_features=3)

        ret = (1, {"a": 3})
        regressor.fit_infer.return_value = ret

        actual_ret = fit_infer_ar(regressor, [])

        self.assertEqual(actual_ret, ret)

    def test_does_not_pass_bias_to_regressor_fit_infer(self):
        regressor = mock.Mock(n_features=3)
        fit_infer_ar(regressor, [], bias=True)

        call_args = regressor.fit_infer.call_args
        self.assertNotIn("bias", call_args.kwargs)

    def test_passes_additional_kwargs_to_regressor_fit_infer(self):
        regressor = mock.Mock(n_features=3)
        kwargs = {"foo": 3, "bar": 5}
        fit_infer_ar(regressor, [], **kwargs)

        call_kwargs = regressor.fit_infer.call_args.kwargs
        for key in kwargs:
            self.assertIn(key, call_kwargs)
            self.assertEqual(kwargs[key], call_kwargs[key])

    def test_calls_regressor_fit_infer_with_two_positional_args(self):
        regressor = mock.Mock(n_features=3)
        fit_infer_ar(regressor, [])

        self.assertEqual(len(regressor.fit_infer.call_args.args), 2)

    def test_args_sent_to_regressor_fit_infer_have_the_correct_length(self):
        p = 4
        regressor = mock.Mock(n_features=p)

        n = 52
        a = n * [0]
        fit_infer_ar(regressor, a)

        call_args = regressor.fit_infer.call_args.args
        self.assertEqual(len(call_args[0]), n - p)
        self.assertEqual(len(call_args[1]), n - p)

    def test_size_of_predictor_variable_correct_when_bias_is_false(self):
        p = 4
        regressor = mock.Mock(n_features=p)

        n = 23
        a = n * [0]
        fit_infer_ar(regressor, a, bias=False)

        call_args = regressor.fit_infer.call_args.args
        self.assertEqual(np.shape(call_args[0])[1], p)

    def test_size_of_predictor_variable_correct_when_bias_is_true(self):
        p = 5
        regressor = mock.Mock(n_features=p + 1)

        n = 20
        a = n * [0]
        fit_infer_ar(regressor, a, bias=True)

        call_args = regressor.fit_infer.call_args.args
        self.assertEqual(np.shape(call_args[0])[1], p + 1)

    def test_bias_false_by_default(self):
        p = 4
        regressor1 = mock.Mock(n_features=p)
        regressor2 = mock.Mock(n_features=p)

        n = 10
        a = n * [0]
        fit_infer_ar(regressor1, a, bias=False)
        fit_infer_ar(regressor2, a)

        call_args1 = regressor1.fit_infer.call_args.args
        call_args2 = regressor2.fit_infer.call_args.args

        self.assertEqual(np.shape(call_args1[0]), np.shape(call_args2[0]))

    def test_y_values_passed_to_regressor_infer_fit_correct_when_bias_is_false(self):
        rng = np.random.default_rng(1)

        n = 57
        y = rng.normal(size=n)

        p = 4
        regressor = mock.Mock(n_features=p)
        fit_infer_ar(regressor, y, bias=False)

        call_args = regressor.fit_infer.call_args.args
        np.testing.assert_allclose(call_args[1], y[p:])

    def test_y_values_passed_to_regressor_infer_fit_correct_when_bias_is_true(self):
        rng = np.random.default_rng(1)

        n = 57
        y = rng.normal(size=n)

        p = 4
        regressor = mock.Mock(n_features=p + 1)
        fit_infer_ar(regressor, y, bias=True)

        call_args = regressor.fit_infer.call_args.args
        np.testing.assert_allclose(call_args[1], y[p:])

    def test_X_values_passed_to_regressor_infer_fit_correct_when_bias_is_false(self):
        rng = np.random.default_rng(1)

        n = 57
        y = rng.normal(size=n)

        p = 4
        regressor = mock.Mock(n_features=p)
        fit_infer_ar(regressor, y, bias=False)

        call_args = regressor.fit_infer.call_args.args

        expected_X = np.zeros((len(y) - p, p))
        for i in range(len(y) - p):
            for j in range(p):
                expected_X[i, j] = y[i - j + p - 1]

        np.testing.assert_allclose(call_args[0], expected_X)

    def test_X_values_augmented_by_constant_one_when_bias_is_true(self):
        rng = np.random.default_rng(2)

        n = 23
        y = rng.normal(size=n)

        p = 5
        regressor1 = mock.Mock(n_features=p)
        regressor2 = mock.Mock(n_features=p + 1)

        fit_infer_ar(regressor1, y, bias=False)
        fit_infer_ar(regressor2, y, bias=True)

        call_args1 = regressor1.fit_infer.call_args.args
        call_args2 = regressor2.fit_infer.call_args.args

        X = call_args1[0]
        X_bias = call_args2[0]
        X_bias_exp = np.hstack((X, np.ones((len(X), 1))))

        np.testing.assert_allclose(X_bias, X_bias_exp)


if __name__ == "__main__":
    unittest.main()
