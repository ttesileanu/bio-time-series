import unittest

from bioslds.monitor import AttributeMonitor

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np


class TestAttributeMonitorInit(unittest.TestCase):
    def setUp(self):
        self.keys = ["foo", "bar", "x.y"]
        self.monitor = AttributeMonitor(self.keys)

    def test_constructor_default_step_is_one(self):
        self.assertEqual(self.monitor.step, 1)

    def test_setup_makes_history_keys_for_all_attributes(self):
        self.monitor.setup(3)

        is_in_history = [key in self.monitor.history_.__dict__ for key in self.keys]
        self.assertTrue(all(is_in_history))

    def test_setup_makes_none_histories_for_all_attributes(self):
        self.monitor.setup(4)

        is_none = [self.monitor.history_.__dict__[key] is None for key in self.keys]
        self.assertTrue(all(is_none))


class TestAttributeMonitorTypes(unittest.TestCase):
    def setUp(self):
        self.keys = ["x", "v", "hi_d", "i", "i_float", "b", "s", "obj", "i_again"]
        self.monitor = AttributeMonitor(self.keys)

        self.rng = np.random.default_rng(2)
        self.x_data = [1.0, 2, 0.5, -1, 4, 0, 1, 2.1, -5.0, 3.2, 0.0, 0.5]
        self.n = len(self.x_data)
        self.v_data = self.rng.normal(size=(self.n, 5))
        self.hi_d_data = self.rng.normal(size=(self.n, 5, 3, 2, 4))
        self.i_data = np.arange(self.n)
        self.i_float_data = np.arange(self.n, 0, -1).astype(float)
        self.b_data = np.repeat([True, False], self.n // 2)
        self.s_data = np.repeat(["foo", "bar", "goo"], self.n // 3)
        self.obj_data = self.n * [object()]
        self.i_again_data = [1.0] + [2] * (self.n - 1)

        @dataclass
        class ToTrack:
            x: float = 0
            v: np.ndarray = 0
            hi_d: np.ndarray = 0
            i: int = 0
            i_float: float = 0
            b: bool = False
            s: str = ""
            obj: object = None
            i_again: float = 0

        self.obj = ToTrack()
        self.monitor.setup(self.n)

        for x, v, hi_d, i, i_float, b, s, obj, i_again in zip(
            self.x_data,
            self.v_data,
            self.hi_d_data,
            self.i_data,
            self.i_float_data,
            self.b_data,
            self.s_data,
            self.obj_data,
            self.i_again_data,
        ):
            self.obj.x = x
            self.obj.v = v
            self.obj.hi_d = hi_d
            self.obj.i = i
            self.obj.i_float = i_float
            self.obj.b = b
            self.obj.s = s
            self.obj.obj = obj
            self.obj.i_again = i_again

            self.monitor.record(self.obj)

    def test_store_numeric_scalar(self):
        np.testing.assert_allclose(self.monitor.history_.x, self.x_data)

    def test_store_numeric_array(self):
        np.testing.assert_allclose(self.monitor.history_.v, self.v_data)

    def test_store_high_dimensional_numeric_array(self):
        np.testing.assert_allclose(self.monitor.history_.hi_d, self.hi_d_data)

    def test_stores_ints_as_int(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.i.dtype, np.integer))

    def test_stores_floats_as_float(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.i_float.dtype, np.inexact))

    def test_stores_bool_as_bool(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.b.dtype, np.bool_))

    def test_stores_str_as_object(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.s.dtype, np.object_))

    def test_stores_object_as_object(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.obj.dtype, np.object_))

    def test_type_of_first_entry_decides_history_type(self):
        self.assertTrue(np.issubdtype(self.monitor.history_.i_again.dtype, np.inexact))


class TestAttributeMonitorStep(unittest.TestCase):
    def test_correct(self):
        x_data = [1.0, 2, 0.5, -1, 4, 0, 1, 2.1, -5.0, 3.2, 0.0]
        step = 3

        monitor = AttributeMonitor(["x"], step=step)
        monitor.setup(len(x_data))

        for x in x_data:
            obj = SimpleNamespace(x=x)
            monitor.record(obj)

        np.testing.assert_allclose(monitor.history_.x, x_data[::step])


class TestAttributeMonitorMisc(unittest.TestCase):
    def test_store_copies_objects(self):
        @dataclass
        class ToTrack:
            x: object = None

        obj = ToTrack()

        x_data = [None, [1, 2, 3]]

        monitor = AttributeMonitor(["x"])
        monitor.setup(len(x_data))

        for x in x_data:
            obj.x = x
            monitor.record(obj)

        x0 = list(x_data[1])
        x_data[1][0] = 0

        np.testing.assert_allclose(monitor.history_.x[1], x0)


class TestAttributeMonitorStrAndRepr(unittest.TestCase):
    def setUp(self):
        self.names = ["foo", "bar"]
        self.step = 3
        self.monitor = AttributeMonitor(self.names, self.step)

    def test_str(self):
        s_exp = (
            f"AttributeMonitor(names={str(self.names)}, n_=None, "
            + f"step={self.step}, t_=None)"
        )
        s = str(self.monitor)

        self.assertEqual(s, s_exp)

    def test_str_after_setup(self):
        n = 12
        self.monitor.setup(n)
        s_exp = (
            f"AttributeMonitor(names={str(self.names)}, n_={n}, "
            + f"step={self.step}, t_=0)"
        )
        s = str(self.monitor)

        self.assertEqual(s, s_exp)

    def test_repr(self):
        r_exp = (
            f"AttributeMonitor(names={repr(self.names)}, n_=None, "
            + f"step={self.step}, t_=None, i_=None, history_=None)"
        )
        r = repr(self.monitor)

        self.assertEqual(r, r_exp)

    def test_repr_after_setup(self):
        n = 12
        self.monitor.setup(n)
        exp_hist = SimpleNamespace()
        for name in self.names:
            setattr(exp_hist, name, None)
        r_exp = (
            f"AttributeMonitor(names={repr(self.names)}, n_={n}, "
            + f"step={self.step}, t_=0, i_=0, history_={repr(exp_hist)})"
        )
        r = repr(self.monitor)

        self.assertEqual(r, r_exp)


if __name__ == "__main__":
    unittest.main()
