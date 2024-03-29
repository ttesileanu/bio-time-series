import unittest

from bioslds.monitor import AttributeMonitor

import copy

from types import SimpleNamespace

import numpy as np


class TestAttributeMonitorInit(unittest.TestCase):
    def setUp(self):
        self.keys = ["foo", "bar", "x_y", "y.x"]
        self.monitor = AttributeMonitor(self.keys)

    def test_constructor_default_step_is_one(self):
        self.assertEqual(self.monitor.step, 1)

    def test_setup_makes_history_keys_for_all_attributes_without_dots(self):
        self.monitor.setup(3)

        is_in_history = [
            key in self.monitor.history_.__dict__ for key in self.keys if "." not in key
        ]
        self.assertTrue(all(is_in_history))

    def test_setup_makes_none_histories_for_all_attributes_without_dots(self):
        self.monitor.setup(4)

        is_none = [
            self.monitor.history_.__dict__[key] is None
            for key in self.keys
            if "." not in key
        ]
        self.assertTrue(all(is_none))

    def test_setup_makes_hierarchical(self):
        self.monitor.setup(1)
        self.assertTrue(hasattr(self.monitor.history_, "y"))
        self.assertTrue(hasattr(self.monitor.history_.y, "x"))


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

        class ToTrack(object):
            def __init__(self):
                self.x = 0.0
                self.v = 0
                self.hi_d = np.array([])
                self.i = 0
                self.i_float = 0.0
                self.b = False
                self.s = ""
                self.obj: object = None
                self.i_again = 0

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
        class ToTrack(object):
            def __init__(self):
                self.x = None

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

    def test_monitor_nothing(self):
        monitor = AttributeMonitor([])
        n = 10
        monitor.setup(n)

        for i in range(n):
            obj = SimpleNamespace()
            monitor.record(obj)

        self.assertEqual(len(monitor.history_.__dict__), 0)


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


class TestAttributeMonitorRecordBatch(unittest.TestCase):
    def setUp(self):
        self.n = 60
        self.names = ["x"]
        self.monitor = AttributeMonitor(self.names)
        self.monitor.setup(self.n)

    def test_raises_value_error_if_not_all_variables_are_the_same_length(self):
        monitor_alt = AttributeMonitor(["x", "y"])
        monitor_alt.setup(3)

        obj = SimpleNamespace(x=np.zeros(3), y=[1])
        with self.assertRaises(ValueError):
            monitor_alt.record_batch(obj)

    def test_one_by_one_same_as_batch_when_step_is_one(self):
        monitor_alt = AttributeMonitor(self.names)
        monitor_alt.setup(self.n)

        rng = np.random.default_rng(0)
        v = rng.normal(size=self.n)

        for i in range(self.n):
            obj = SimpleNamespace(x=v[i])
            self.monitor.record(obj)

        i = 0
        while i < self.n:
            k = min(rng.integers(1, 10), self.n - i)
            crt_v = v[i : i + k]
            obj_batch = SimpleNamespace(x=crt_v)
            monitor_alt.record_batch(obj_batch)

            i += k

        np.testing.assert_allclose(self.monitor.history_.x, monitor_alt.history_.x)

    def test_one_by_one_same_as_batch_when_step_is_not_one(self):
        step = 7
        self.monitor.step = step
        monitor_alt = AttributeMonitor(self.names, step=step)
        monitor_alt.setup(self.n)

        rng = np.random.default_rng(1)
        v = rng.normal(size=self.n)

        for i in range(self.n):
            obj = SimpleNamespace(x=v[i])
            self.monitor.record(obj)

        i = 0
        while i < self.n:
            k = min(rng.integers(1, 10), self.n - i)
            crt_v = v[i : i + k]
            obj_batch = SimpleNamespace(x=crt_v)
            monitor_alt.record_batch(obj_batch)

            i += k

        np.testing.assert_allclose(self.monitor.history_.x, monitor_alt.history_.x)

    def test_store_copies_objects(self):
        lst = [1, 2, 3]
        obj = SimpleNamespace(x=[lst])
        self.monitor.record_batch(obj)

        lst0 = copy.copy(lst)
        lst[2] = -1

        np.testing.assert_allclose(self.monitor.history_.x[0], lst0)

    def test_monitor_nothing(self):
        monitor = AttributeMonitor([])
        monitor.setup(5)

        obj = SimpleNamespace()
        monitor.record_batch(obj)

        self.assertEqual(len(monitor.history_.__dict__), 0)


class TestAttributeMonitorHierarchical(unittest.TestCase):
    def setUp(self):
        self.n = 60
        self.names = ["obj.y"]
        self.rng = np.random.default_rng(0)
        self.y = self.rng.normal(size=self.n)

        self.monitor = AttributeMonitor(self.names)
        self.monitor.setup(self.n)

    def test_record_works_correctly(self):
        for i in range(self.n):
            tracked = SimpleNamespace(obj=SimpleNamespace(y=self.y[i]))
            self.monitor.record(tracked)

        self.assertTrue(hasattr(self.monitor.history_, "obj"))
        self.assertTrue(hasattr(self.monitor.history_.obj, "y"))
        np.testing.assert_allclose(self.monitor.history_.obj.y, self.y)

    def test_record_batch_works_correctly(self):
        chunk_size = 13
        for i in range(0, self.n, chunk_size):
            tracked = SimpleNamespace(obj=SimpleNamespace(y=self.y[i : i + chunk_size]))
            self.monitor.record_batch(tracked)

        self.assertTrue(hasattr(self.monitor.history_, "obj"))
        self.assertTrue(hasattr(self.monitor.history_.obj, "y"))
        np.testing.assert_allclose(self.monitor.history_.obj.y, self.y)


if __name__ == "__main__":
    unittest.main()
