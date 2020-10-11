import unittest

import h5py

import numpy as np

from types import SimpleNamespace

from bioslds.hdf import write_dict_hierarchy, read_dict_hierarchy
from bioslds.hdf import write_object_hierarchy


class TestWriteDictHierarchy(unittest.TestCase):
    def test_write_flat(self):
        d = {"foo": np.array([1, 5]), "bar": [1, 2, 3]}
        fname = "test_write_flat.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d)

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            self.assertIn("bar", f)

            np.testing.assert_allclose(f["foo"][()], d["foo"])
            np.testing.assert_allclose(f["bar"][()], d["bar"])

    def test_write_string_when_scalars_as_attribs_is_false(self):
        d = {"foo": "bar"}

        fname = "test_write_str.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d, scalars_as_attribs=False)

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            self.assertEqual(f["foo"][0].decode(), "bar")

    def test_write_string_when_scalars_as_attribs_is_true(self):
        d = {"foo": "bar"}

        fname = "test_write_str.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d, scalars_as_attribs=True)

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f.attrs)
            self.assertEqual(f.attrs["foo"].decode(), "bar")

    def test_scalars_as_attribs_by_default(self):
        d = {"foo": 3}
        fname = "test_write_scalar_attribs.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d)

        with h5py.File(fname, "r") as f:
            self.assertNotIn("foo", f)
            self.assertIn("foo", f.attrs)
            self.assertAlmostEqual(f.attrs["foo"], d["foo"])

    def test_disable_scalars_as_attribs(self):
        d = {"foo": 3}
        fname = "test_write_scalar_as_not_attribs.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d, scalars_as_attribs=False)

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            np.testing.assert_allclose(f["foo"][()], d["foo"])

    def test_hierarchy(self):
        d1 = {"subfoo": [1, 2.3]}
        d = {"d1": d1}
        fname = "test_write_hierarchical.hdf5"
        with h5py.File(fname, "w") as f:
            write_dict_hierarchy(f, d, scalars_as_attribs=False)

        with h5py.File(fname, "r") as f:
            self.assertIn("d1", f)
            self.assertIn("subfoo", f["d1"])
            np.testing.assert_allclose(f["d1/subfoo"][()], d1["subfoo"])


class TestReadDictHierarchy(unittest.TestCase):
    def test_read_flat(self):
        d_exp = {"foo": np.array([-0.1, 5]), "bar": [1, 0.2, -3]}
        fname = "test_read_flat.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=d_exp["foo"])
            f.create_dataset("bar", data=d_exp["bar"])

        with h5py.File(fname, "r") as f:
            d = read_dict_hierarchy(f)

            self.assertIn("foo", d)
            self.assertIn("bar", d)

            np.testing.assert_allclose(d["foo"], d_exp["foo"])
            np.testing.assert_allclose(d["bar"], d_exp["bar"])

    def test_read_attribs_as_scalars(self):
        d_exp = {"foo": 3}
        fname = "test_read_scalar_attribs.hdf5"
        with h5py.File(fname, "w") as f:
            f.attrs.create("foo", d_exp["foo"])

        with h5py.File(fname, "r") as f:
            d = read_dict_hierarchy(f)

            self.assertIn("foo", d)
            self.assertAlmostEqual(d["foo"], d_exp["foo"])

    def test_conflict_dataset_name_and_attrib_name(self):
        foo = [1, 2, 3]
        attr_foo = -3.5
        fname = "test_read_scalar_attribs_conflict.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=foo)
            f.attrs.create("foo", attr_foo)

        with h5py.File(fname, "r") as f:
            d = read_dict_hierarchy(f)

            self.assertIn("foo", d)
            np.testing.assert_allclose(d["foo"], foo)

            self.assertIn("attr_foo", d)
            self.assertAlmostEqual(d["attr_foo"], attr_foo)

    def test_conflict_attrib_name_with_dataset_name_with_and_without_attr(self):
        foo = [1, 2, 3]
        dataset_attr_foo = [2, 3, 1]
        attr_foo = -3.5
        fname = "test_read_scalar_attribs_conflict.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=foo)
            f.create_dataset("attr_foo", data=dataset_attr_foo)
            f.attrs.create("foo", attr_foo)

        with h5py.File(fname, "r") as f:
            d = read_dict_hierarchy(f)

            self.assertIn("attr_foo", d)
            self.assertAlmostEqual(d["attr_foo"], attr_foo)

    def test_hierarchy(self):
        foo = [1, 2, 0.5]
        subfoo = [3, 2, 1]
        fname = "test_read_hierarchical.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=foo)
            f_dict = f.create_group("dict")
            f_dict.create_dataset("subfoo", data=subfoo)

        with h5py.File(fname, "r") as f:
            d = read_dict_hierarchy(f)

            self.assertIn("foo", d)
            self.assertIn("dict", d)

            self.assertIn("subfoo", d["dict"])
            np.testing.assert_allclose(f["dict/subfoo"][()], subfoo)


class TestWriteReadDictHierarchyRoundtrip(unittest.TestCase):
    def setUp(self):
        self.d1 = {"foo": [1, 2, 3], "bar": 0.3}
        self.d = {
            "foo": [0.2, 0.3],
            "bar": [0.1, 0.2, 0.3],
            "foobar": "foobar",
            "dict": self.d1,
        }

        self.fname = "test_write_read_roundtrip.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_dict_hierarchy(f, self.d)

    def test_read_returns_original_dict_up_to_types(self):
        with h5py.File(self.fname, "r") as f:
            d = read_dict_hierarchy(f)

        self.assertIn("foo", d)
        self.assertIn("bar", d)
        self.assertIn("foobar", d)
        self.assertIn("dict", d)

        np.testing.assert_allclose(d["foo"], self.d["foo"])
        np.testing.assert_allclose(d["bar"], self.d["bar"])
        self.assertEqual(d["foobar"], self.d["foobar"])

        self.assertIn("foo", d["dict"])
        self.assertIn("bar", d["dict"])

        np.testing.assert_allclose(d["dict"]["foo"], self.d1["foo"])
        np.testing.assert_allclose(d["dict"]["bar"], self.d1["bar"])


class TestWriteReadDictHierarchyRoundtripWhenScalarsNotAttribs(unittest.TestCase):
    def setUp(self):
        self.d1 = {"foo": [1, 2, 3], "bar": 0.3}
        self.d = {
            "foo": [0.2, 0.3],
            "bar": [0.1, 0.2, 0.3],
            "foobar": "foobar",
            "dict": self.d1,
        }

        self.fname = "test_write_read_roundtrip.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_dict_hierarchy(f, self.d, scalars_as_attribs=False)

    def test_read_returns_original_dict_up_to_types(self):
        with h5py.File(self.fname, "r") as f:
            d = read_dict_hierarchy(f)

        self.assertIn("foo", d)
        self.assertIn("bar", d)
        self.assertIn("foobar", d)
        self.assertIn("dict", d)

        np.testing.assert_allclose(d["foo"], self.d["foo"])
        np.testing.assert_allclose(d["bar"], self.d["bar"])
        self.assertEqual(d["foobar"], self.d["foobar"])

        self.assertIn("foo", d["dict"])
        self.assertIn("bar", d["dict"])

        np.testing.assert_allclose(d["dict"]["foo"], self.d1["foo"])
        np.testing.assert_allclose(d["dict"]["bar"], self.d1["bar"])


class SubObject:
    def __init__(self, foobar):
        self.foobar = foobar
        self._bar = [1, 2]
        self.d = {"foobar": foobar}


class AnObject:
    def __init__(self, foo, bar, sub):
        self.foo = foo
        self.bar = bar
        self._foo = [5, 4]
        self.sub = sub

    def fct(self):
        pass


class TestWriteObjectHierarchyWhenScalarsNotAttribs(unittest.TestCase):
    def setUp(self):
        self.sub_obj = SubObject([3.5, -0.5])
        self.obj = AnObject("bar", 5.2, self.sub_obj)
        self.fname = "test_obj_write.hdf5"

        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj, scalars_as_attribs=False)

    def test_structure_is_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("foo", f)
            self.assertIn("bar", f)
            self.assertIn("sub", f)
            self.assertNotIn("_foo", f)
            self.assertNotIn("fct", f)

            g = f["sub"]
            self.assertIn("foobar", g)
            self.assertIn("d", g)
            self.assertNotIn("_bar", g)

            h = g["d"]
            self.assertIn("foobar", h)

    def test_type_attributes_first_level(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("_type", f.attrs)
            self.assertEqual(f.attrs["_type"].decode(), str(type(self.obj)))

    def test_type_attributes_second_level(self):
        with h5py.File(self.fname, "r") as f:
            g = f["sub"]
            self.assertIn("_type", g.attrs)
            self.assertEqual(g.attrs["_type"].decode(), str(type(self.sub_obj)))

    def test_type_attributes_dict(self):
        with h5py.File(self.fname, "r") as f:
            h = f["sub/d"]
            self.assertIn("_type", h.attrs)
            self.assertEqual(h.attrs["_type"].decode(), "dict")

    def test_first_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["foo"][0].decode(), self.obj.foo)
            np.testing.assert_allclose(f["bar"][()], self.obj.bar)

    def test_second_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            np.testing.assert_allclose(f["sub/foobar"], self.sub_obj.foobar)

    def test_dict_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            np.testing.assert_allclose(f["sub/d/foobar"], self.sub_obj.foobar)


class TestWriteObjectHierarchySimpleNamespaceWhenScalarsAreAttribs(unittest.TestCase):
    def setUp(self):
        self.obj = SimpleNamespace(foo="foo", sub=SimpleNamespace(bar=5.2))
        self.fname = "test_obj_write.hdf5"

        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj, scalars_as_attribs=True)

    def test_structure_is_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertNotIn("foo", f)
            self.assertIn("foo", f.attrs)
            self.assertIn("sub", f)

            g = f["sub"]
            self.assertNotIn("bar", g)
            self.assertIn("bar", g.attrs)

    def test_first_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f.attrs["foo"].decode(), self.obj.foo)

    def test_second_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            np.testing.assert_allclose(f["sub"].attrs["bar"], self.obj.sub.bar)


class TestWriteObjectHierarchyDefaultScalarsAsAttribs(unittest.TestCase):
    def test_default_scalars_as_attribs_is_true(self):
        fname = "test_obj_write.hdf5"
        with h5py.File(fname, "w") as f:
            write_object_hierarchy(f, SimpleNamespace(foo=3.5))

        with h5py.File(fname, "r") as f:
            self.assertNotIn("foo", f)
            self.assertIn("foo", f.attrs)


class TestWriteObjectHierarchyBoolArray(unittest.TestCase):
    def test_writing_bool_array_works(self):
        foo = [True, False, False, True]

        fname = "test_obj_write_bool.hdf5"
        with h5py.File(fname, "w") as f:
            write_object_hierarchy(f, SimpleNamespace(foo=foo))

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            np.testing.assert_allclose(f["foo"][()], foo)


class TestWriteObjectHierarchyInaccessibleAttribute(unittest.TestCase):
    def test_silently_ignores_inaccessible_attributes(self):
        class AnotherObject:
            def __init__(self):
                self.foo = [1, 2, 3]

            @property
            def bar(self):
                raise KeyError

        obj = AnotherObject()
        fname = "test_obj_write_skip_inaccessible.hdf5"
        with h5py.File(fname, "w") as f:
            write_object_hierarchy(f, obj)

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            self.assertNotIn("bar", f)


if __name__ == "__main__":
    unittest.main()
