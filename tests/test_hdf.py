import unittest

import h5py

import numpy as np

from bioslds.hdf import write_dict_hierarchy, read_dict_hierarchy


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


if __name__ == "__main__":
    unittest.main()
