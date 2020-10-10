import unittest

import h5py

import numpy as np

from bioslds.hdf import write_dict_hierarchy


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


if __name__ == '__main__':
    unittest.main()
