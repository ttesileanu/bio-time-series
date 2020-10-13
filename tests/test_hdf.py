import unittest

import h5py

import numpy as np

from types import SimpleNamespace

from bioslds.hdf import write_dict_hierarchy, read_dict_hierarchy
from bioslds.hdf import write_object_hierarchy, read_namespace_hierarchy


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
            np.testing.assert_allclose(d["dict"]["subfoo"], subfoo)


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


class TestWriteObjectHierarchyGeneric(unittest.TestCase):
    def setUp(self):
        self.sub_obj = SubObject(np.array([3.5, -0.5]))
        self.obj = AnObject("bar", 5.2, self.sub_obj)
        self.fname = "test_obj_write.hdf5"

        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_structure_is_correct(self):
        with h5py.File(self.fname, "r") as f:
            # scalars are written as attributes
            self.assertNotIn("foo", f)
            self.assertNotIn("bar", f)

            self.assertIn("foo", f.attrs)
            self.assertIn("bar", f.attrs)
            self.assertIn("sub", f)

            # private and callable attributes aren't written
            self.assertNotIn("_foo", f)
            self.assertNotIn("fct", f)

            g = f["sub"]
            self.assertIn("foobar", g)
            self.assertIn("d", g)
            self.assertNotIn("_bar", g)

            h = g["d"]
            self.assertIn("_0", h)
            self.assertIn("_len", h.attrs)

            k = h["_0"]
            self.assertIn("_0", k.attrs)
            self.assertIn("_1", k)
            self.assertIn("_len", k.attrs)

    def test_type_attributes_first_level(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("_type", f.attrs)
            self.assertEqual(f.attrs["_type"], str(type(self.obj)))

    def test_type_attributes_second_level(self):
        with h5py.File(self.fname, "r") as f:
            g = f["sub"]
            self.assertIn("_type", g.attrs)
            self.assertEqual(g.attrs["_type"], str(type(self.sub_obj)))

    def test_type_attribute_dict(self):
        with h5py.File(self.fname, "r") as f:
            h = f["sub/d"]
            self.assertIn("_type", h.attrs)
            self.assertEqual(h.attrs["_type"], str(dict))

    def test_special_type_attribute_dict(self):
        with h5py.File(self.fname, "r") as f:
            h = f["sub/d"]
            self.assertIn("_type", h.attrs)
            self.assertEqual(h.attrs["_special_type"], "dict")

    def test_special_type_attribute_dict_elem(self):
        with h5py.File(self.fname, "r") as f:
            h = f["sub/d/_0"]
            self.assertIn("_type", h.attrs)
            self.assertEqual(h.attrs["_special_type"], "tuple")

    def test_first_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f.attrs["foo"], self.obj.foo)
            np.testing.assert_allclose(f.attrs["bar"], self.obj.bar)

    def test_second_level_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            np.testing.assert_allclose(f["sub/foobar"], self.sub_obj.foobar)

    def test_dict_values_are_correct(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["sub/d/_0/"].attrs["_0"], "foobar")
            np.testing.assert_allclose(f["sub/d/_0/_1"], self.sub_obj.foobar)


class TestWriteObjectHierarchyBoolArray(unittest.TestCase):
    def test_writing_bool_array_works(self):
        foo = np.array([True, False, False, True])

        fname = "test_obj_write_bool.hdf5"
        with h5py.File(fname, "w") as f:
            write_object_hierarchy(f, SimpleNamespace(foo=foo))

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            np.testing.assert_allclose(f["foo"][()], foo)


class TestWriteObjectHierarchyNonNumericNumpyArrayAsLists(unittest.TestCase):
    def test_writing_non_numeric_array_yields_list(self):
        foo = np.array([None, (2, 3), 0.3], dtype=object)

        fname = "test_obj_write_non_numeric_array.hdf5"
        with h5py.File(fname, "w") as f:
            write_object_hierarchy(f, SimpleNamespace(foo=foo))

        with h5py.File(fname, "r") as f:
            self.assertIn("foo", f)
            self.assertTrue(isinstance(f["foo"], h5py.Group))
            self.assertEqual(f["foo"].attrs["_len"], len(foo))


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


class TestWriteObjectHierarchyNonStandardSequenceWithoutLen(unittest.TestCase):
    def setUp(self):
        class NonStdSequence:
            def __init__(self):
                self.length = 2

            def __getitem__(self, item: int) -> int:
                if item < self.length:
                    return item ** 2
                else:
                    raise IndexError()

        self.obj = SimpleNamespace(sub=NonStdSequence())

        self.fname = "test_obj_write_non_std_seq.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_nonstandard_sequence_written_as_group(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("sub", f)
            self.assertTrue(isinstance(f["sub"], h5py.Group))

    def test_nonstandard_sequence_has_list_special_type(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["sub"].attrs["_special_type"], "list")

    def test_nonstandard_sequence_has_correct_length(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["sub"].attrs["_len"], self.obj.sub.length)

    def test_nonstandard_sequence_has_correct_elements(self):
        with h5py.File(self.fname, "r") as f:
            for i, x in enumerate(self.obj.sub):
                self.assertAlmostEqual(f["sub"].attrs[f"_{i}"], x)


class TestWriteObjectHierarchyNonStandardSequenceWithLen(unittest.TestCase):
    def setUp(self):
        class NonStdSequence:
            def __len__(self):
                return 2

            def __getitem__(self, item: int) -> int:
                if item < len(self):
                    return item ** 2
                else:
                    raise IndexError()

        self.obj = SimpleNamespace(sub=NonStdSequence())

        self.fname = "test_obj_write_non_std_seq.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_nonstandard_sequence_has_correct_length(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["sub"].attrs["_len"], len(self.obj.sub))


class TestWriteObjectHierarchyNonStandardIterable(unittest.TestCase):
    def setUp(self):
        class NonStdSequence:
            def __iter__(self):
                return NonStdIterator(self)

        class NonStdIterator:
            def __init__(self, obj):
                self._obj = obj
                self._idx = 0

            def __next__(self):
                if self._idx < 3:
                    value = self._idx ** 3
                    self._idx += 1
                    return value
                else:
                    raise StopIteration

        self.obj = SimpleNamespace(sub=NonStdSequence())

        self.fname = "test_obj_write_non_std_seq.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_nonstandard_sequence_written_as_group(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("sub", f)
            self.assertTrue(isinstance(f["sub"], h5py.Group))

    def test_nonstandard_sequence_has_list_special_type(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f["sub"].attrs["_special_type"], "list")

    def test_nonstandard_sequence_has_correct_length(self):
        with h5py.File(self.fname, "r") as f:
            n = len([_ for _ in self.obj.sub])
            self.assertEqual(f["sub"].attrs["_len"], n)

    def test_nonstandard_sequence_has_correct_elements(self):
        with h5py.File(self.fname, "r") as f:
            for i, x in enumerate(self.obj.sub):
                self.assertAlmostEqual(f["sub"].attrs[f"_{i}"], x)


class TestWriteObjectHierarchySet(unittest.TestCase):
    def setUp(self):
        self.obj = {1, 2, 0.3}

        self.fname = "test_obj_write_set.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_set_has_set_special_type(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f.attrs["_special_type"], "set")

    def test_set_has_correct_length(self):
        with h5py.File(self.fname, "r") as f:
            self.assertEqual(f.attrs["_len"], len(self.obj))

    def test_set_has_correct_elements(self):
        with h5py.File(self.fname, "r") as f:
            for i, x in enumerate(self.obj):
                self.assertAlmostEqual(f.attrs[f"_{i}"], x)


class TestWriteObjectHierarchyTupleOrList(unittest.TestCase):
    def setUp(self):
        self.obj = SimpleNamespace(
            n_tpl=(1, 2, 3), v_tpl=("foo", 3), n_lst=[0, 1, 2], v_lst=["bar", 2.0]
        )

        self.fname = "test_obj_write_tuple_or_list.hdf5"
        with h5py.File(self.fname, "w") as f:
            write_object_hierarchy(f, self.obj)

    def test_numeric_tuple_stored_as_tuple(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("n_tpl", f)
            self.assertTrue(isinstance(f["n_tpl"], h5py.Group))

            self.assertIn("_special_type", f["n_tpl"].attrs)
            self.assertEqual(f["n_tpl"].attrs["_special_type"], "tuple")

    def test_numeric_tuple_contents_correct(self):
        with h5py.File(self.fname, "r") as f:
            g = f["n_tpl"]
            self.assertEqual(g.attrs["_len"], len(self.obj.n_tpl))

            for i, x in enumerate(self.obj.n_tpl):
                crt_name = f"_{i}"
                self.assertIn(crt_name, g.attrs)
                self.assertAlmostEqual(g.attrs[crt_name], x)

    def test_generic_tuple_stored_as_tuple(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("v_tpl", f)
            self.assertTrue(isinstance(f["v_tpl"], h5py.Group))

            self.assertIn("_special_type", f["v_tpl"].attrs)
            self.assertEqual(f["v_tpl"].attrs["_special_type"], "tuple")

    def test_generic_tuple_contents_correct(self):
        with h5py.File(self.fname, "r") as f:
            g = f["v_tpl"]
            self.assertEqual(g.attrs["_len"], len(self.obj.v_tpl))

            for i, x in enumerate(self.obj.v_tpl):
                crt_name = f"_{i}"
                self.assertIn(crt_name, g.attrs)
                if isinstance(x, str):
                    self.assertEqual(g.attrs[crt_name], x)
                else:
                    self.assertAlmostEqual(g.attrs[crt_name], x)

    def test_numeric_list_stored_as_tuple(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("n_lst", f)
            self.assertTrue(isinstance(f["n_lst"], h5py.Group))

            self.assertIn("_special_type", f["n_lst"].attrs)
            self.assertEqual(f["n_lst"].attrs["_special_type"], "list")

    def test_numeric_list_contents_correct(self):
        with h5py.File(self.fname, "r") as f:
            g = f["n_lst"]
            self.assertEqual(g.attrs["_len"], len(self.obj.n_lst))

            for i, x in enumerate(self.obj.n_lst):
                crt_name = f"_{i}"
                self.assertIn(crt_name, g.attrs)
                self.assertAlmostEqual(g.attrs[crt_name], x)

    def test_generic_list_stored_as_tuple(self):
        with h5py.File(self.fname, "r") as f:
            self.assertIn("v_lst", f)
            self.assertTrue(isinstance(f["v_lst"], h5py.Group))

            self.assertIn("_special_type", f["v_lst"].attrs)
            self.assertEqual(f["v_lst"].attrs["_special_type"], "list")

    def test_generic_list_contents_correct(self):
        with h5py.File(self.fname, "r") as f:
            g = f["v_lst"]
            self.assertEqual(g.attrs["_len"], len(self.obj.v_lst))

            for i, x in enumerate(self.obj.v_lst):
                crt_name = f"_{i}"
                self.assertIn(crt_name, g.attrs)
                if isinstance(x, str):
                    self.assertEqual(g.attrs[crt_name], x)
                else:
                    self.assertAlmostEqual(g.attrs[crt_name], x)


class TestReadNamespaceHierarchy(unittest.TestCase):
    def test_read_flat(self):
        d_exp = {"foo": np.array([-0.1, 5]), "bar": [1, 0.2, -3]}
        fname = "test_read_flat.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=d_exp["foo"])
            f.create_dataset("bar", data=d_exp["bar"])

        with h5py.File(fname, "r") as f:
            d = read_namespace_hierarchy(f)

            self.assertTrue(hasattr(d, "foo"))
            self.assertTrue(hasattr(d, "bar"))

            np.testing.assert_allclose(d.foo, d_exp["foo"])
            np.testing.assert_allclose(d.bar, d_exp["bar"])

    def test_read_attribs_as_scalars(self):
        d_exp = {"foo": 3}
        fname = "test_read_scalar_attribs.hdf5"
        with h5py.File(fname, "w") as f:
            f.attrs.create("foo", d_exp["foo"])

        with h5py.File(fname, "r") as f:
            d = read_namespace_hierarchy(f)

            self.assertTrue(hasattr(d, "foo"))
            self.assertAlmostEqual(d.foo, d_exp["foo"])

    def test_conflict_dataset_name_and_attrib_name(self):
        foo = [1, 2, 3]
        attr_foo = -3.5
        fname = "test_read_scalar_attribs_conflict.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=foo)
            f.attrs.create("foo", attr_foo)

        with h5py.File(fname, "r") as f:
            d = read_namespace_hierarchy(f)

            self.assertTrue(hasattr(d, "foo"))
            np.testing.assert_allclose(d.foo, foo)

            self.assertTrue(hasattr(d, "attr_foo"))
            self.assertAlmostEqual(d.attr_foo, attr_foo)

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
            d = read_namespace_hierarchy(f)

            self.assertTrue(hasattr(d, "attr_foo"))
            np.testing.assert_allclose(d.attr_foo, dataset_attr_foo)

    def test_hierarchy(self):
        foo = [1, 2, 0.5]
        subfoo = [3, 2, 1]
        fname = "test_read_hierarchical.hdf5"
        with h5py.File(fname, "w") as f:
            f.create_dataset("foo", data=foo)
            f_sub = f.create_group("sub")
            f_sub.create_dataset("subfoo", data=subfoo)

        with h5py.File(fname, "r") as f:
            d = read_namespace_hierarchy(f)

            self.assertTrue(hasattr(d, "foo"))
            self.assertTrue(hasattr(d, "sub"))

            self.assertTrue(hasattr(d.sub, "subfoo"))
            np.testing.assert_allclose(d.sub.subfoo, subfoo)


class TestReadNamespaceHierarchyListOrTuple(unittest.TestCase):
    def setUp(self):
        self.fname = "test_read_seq.hdf5"
        self.seq = [3, "foo", np.array([1, 2, 5])]

        with h5py.File(self.fname, "w") as f:
            for name, t in {"tpl": "tuple", "lst": "list"}.items():
                f.create_group(name)
                f[name].attrs.create("_special_type", t)
                f[name].attrs.create("_len", len(self.seq))
                f[name].attrs.create("_0", self.seq[0])
                f[name].attrs.create("_1", self.seq[1])
                f[name].create_dataset("_2", data=self.seq[2])

        with h5py.File(self.fname, "r") as f:
            self.loaded = read_namespace_hierarchy(f)

    def test_tuple_loaded_as_tuple(self):
        self.assertTrue(isinstance(self.loaded.tpl, tuple))

    def test_list_loaded_as_list(self):
        self.assertTrue(isinstance(self.loaded.lst, list))

    def test_tuple_elements_correct(self):
        self.assertEqual(self.loaded.tpl[0], self.seq[0])
        self.assertEqual(self.loaded.tpl[1], self.seq[1])
        np.testing.assert_allclose(self.loaded.tpl[2], self.seq[2])

    def test_list_elements_correct(self):
        self.assertEqual(self.loaded.lst[0], self.seq[0])
        self.assertEqual(self.loaded.lst[1], self.seq[1])
        np.testing.assert_allclose(self.loaded.lst[2], self.seq[2])


class TestReadNamespaceHierarchySet(unittest.TestCase):
    def setUp(self):
        self.fname = "test_read_seq.hdf5"
        self.seq = [3, "foo"]

        with h5py.File(self.fname, "w") as f:
            f.create_group("s")
            f["s"].attrs.create("_special_type", "set")
            f["s"].attrs.create("_len", len(self.seq))
            f["s"].attrs.create("_0", self.seq[0])
            f["s"].attrs.create("_1", self.seq[1])

        with h5py.File(self.fname, "r") as f:
            self.loaded = read_namespace_hierarchy(f)

    def test_set_loaded_as_set(self):
        self.assertTrue(isinstance(self.loaded.s, set))

    def test_set_elements_correct(self):
        self.assertEqual(self.loaded.s, set(self.seq))


class TestReadNamespaceHierarchyListIndexErrors(unittest.TestCase):
    def setUp(self):
        self.fname = "test_read_seq.hdf5"
        self.n = 4

        with h5py.File(self.fname, "w") as f:
            f.create_group("lst")
            f["lst"].attrs.create("_special_type", "list")
            f["lst"].attrs.create("_len", self.n)
            f["lst"].attrs.create("_0", 1)
            f["lst"].attrs.create("_2", 1)
            f["lst"].attrs.create("_4", 2)

        with h5py.File(self.fname, "r") as f:
            self.loaded = read_namespace_hierarchy(f)

    def test_missing_elements_set_to_none(self):
        self.assertIsNone(self.loaded.lst[1])
        self.assertIsNone(self.loaded.lst[3])

    def test_additional_elements_ignored(self):
        self.assertEqual(len(self.loaded.lst), 4)


class TestReadNamespaceHierarchySetIndexErrors(unittest.TestCase):
    def setUp(self):
        self.fname = "test_read_seq.hdf5"
        self.n = 4

        with h5py.File(self.fname, "w") as f:
            f.create_group("s")
            f["s"].attrs.create("_special_type", "set")
            f["s"].attrs.create("_len", self.n)
            f["s"].attrs.create("_0", 1)
            f["s"].attrs.create("_2", 2)
            f["s"].attrs.create("_4", 3)

        with h5py.File(self.fname, "r") as f:
            self.loaded = read_namespace_hierarchy(f)

    def test_existing_elements_loaded_correctly(self):
        self.assertIn(1, self.loaded.s)
        self.assertIn(2, self.loaded.s)

    def test_missing_elements_ignored(self):
        self.assertNotIn(None, self.loaded.s)

    def test_additional_elements_ignored(self):
        self.assertNotIn(3, self.loaded.s)


class TestReadNamespaceHierarchyDict(unittest.TestCase):
    def setUp(self):
        self.fname = "test_read_dict.hdf5"
        self.d = {"foo": "foo2", "bar": 3.5, 3: np.array([2, 3]), (2, 3): 5.0}

        with h5py.File(self.fname, "w") as f:
            f.create_group("d")
            f["d"].attrs.create("_special_type", "dict")
            f["d"].attrs.create("_len", len(self.d))
            for i in range(len(self.d)):
                f["d"].create_group(f"_{i}")
                f[f"d/_{i}"].attrs.create("_special_type", "tuple")
                f[f"d/_{i}"].attrs.create("_len", 2)

            f["d/_0"].attrs.create("_0", "foo")
            f["d/_0"].attrs.create("_1", "foo2")

            f["d/_1"].attrs.create("_0", "bar")
            f["d/_1"].attrs.create("_1", 3.5)

            f["d/_2"].attrs.create("_0", 3)
            f["d/_2"].create_dataset("_1", data=self.d[3])

            f["d/_3"].create_group("_0")
            f["d/_3/_0"].attrs.create("_special_type", "tuple")
            f["d/_3/_0"].attrs.create("_len", 2)
            f["d/_3/_0"].attrs.create("_0", 2)
            f["d/_3/_0"].attrs.create("_1", 3)
            f["d/_3"].attrs.create("_1", 5.0)

        with h5py.File(self.fname, "r") as f:
            self.loaded = read_namespace_hierarchy(f)

    def test_set_loaded_as_dict(self):
        self.assertTrue(isinstance(self.loaded.d, dict))

    def test_dict_keys_correct(self):
        self.assertIn("foo", self.loaded.d)
        self.assertIn("bar", self.loaded.d)
        self.assertIn(3, self.loaded.d)
        self.assertIn((2, 3), self.loaded.d)

    def test_dict_size_correct(self):
        self.assertEqual(len(self.loaded.d), 4)

    def test_dict_foo_correct(self):
        self.assertEqual(self.loaded.d["foo"], self.d["foo"])

    def test_dict_bar_correct(self):
        self.assertAlmostEqual(self.loaded.d["bar"], self.d["bar"])

    def test_dict_three_correct(self):
        np.testing.assert_allclose(self.loaded.d[3], self.d[3])

    def test_dict_tuple_correct(self):
        self.assertAlmostEqual(self.loaded.d[(2, 3)], self.d[(2, 3)])


if __name__ == "__main__":
    unittest.main()
