""" Define some functions to facilitate serializing to HDF files. """

import h5py

import numbers

import numpy as np

from types import SimpleNamespace
from typing import Any


def write_dict_hierarchy(group: h5py.Group, d: dict, scalars_as_attribs: bool = True):
    """ Write a nested dictionary structure to an HDF file.

    This turns entries that are `dict`s into HDF groups. All other entries need to be
    numbers, numeric arrays, or lists.

    This function is adapted from https://stackoverflow.com/a/44077610.

    Parameters
    ----------
    group
        HDF group where to save the data.
    d
        The data to save.
    scalars_as_attribs
        Single numbers are stored as attributes.
    """
    for key, value in d.items():
        if isinstance(value, dict):
            sub_group = group.create_group(key)
            write_dict_hierarchy(sub_group, value)
        else:
            is_seq = hasattr(value, "__len__")
            is_str = isinstance(value, str)
            if is_str:
                value = np.string_(value)
            if (is_seq and not is_str) or not scalars_as_attribs:
                group.create_dataset(key, data=np.atleast_1d(value))
            else:
                group.attrs.create(key, value)


def read_dict_hierarchy(group: h5py.Group) -> dict:
    """ Recurse through an HDF's group structure, and return it as a nested dictionary.

    This is the converse to `write_dict_hierarchy`. The roundtrip is not perfect: all
    sequences are returned as Numpy arrays.

    The group's attributes are also stored in the dictionary. If an attribute name
    conflicts with a dataset's name, it is prefixed by "attr_". If this prefixed version
    of the name also conflicts, it is ignored.

    Parameters
    ----------
    group
        HDF group from where to read.

    Returns a nested dictionary with the contents of the HDF group.
    """
    d = {}
    for key in group.keys():
        value = group[key]
        if not isinstance(value, h5py.Group):
            value = value[()]
            if np.issubdtype(value.dtype, np.string_) and len(value) == 1:
                value = value[0].decode()
            d[key] = value
        else:
            d[key] = read_dict_hierarchy(value)

    for key in group.attrs.keys():
        value = group.attrs[key]
        if isinstance(value, bytes):
            value = value.decode()
        if key not in d:
            d[key] = value
        else:
            d["attr_" + key] = value

    return d


def write_object_hierarchy(
    group: h5py.Group, obj: Any, scalars_as_attribs: bool = True
):
    """ Write an object with all its sub-objects to an HDF file.

    This first skips all attributes that start with an underscore or that are callable.
    It then writes as datasets all attributes that are either numbers, strings, lists,
    or Numpy arrays. Instances of `dict` are treating in a special way: its keys are
    treated as attribute names and its values are treated as attributes. The function
    then recursively goes through sub-objects to store those as well. The object's
    `type` is stored as a string attribute called "_type".

    Because of the dynamic way in which Python processes attribute access, it is
    entirely possible that accessing an attribute is a non-trivial operation that could
    even fail. For example, trying to access the `cffi` attribute of an
    `np.BitGenerator` can raise `ImportError`. For this reason, in this function we
    catch any exceptions raised while accessing an attribute, and silently ignore the
    attributes that fail to be accessed.

    Parameters
    ----------
    group
        HDF group where to save the data.
    obj
        The object to save.
    scalars_as_attribs
        Single numbers are stored as attributes.
    """
    if isinstance(obj, dict):
        is_dict = True
        attrib_names = obj.keys()
        group.attrs.create("_type", np.string_("dict"))
    else:
        is_dict = False
        attrib_names = dir(obj)
        group.attrs.create("_type", np.string_(str(type(obj))))

    for attrib_name in attrib_names:
        if attrib_name.startswith("_"):
            continue

        if not is_dict:
            # noinspection PyBroadException
            try:
                attrib = getattr(obj, attrib_name)
            except Exception:
                continue
        else:
            attrib = obj[attrib_name]
        if callable(attrib):
            continue

        is_str = isinstance(attrib, str)
        is_seq = False
        is_scalar = False
        if is_str:
            is_scalar = True
            attrib = np.string_(attrib)
        else:
            is_seq = isinstance(attrib, (list, np.ndarray))
            if is_seq:
                attrib_arr = np.asarray(attrib)
                is_numeric = np.issubdtype(attrib_arr.dtype, np.number)
                is_bool = np.issubdtype(attrib_arr.dtype, np.bool_)
                if not is_numeric and not is_bool:
                    is_seq = False
                else:
                    attrib = attrib_arr
            else:
                if isinstance(attrib, numbers.Number):
                    is_scalar = True

        is_sub_obj = not(is_seq or is_scalar)
        if not is_sub_obj:
            if not is_scalar or not scalars_as_attribs:
                group.create_dataset(attrib_name, data=np.atleast_1d(attrib))
            else:
                group.attrs.create(attrib_name, attrib)
        else:
            sub_group = group.create_group(attrib_name)
            write_object_hierarchy(sub_group, attrib)


def read_namespace_hierarchy(group: h5py.Group) -> SimpleNamespace:
    """ Recurse through an HDF's group structure, and return it as a nested namespace.

    This acts as a converse to `write_object_hierarchy`. While it does not attempt to
    create instances of the appropriate objects (beyond `dict` -- see below), it should
    return a hierarchy that can be accessed in the same way as the original object did
    before saving to HDF.

    The group's attributes are also stored in the namespace. If an attribute name
    conflicts with a dataset's name, it is prefixed by "attr_". If this prefixed version
    of the name also conflicts, it is ignored.

    The function looks for an attribute "_type" in every group. If this is the string
    "dict", that group is loaded as a `dict` instead of a namespace.

    Parameters
    ----------
    group
        HDF group from where to read.

    Returns a nested `SimpleNamespace` with the contents of the HDF group.
    """
    d = SimpleNamespace()
    for key in group.keys():
        value = group[key]
        if not isinstance(value, h5py.Group):
            value = value[()]
            if np.issubdtype(value.dtype, np.string_) and len(value) == 1:
                value = value[0].decode()
            setattr(d, key, value)
        else:
            setattr(d, key, read_namespace_hierarchy(value))

    for key in group.attrs.keys():
        value = group.attrs[key]
        if isinstance(value, bytes):
            value = value.decode()
        if not hasattr(d, key):
            setattr(d, key, value)
        else:
            setattr(d, "attr_" + key, value)

    return d
