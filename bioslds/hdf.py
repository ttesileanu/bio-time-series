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
    group: h5py.Group, obj: Any
):
    """ Write an object with all its sub-objects to an HDF file.

    This first skips all attributes that start with an underscore or that are callable.
    It then writes as datasets all attributes that are either numbers, strings, or
    numeric Numpy arrays (including boolean). Scalars -- numbers and strings -- are
    stored as `h5py` attributes. Non-numeric Numpy arrays are stored as lists (see
    below). Everything that is not a scalar is stored as either a dataset or a group.
    The function recursively goes through object attributes, writing them to file in a
    hierarchical fashion.

    The following types are treated in a special way:
    1. Instances of `tuple` or `list`. Each item is stored using a name of the form
       "_idx", and a special attribute called "_special_type" is set to "tuple" or
       "list", respectively. A special attribute called "_len" is set to the number of
       elements.
    2. Instances of `dict`. This is stored as a list of tuples containing the
       dictionary's items (that is, `list(d.items())`). The "_special_type" is set to
       "dict".
    3. Sequences, identified either by an iterable interface (i.e., has an `__iter__`
       method) or by a random-access interface based on `__getitem__`. These are stored
       as lists.
    In each of the cases above, any additional non-callable attributes that do not start
    with an underscore are also stored.

    In all cases, the string representation of the object's Python `type` is stored as a
    string attribute called "_type".

    Because of the dynamic way in which Python processes attribute access, it is
    entirely possible that accessing an attribute is a non-trivial operation that could
    even potentially fail. For example, trying to access the `cffi` attribute of an
    `np.BitGenerator` can raise `ImportError`. For this reason, in this function we
    catch any exceptions raised while accessing an attribute, and silently ignore the
    attributes that fail to be accessed.

    Parameters
    ----------
    group
        HDF group where to save the data.
    obj
        The object to save.
    """
    # store dictionaries as lists of (key, value) pairs
    if isinstance(obj, dict):
        write_object_hierarchy(group, list(obj.items()))

        # but ensure that the type indicates that this was a dict
        group.attrs.create("_special_type", "dict")
        group.attrs.create("_type", str(type(obj)))

        return

    # store the object's type
    group.attrs.create("_type", str(type(obj)))

    # get the non-private attributes
    attrib_names = [_ for _ in dir(obj) if not _.startswith("_")]

    # ...but add some special and dummy attributes for sequences
    is_seq = hasattr(obj, "__getitem__")
    is_iter = hasattr(obj, "__iter__")
    elem_list = None
    if is_seq or is_iter:
        # store a special type, and the sequence length
        special_type = "list"
        if isinstance(obj, tuple):
            special_type = "tuple"
        elif isinstance(obj, set):
            special_type = "set"

        group.attrs.create("_special_type", special_type)
        if not is_seq or not hasattr(obj, "__len__"):
            elem_list = [_ for _ in obj]

        if elem_list is None:
            n = len(obj)
        else:
            n = len(elem_list)

        group.attrs.create("_len", n)

        # add attributes for each element
        attrib_names.extend(f"_{_}" for _ in range(n))

    for attrib_name in attrib_names:
        if attrib_name.startswith("_"):
            # handle the special attributes for sequences
            idx = int(attrib_name[1:])
            crt_obj = obj if is_seq else elem_list
            # noinspection PyBroadException
            try:
                attrib = crt_obj[idx]
            except Exception:
                # bail out if getitem fails for any reason
                continue
        else:
            # otherwise get attribute value
            # noinspection PyBroadException
            try:
                attrib = getattr(obj, attrib_name)
            except Exception:
                # bail out if getattr fails for any reason
                continue

        # skip callable attributes
        if callable(attrib):
            continue

        # store single numbers or strings as attributes
        if isinstance(attrib, numbers.Number) or isinstance(attrib, str):
            group.attrs.create(attrib_name, attrib)
        else:
            # store numeric Numpy arrays (including boolean) as datasets
            is_array = isinstance(attrib, np.ndarray)
            is_number_array = is_array and np.issubdtype(attrib.dtype, np.number)
            is_bool_array = is_array and np.issubdtype(attrib.dtype, np.bool_)
            if is_number_array or is_bool_array:
                group.create_dataset(attrib_name, data=np.atleast_1d(attrib))
            else:
                # store everything else as a sub-object

                # in particular, store non-numeric Numpy arrays as lists
                if is_array:
                    attrib = list(attrib)

                sub_group = group.create_group(attrib_name)
                write_object_hierarchy(sub_group, attrib)


def read_namespace_hierarchy(group: h5py.Group) -> SimpleNamespace:
    """ Recurse through an HDF's group structure, and return it as a nested namespace.

    This acts as a converse to `write_object_hierarchy`. While it does not attempt to
    create instances of the appropriate objects beyond a few special cases (see below),
    it returns a hierarchy that can be accessed in the same way as the original object
    did before saving to HDF.

    The group's attributes, not only its datasets, are also stored in the returned
    namespace. If an attribute name conflicts with a dataset's name, it is prefixed by
    "attr_". If this prefixed version of the name also conflicts, it is ignored.

    The function treats `tuple`s, `list`s, `set`s, and `dict`s in a special way. It
    looks for an attribute called "_special_type" in every group. If this exists and is
    equal to "tuple", "list", "set", or "dict", it attempts to load that group as the
    respective Python type. Dictionaries are assumed to be saved as lists of tuples, so
    the function attempts to read an object with `special_type == "dict"` as a list and
    then cast that into a `dict`.

    For reading a sequence to work, an attribute called "_len" must exist, indicating
    the number of elements in the sequence. Then for each index `idx` from 0 to this
    length (excluding the length), the function looks for either a dataset or an
    attribute called "_idx", and assigns this as the corresponding element in the list.
    Missing elements are replaced by `None`, except for a `set`, where the numbering has
    no effect.

    If an object has `_special_type == "dict"` but is not stored as a list of tuples
    with length 2, the list itself is returned instead.

    Parameters
    ----------
    group
        HDF group from where to read.

    Returns a nested `SimpleNamespace` with the contents of the HDF group.
    """
    # check if this is a special type
    if "_special_type" in group.attrs:
        special_type = group.attrs["_special_type"]
    else:
        special_type = ""

    has_len = "_len" in group.attrs
    is_seq_or_dict = special_type in ["tuple", "list", "set", "dict"] and has_len
    if is_seq_or_dict:
        n = group.attrs["_len"]
        d = []
        for i in range(n):
            crt_name = f"_{i}"
            crt_value = None
            if crt_name in group:
                crt_value = group[crt_name]
                if isinstance(crt_value, h5py.Group):
                    crt_value = read_namespace_hierarchy(crt_value)
                else:
                    crt_value = crt_value[()]
            elif crt_name in group.attrs:
                crt_value = group.attrs[crt_name]

            if crt_value is not None or special_type != "set":
                d.append(crt_value)

        if special_type == "tuple":
            d = tuple(d)
        elif special_type == "set":
            d = set(d)
        elif special_type == "dict":
            if all(isinstance(_, tuple) and len(_) == 2 for _ in d):
                d = dict(d)
    else:
        d = SimpleNamespace()
        for key in group.keys():
            value = group[key]
            if not isinstance(value, h5py.Group):
                value = value[()]
                if np.issubdtype(value.dtype, np.string_) and len(value) == 1:
                    value = value[0]
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
                key = "attr_" + key
                if not hasattr(d, key):
                    setattr(d, key, value)

    return d
