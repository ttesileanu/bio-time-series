""" Define some functions to facilitate serializing to HDF files. """

import h5py

import numpy as np


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
            if hasattr(value, "__len__") or not scalars_as_attribs:
                group.create_dataset(key, data=np.atleast_1d(value))
            else:
                group.attrs.create(key, value)
