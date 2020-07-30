""" Define various utility functions. """

import numpy as np

from typing import Sequence


def rle_encode(seq: Sequence) -> list:
    """ Run-length-encode a sequence.

    Converts a sequence to a list of runs of the same element.

    Parameters
    ----------
    seq
        Sequence to convert.

    Returns a list of tuples of the form `(elem, n_rep)` indicating rows of
    elements where the value `elem` repeats `n_rep` times.
    """
    if len(seq) == 0:
        return []

    starts = np.hstack(([0], np.diff(seq).nonzero()[0] + 1, len(seq)))
    rle = [(seq[i1], i2 - i1) for i1, i2 in zip(starts, starts[1:])]

    return rle


def rle_decode(seq_rle: Sequence) -> list:
    """ Decode a sequence from its run-length encoding.

    Takes in a sequence of tuples `(elem, n_rep)`, as returned by `rle_encode`,
    and returns a list of states corresponding to that encoding.

    Parameters
    ----------
    seq_rle
        Run-length-encoded sequence to convert.

    Returns a sequence of elements whose run-length encoding is `seq_rle`.
    """
    seq = []
    for elem, n_rep in seq_rle:
        seq.extend(n_rep * [elem])

    return seq
