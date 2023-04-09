# uncompyle6 version 3.9.0
# Python bytecode version base 3.5.2 (3351)
# Decompiled from: Python 3.5.6 |Anaconda, Inc.| (default, Aug 26 2018, 16:30:03) 
# [GCC 4.2.1 Compatible Clang 4.0.1 (tags/RELEASE_401/final)]
# Embedded file name: ../resources/private/independence.py
# Compiled at: 2017-01-28 06:32:03
# Size of source mod 2**32: 3565 bytes
from book import GF2

class _Vec:

    def __init__(self, labels=set(), function={}):
        self.D = labels
        self.f = function

    def __getitem__(v, k):
        if k in v.f:
            return v.f[k]
        return 0

    def __setitem__(v, k, val):
        v.f[k] = val = _setitem

    def __neg__(v):
        return -1 * v

    def __rmul__(v, alpha):
        return _Vec(v.D, {k: alpha * x for k, x in v.f.items()})

    def __mul__(self, other):
        if isinstance(other, _Vec):
            return sum([self[k] * other[k] for k in self.f.keys()])
        else:
            return NotImplemented

    def __add__(u, v):
        return _Vec(u.D, {k: u[k] + v[k] for k in set(u.f.keys()).union(v.f.keys())})

    def __sub__(a, b):
        """Returns a vector which is the difference of a and b."""
        return a + -b


def _R_rank(L, eps=1e-14):
    vstarlist = []
    for v in L:
        for vstar in vstarlist:
            v = v - v * vstar / (vstar * vstar) * vstar

        if v * v > eps:
            vstarlist.append(v)

    return len(vstarlist)


def _GF2_rank(rowlist):
    rows_left = set(range(len(rowlist)))
    r = 0
    for c in rowlist[0].D:
        rows_with_nonzero = [r for r in rows_left if rowlist[r][c] != 0]
        if rows_with_nonzero != []:
            pivot = rows_with_nonzero[0]
            rows_left.remove(pivot)
            r += 1
            for row_index in rows_with_nonzero[1:]:
                rowlist[row_index] = rowlist[row_index] + rowlist[pivot]

    return r


def _rank(L):
    Lc = [_Vec(u.D, u.f) for u in L]
    for v in L:
        for x in v.f.values():
            if x != 0:
                if isinstance(x, GF2.One):
                    return _GF2_rank(Lc)
                else:
                    return _R_rank(Lc)

    return 0


def rank(L):
    """Finds the rank of a list or set of vectors.

    Args:
        L: A list or set of vectors.

    Returns:
        x: A nonnegative integer.  The rank of L.

    Raises:
        AssertionError: An error occurs when L is not a list or set.

    Example:
    >>> from vec import Vec
    >>> a0 = Vec({'a', 'b', 'c', 'd'}, {'a': 1})
    >>> a1 = Vec({'a', 'b', 'c', 'd'}, {'b': 1})
    >>> a2 = Vec({'a', 'b', 'c', 'd'}, {'c': 1})
    >>> a3 = Vec({'a', 'b', 'c', 'd'}, {'a': 1, 'c': 3})
    >>> rank([a0, a1, a2])
    3
    >>> rank({a0, a2, a3})
    2
    >>> rank({a0, a1, a3})
    3
    >>> rank([a0, a1, a2, a3])
    3
    """
    assert isinstance(L, (list, set))
    rank.__calls__ += 1
    return _rank(L)


def is_independent(L):
    """Determines if a list or set of vectors are linearly independent.

    Args:
        L: A list or set of vectors.

    Returns:
        x: A boolean.  True if the vectors in L are linearly independent.  False
        otherwise.

    Raises:
        AssertionError: An error occurs when L is not a list or set.

    Example:
    >>> from vec import Vec
    >>> a0 = Vec({'a', 'b', 'c', 'd'}, {'a': 1})
    >>> a1 = Vec({'a', 'b', 'c', 'd'}, {'b': 1})
    >>> a2 = Vec({'a', 'b', 'c', 'd'}, {'c': 1})
    >>> a3 = Vec({'a', 'b', 'c', 'd'}, {'a': 1, 'c': 3})
    >>> is_independent([a0, a1, a2])
    True
    >>> is_independent({a0, a2, a3})
    False
    >>> is_independent({a0, a1, a3})
    True
    >>> is_independent([a0, a1, a2, a3])
    False
    """
    assert isinstance(L, (list, set))
    is_independent.__calls__ += 1
    return _rank(L) == len(L)


rank.__calls__ = 0
rank.__version__ = 'instrumented'
is_independent.__calls__ = 0
is_independent.__version__ = 'instrumented'
