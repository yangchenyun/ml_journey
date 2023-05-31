# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     argv:
#     - /Users/steveyang/miniconda3/bin/python
#     - -m
#     - ipykernel_launcher
#     - -f
#     - '{connection_file}'
#     display_name: Python 3 (ipykernel)
#     env: null
#     interrupt_mode: signal
#     language: python
#     metadata:
#       debugger: true
#     name: python3
# ---

# +
# Reviews and Exercises
# 1. What's coordinate representation of a vector?
# A vector could be written as a linear combination of generators of a vector space.
# The generators [a1, a2, a3, ...] are called coordinate system.
# The coefficients are called coordinate representation in terms of the coordinate system.
#
# (How to ensure existence and uniqueness of representation?)
#
# 2. How can you express conversion between a vector and its coordinate representation using matrices?
# Let A = [a1, a2, a3, ...] be a coordinate system.
# By definition v = A*u, u = [u1, u2, u3, ...] is the coordinate representation of v in terms of A.
# To find the coordinate representation is equivalent of solving v = A*x.
#
# 3. What's linear dependence?
# It is a proper to describe a set of vectors; a set of vectors are linear dependent if
# 
# One vector could be represented as a linear combination of the others.
# It is equiavelent to
# The zero vector could be expressed as non-trivial linear combination of the vectors.
#
# 4. How would you prove a set of vectors are linearly independent?
# Approach 1: using definition, prove that the zero vector is not in the span of the vectors. (except trivial case)
#
# Computational Problem 5.5.5: Testing linear dependence, gaussian_elimination algorithm
#
# 5. Grow algorithm
# Start with empty set T, each step add the vector which are 1. minimal weight, 2. linearly independent from other vectors of T
# Stop when this process cannot be continued.
#
# 6. Shrink algorithm
# Start with B, each step remove the vector which are 1. maximal weight, 2. linearly dependent from other vectors of B (span B remains the same)
# Stop when this process cannot be continued.
#
# 7. How do the concepts of linear dependence and spanning apply to subsets of edges of graphs?
# Define path as sequence of edges [{x1, x2}, {x2, x3}, {x3, x4}, ...]
# A set of edges S is spanning if there is for any {x, y} in G, there is a path in S from x to y.
# 
# linear dependence: a set of edges are linear dependent means there is a cycle between in a graph formed by those edges. 
#
# Key properties:
# 1. Subset of linearly independent edges are independent.
#    Any subset of MSF edges is linear independent (on cycle).
# 2. Span Lemma. Given V, vi in V is in the span of V - {vi} if and only if V is linerally dependent
#    An edge is in the span of other MSF edges if and only if it is in a cycle.
#
# 8. What is a basis?
# Linear independent set of generators of V. (Span B = V, B is linear independent)
# 
# Lemma 5.6.9: The standard generators for FD form a basis.
# Lemma 5.6.11 (Subset-Basis Lemma): Any finite set T of vectors contains a subset B that is a basis for Span T.
#
# 9. What is unique representation?
# Unique-Representation Lemma, if the coordinates system is forms of basis for V,
# then the coordinate representation is unique.
# 
# 10. What is change of basis?
# Given two coordinates systems of basis for V, the same vector could be represented: v = A*u, v = B*m
# Change of basis is defining function to convert u to m
# 
# Let f(x) = A*x, g(x) = B*x; because A and B are basis, using
# Unique-Representation Lemma, f and g are one-to-one and onto, thus invertible.
# 
# m -> u, u = A.inv()*B*m
# u -> m, m = B.inv()*A*u
#
# 11. What is the Exchange Lemma?
# Suppose z is a vector in Span S and not in A such that A ∪ {z} is linearly independent.
# There _always exists_ a w in S - A could be exchanged with z, span S = Span ({z} ∪ (S - {w}))
#
# Ways to construct new basis.

# +
import importlib

from book.GF2 import one, zero
import book.GF2
from book.vecutil import *
from book.matutil import *
import solve
# -

# 5.14.9
# (a)
list2vec([one, one, one, one]) + list2vec([one, zero, one, zero]) + list2vec([zero, one, zero, one])
# (b)
list2vec([0, 0, one, 0]) + list2vec([one, one, 0, one]) + list2vec([one, one, one, one])


# +
# 5.14.10
# looking for cycles in the graph

# +
# Problem 5.14.13
def rep2vec(u, veclist):
    """Returns the vector given coordinate represention u and coordinates in veclist."""
    return coldict2mat(veclist) * u

a0 = Vec({'a','b','c','d'}, {'a':1})
a1 = Vec({'a','b','c','d'}, {'b':1})
a2 = Vec({'a','b','c','d'}, {'c':1})

[
    rep2vec(Vec({0,1,2}, {0:2, 1:4, 2:6}), [a0,a1,a2]),
    rep2vec(list2vec([5, 3, -2]),
            [list2vec(l) for l in [[1,0,2,0],[1,2,5,1],[1,5,-1,3]]]),
    rep2vec(list2vec([one,one,0]),
            [list2vec(l) for l in [[one,0,one],[one,one,0],[0,0,one]]])
]


# +
# problem 5.14.14
def vec2rep(veclist, v):
    """Returns the coordinate representation of v in terms of the vectors in veclist."""
    return solve.solve(coldict2mat(veclist), v)

[
    vec2rep([a0,a1,a2], Vec({'a','b','c','d'}, {'a':3, 'c':-2})),
    vec2rep([list2vec(l) for l in [[1,0,2,0],[1,2,5,1],[1,5,-1,3]]], list2vec([6, -4, 27, -3])),
    vec2rep([list2vec(l) for l in [[one,0,one],[one,one,0],[0,0,one]]], list2vec([0, one, one])),
]


# +
# 5.14.15
importlib.reload(solve)
importlib.reload(book.GF2)
from book.GF2 import one, zero

def sqr(x): return x*x

def is_gf2_one(v):
    return any([type(v) is type(one) for v in v.f.values()])

def is_superfluous(L, i):
    """Returns True if the i-th vector in L is superfluous."""
    epsilon = 1e-20
    if len(L) == 1:
        return False

    assert i < len(L)

    Li = L[i]
    L = L[:i] + L[i+1:]
    solution = solve.solve(coldict2mat(L), Li)
    # NOTE: no guarantee the solution
    if solution == zero_vec(solution.D):
        return False
    residual = coldict2mat(L) * solution - Li

    # Dot-product of GF2 is always zero, so here only check one
    if is_gf2_one(residual):
        return False

    return sqr(residual) < epsilon

a0 = Vec({'a','b','c','d'}, {'a':1})
a1 = Vec({'a','b','c','d'}, {'b':1})
a2 = Vec({'a','b','c','d'}, {'c':1})
a3 = Vec({'a','b','c','d'}, {'a':1,'c':3})

[
    is_superfluous([a0,a1,a2,a3], 3),
    is_superfluous([a0,a1,a2,a3], 0),
    is_superfluous([a0,a1,a2,a3], 1),
    is_superfluous([list2vec(l) for l in [[1, 2, 3]]], 0),
    is_superfluous([list2vec(l) for l in [[2, 5, 5, 6], [2, 0, 1, 3], [0, 5, 4, 3]]], 2),
    is_superfluous([list2vec(l) for l in [[one, one, 0, 0], [one, one, one, one], [0, 0, 0, one]]], 2),
]


# +
# problem 5.14.16
# NOTE: brute force
def is_independent(L):
    return all([not is_superfluous(L, i) for i in range(len(L))])

# TODO: Do it in one gaussilimination?


[
    is_independent([a0, a1, a2]),
    is_independent([a0, a2, a3]),
    is_independent([a0, a1, a3]),
    is_independent([a0, a1, a2, a3]),
    is_independent([list2vec(l) for l in [[2, 4, 0], [8, 16, 4], [0, 0, 7]]]),
    is_independent([list2vec(l) for l in [[1, 3, 0, 0], [2, 1, 1, 0], [1, 1, 4, -1]]]),
    is_independent([list2vec(l) for l in [[one, 0, one, 0], [0, one, 0, 0], [one, one, one, one], [one, 0, 0, one]]]),
]

# is_superfluous([list2vec(l) for l in 
#                 [
#                     [one,one,0,0],
#                     [one,one,one,one],
#                     [0,0,one,one],
#                     # [0,0,0,one],
#                     # [0,0,one,0]
#                 ]], 0) 

# +
# Problem 5.14.17
import functools

def subset_basis(T):
    """Returns a subset basis for spanning T."""
    return functools.reduce(
        lambda acc, v: acc + [v] if is_independent(acc + [v]) else acc, T, [])

[
    subset_basis([a0,a1,a2,a3]),
    subset_basis([a0,a3,a1,a2]),
    subset_basis([list2vec(l) for l in [
        [1,1,2,1],[2,1,1,1],[1,2,2,1],[2,2,1,2],[2,2,2,2]
    ]]),
    subset_basis([list2vec(l) for l in [
        [one,one,0,0],[one,one,one,one],[0,0,one,one],[0,0,0,one],[0,0,one,0]
    ]]),
]


# +
# Problem 5.14.18
def superset_basis(T, L):
    """Returns a superset basis containing T which equals to span L."""
    return functools.reduce(
        lambda acc, v: acc + [v] if is_independent(acc + [v]) else acc, L, T)

[
    superset_basis([a0, a3], [a0, a1, a2]),
    superset_basis(
        [list2vec(l) for l in [[0, 5, 3], [0, 2, 2], [1, 5, 7]]],
        [list2vec(l) for l in [[1, 1, 1], [0, 1, 1], [0, 0, 1]]],
    ),
    superset_basis(
        [list2vec(l) for l in [[0, 5, 3], [0, 2, 2]]],
        [list2vec(l) for l in [[1, 1, 1], [0, 1, 1], [0, 0, 1]]],
    ),
    superset_basis([list2vec(l) for l in [
        [0,one,one,0],[one,0,0,one]
    ]], [list2vec(l) for l in [
        [one,one,one,one],[one,0,0,0], [0,0,0,one]
    ]]),
]


# +
# problem 5.14.19
def exchange(S, A, z):
    assert is_independent(A + [z])
    B = list(set(S) - set(A))
    coeff = vec2rep(B+A, z)

    # turns into list
    coeff = list(coeff.f.values())

    # find non-zero coefficient in B
    B_coeff = coeff[:len(B)]

    # find the first non-zero value in B_coeff
    i = next(i for i, v in enumerate(B_coeff) if v != 0)
    bi = B_coeff[i]

    coeff_rest = coeff[:i] + coeff[i+1:]
    B_rest = B[:i] + B[i+1:]

    # lined up the new vector
    u = list2vec([1/bi] + [-v / bi for v in coeff_rest])
    veclist = [z]+B_rest+A

    return rep2vec(u, veclist) 

def accept_list(fn):
    def wrapper(*args):
        def convert(arg):
            if not isinstance(arg, list): return arg
            if len(arg) == 0: return arg

            assert len(arg) > 0 and isinstance(arg, list)

            # nested case
            if isinstance(arg[0], list):
                converted = [convert(v) for v in arg]
                return converted

            return list2vec(arg) if not isinstance(arg[0], Vec) else arg

        args = [convert(v) for v in args]
        return fn(*args)
    return wrapper
    
exchangeL = accept_list(exchange)

S=[list2vec(v) for v in [[0,0,5,3] , [2,0,1,3],[0,0,1,0],[1,2,3,4]]]
A=[list2vec(v) for v in [[0,0,5,3],[2,0,1,3]]]
z=list2vec([0,2,1,1])

[
    exchange(S, A, z),
    exchange(S, A, z) == exchangeL(
        [[0,0,5,3] , [2,0,1,3],[0,0,1,0],[1,2,3,4]],
        [[0,0,5,3],[2,0,1,3]],
        [0,2,1,1]
    ),
    exchangeL(
        [[0,0,5,3],[2,0,1,3],[0,0,1,0],[1,2,3,4]],[[0,0,5,3],[2,0,1,3]],[0,2,1,1]
    ),
    exchangeL(
        [[0,one,one,one],[one,0,one,one],[one,one,0,one],[one,one,one,0]],
        [[0,one,one,one],[one,one,0,one]],
        [one,one,one,one]
    )
]

