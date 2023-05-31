"""
Codewar Kata - Hard Sudoku

https://www.codewars.com/kata/5588bd9f28dbb06f43000085/train/python

Is correct, but 50% slower than the required submission, completed all in 22778.37

Progress:
- Having a bug in the backtrack to make it working, see TODO:
- Next up think about reducing the problem in symettry.
"""

import functools
import itertools
import math
import copy
import time
from collections import defaultdict
from contextlib import contextmanager

# Optimization Strategy
# 1. [Done] Cache some rules, no clear improvement
# 2. [Done] Pre-compute the rules and neighbors, no clear improvemnt
# 3. [Done] Avoid deep copy the board in backtrack; instead tracking all the ops
#    1.342s => 0.593s
# 4. [Done] Only trying to find multiple solutiosn when needed
#    - Not reliable
# 5. [Done] Implement back jumping, only ~5% change and a bug
#
# Before(hard)
#
#     backtrack
#     460, total: 16.481s, per_call: 35828.1970us
#     backtrack out call
#     5, total: 0.487s, per_call: 97357.8930us
#
# After(hard)
#
#     backtrack
#     455, total: 15.200s, per_call: 33406.6003us
#     backtrack out call
#     5, total: 0.458s, per_call: 91536.3312us

# 6. [Todo] Implement symmetry reduction

# Precomputed with
# saved_neighbors = {}
# for i in range(9):
#     for j in range(9):
#         var = (i, j)
#         saved_neighbors[var] = find_rule_neighbors(var)
# print(saved_neighbors)
# neighbors = {...}

# Prebuilt with:
#     conflicts_map = {}
#     for i in range(9):
#         for j in range(9):
#             foo = (i, j)
#             conflicts_map[foo] = conflicts_map.get(foo, {})
#             for w in range(9):
#                 for z in range(9):
#                     bar = (w, z)
#                     conflicts_map[foo][bar] = conflicts_map[foo].get(bar, {})
#                     conflicts_map[foo][bar] = in_conflict(foo, bar)
#     print(conflicts_map)
# conflicts_map = {...}

global_profilers = {}
ENABLE_PROFILE = False
VERBOSE = False


def reset_profilers():
    for fn in global_profilers.keys():
        stats = global_profilers[fn]
        for k in stats:
            stats[k] = 0


def print_profilers():
    for fn in global_profilers.keys():
        print(fn)
        stats = global_profilers[fn]
        c = stats["counter"]
        t = stats["elapse"]
        pc = stats["elapse"] / stats["counter"] * 1000 * 1000
        print(f"{c}, total: {t:.3f}s, per_call: {pc:.4f}us")
    print()


def profile(func):
    if not ENABLE_PROFILE:
        return func

    s = global_profilers[func.__name__] = global_profilers.get(
        func.__name__, {"elapse": 0, "counter": 0}
    )

    def inner(*args, **kwargs):
        before = time.time()
        s["counter"] += 1
        result = func(*args, **kwargs)
        s["elapse"] += time.time() - before
        return result

    return inner


@contextmanager
def profiler(name, *args, **kwds):
    if not ENABLE_PROFILE:
        yield
        return

    # Code to acquire resource, e.g.:
    s = global_profilers[name] = global_profilers.get(name, {"elapse": 0, "counter": 0})
    before = time.time()
    s["counter"] += 1
    try:
        yield name
    finally:
        s["elapse"] += time.time() - before


# @profile
def mrv(game_state, assigned):
    """Return the variable with minimum remaining values."""
    found = False
    mrv = math.inf
    coord = (None, None)
    for i, _ in enumerate(game_state):
        for j, _ in enumerate(game_state[i]):
            if assigned[(i, j)]:
                continue
            if mrv > len(game_state[i][j]):
                mrv = len(game_state[i][j])
                coord = (i, j)
                found = True
    return coord, found


# arc, (tuple, tuple), meaning xi -> xj, xi depends on xj


# @profile
def get_domain(game_state, coord):
    return game_state[coord[0]][coord[1]]


def assign_domain(game_state, coord, val):
    game_state[coord[0]][coord[1]] = val


# @profile
def solved(game_state):
    for i in range(9):
        for j in range(9):
            if len(game_state[i][j]) != 1:
                # print(f"Unsolved at {i}, {j}: {game_state[i][j]}")
                return False
    return True


def solution(game_state):
    return [list(itertools.chain(*game_state[i])) for i in range(9)]


def print_grid(grid):
    for row in grid:
        print(row)
    print()


# @profile
def ac3(variables, init_arcs):
    queue = init_arcs
    conflicts = {}
    while queue:
        xi, xj = queue.pop()
        conflict = revise(variables, xi, xj)
        # Record and merge the result
        for k in conflict:
            if k in conflicts:
                conflicts[k] += conflict[k]
            else:
                conflicts[k] = conflict[k]
        if conflict:
            # domain is zone
            domain_x = get_domain(variables, xi)
            if len(domain_x) == 0:
                return False, conflicts
            for xk in find_rule_neighbors(xi):
                queue.insert(0, (xk, xi))
    return True, conflicts


# @profile
def revise(variables, xi, xj):
    conflict = defaultdict(
        lambda: []
    )  # {Tuple: value, Tuple}, var => removed_val, cause_var
    domain_xi = get_domain(variables, xi)
    domain_xj = get_domain(variables, xj)
    #     print(f"pruning domain {xi}: {domain_x}")
    for xi_v in list(domain_xi):
        if not any(check_values(xi, xj, xi_v, xj_v) for xj_v in domain_xj):
            domain_xi.remove(xi_v)
            #             print(f"pruned domain {xi} to {get_domain(variables, xi)}, {xi_v}")
            conflict[xi].append((xi_v, xj))
    return conflict


def to_block_index(x):
    # index in 3x3 square block
    return (x[0] // 3, x[1] // 3)


def _find_rule_neighbors(x):  # Tuple -> []Tuple
    return neighbors[x]


@functools.cache
def find_rule_neighbors(x):  # Tuple -> []Tuple
    """find all neighbors for a variable.

    Each variable has 3 constraint: row, column, square.
    """
    row = [(x[0], i) for i in range(9)]
    col = [(i, x[1]) for i in range(9)]

    sxi, sxj = to_block_index(x)
    block = [(sxi * 3 + i, sxj * 3 + j) for j in range(3) for i in range(3)]
    result = set(row + col + block)
    result.remove(x)
    return list(result)


def in_row(xi, xj):
    return xi[0] == xj[0]


def in_col(xi, xj):
    return xi[1] == xj[1]


def in_block(xi, xj):
    return to_block_index(xi) == to_block_index(xj)


def _in_conflict(xi, xj):
    return conflicts_map[xi][xj]


@functools.cache
def in_conflict(xi, xj):
    return in_row(xi, xj) or in_col(xi, xj) or in_block(xi, xj)


@profile
def check_values(xi, xj, xi_v, xj_v):
    """return whether xi, xj is valid."""
    if xi_v != xj_v:
        return True
    return not in_conflict(xi, xj)


def same_puzzle(a, b):
    for i, _ in enumerate(a):
        for j, _ in enumerate(b[i]):
            if a[i][j] != b[i][j]:
                return False
    return True


def diff_grid(a, b):
    return [
        ["X" if a[i][j] != b[i][j] else "-" for j, _ in enumerate(b[i])]
        for i, _ in enumerate(a)
    ]


def sudoku_solver(puzzle):
    # CSP problem
    # - Variables
    # - Domains, range(9)
    # - Constrains, 27 rules, implicit representation on neighbors and checks
    assert len(puzzle) == 9
    assert len(puzzle[0]) == 9

    givens = 0

    game_state = copy.deepcopy(puzzle)
    for i, _ in enumerate(game_state):
        for j, _ in enumerate(game_state[i]):
            assert puzzle[i][j] in range(10)
            if puzzle[i][j] == 0:
                game_state[i][j] = set(list(range(1, 10)))
            else:
                game_state[i][j] = [puzzle[i][j]]
                givens += 1

    assigned = defaultdict(lambda: False)
    for i, _ in enumerate(assigned):
        for j, _ in enumerate(assigned[i]):
            if puzzle[i][j] != 0:
                assigned[(i, j)] = True

    results = []

    has_unique_result = False
    if VERBOSE:
        print("givens = ", givens)
    # NOTE: Only find another solution if needed
    has_unique_result = givens > 17  # false assumption

    @profile
    def backtrack(game_state, assigned, prev_conflicts=[], assignments=[]):
        if solved(game_state):
            result = solution(game_state)
            results.append(result)
            if VERBOSE:
                print("Found one solution")
            return result, (None, None)

        var, found = mrv(game_state, assigned)

        #         if not found:
        #             print("NOT FOUND MRV", game_state, assigned)
        #             print(assigned)
        #             raise

        domain_var = get_domain(game_state, var)

        # Leaf return, running out of domain for variable
        if not domain_var:
            return None, (None, None)

        # has_forward, target
        jump_info = (None, None)
        for val in list(domain_var):
            if VERBOSE:
                print(f"assign {var} to {val}")
                if var == (2, 8):
                    print(f"(2, 8): {domain_var}")

            assigned[var] = True
            assignments.append(var)
            #             save for backtracking
            #             with profiler("copy game state"):
            #                 old_game_state = copy.deepcopy(game_state)

            assign_domain(game_state, var, {val})
            # arc, (tuple, tuple), meaning xi -> xj, xi depends on xj
            arcs = [(xi, var) for xi in find_rule_neighbors(var)]
            valid, conflicts = ac3(game_state, arcs)
            prev_conflicts.append(conflicts)
            if valid:
                result, jump_info = backtrack(
                    game_state, assigned, prev_conflicts, assignments
                )
                if VERBOSE:
                    print(f"[backjump in {var}] receive requests {jump_info}")
                if result and has_unique_result:
                    return result, (None, None)  # early success exit

            assignments.pop()
            prev_conflicts.pop()
            assign_domain(game_state, var, domain_var)
            for k in conflicts.keys():
                for removed_val, _ in conflicts[k]:
                    #                     print(f"adding back {k}: [{removed_val}]")
                    game_state[k[0]][k[1]].add(removed_val)
            #             game_state = old_game_state

            # Perform back jump by early returning from traversal until the target state
            if jump_info[0]:
                if jump_info[1] != var:
                    # TODO:
                    # if VERBOSE:
                    #     print(f"[backjump in {var}] early return")
                    # break
                    pass
                else:
                    if VERBOSE:
                        print(f"[backjump in {var}] found target")

        assigned[var] = False

        # Forward the back_jump_target
        if jump_info[1] and jump_info[1] != var:
            target = jump_info[1]
            assert assigned[target]

            if VERBOSE:
                print(f"[backjump in {var}] forward {jump_info[1]}")
            return None, (True, jump_info[1])
        else:
            if VERBOSE:
                print(f"[backjump in {var}] no forward for {jump_info}")

        # Searching all recorded conflicts in order
        # NOTE: Conflict is recorded at (val, tuple), tupel is the cause of conflicg
        # print("prev_conflicts:", prev_conflicts)

        # NOTE: Should be looking in reversed assignment order
        def fetch_conflict(assign):
            """returns the past conflicts for an assignment."""
            conflicts = functools.reduce(
                lambda a, b: a + b,
                [conf[assign] for conf in prev_conflicts if assign in conf],
                [],
            )
            result = {}
            for val, cause_var in conflicts:
                result[cause_var] = val
            return result

        conflicts_for_var = fetch_conflict(var)

        if VERBOSE:
            print(f"conflicts for {var}", conflicts_for_var)
            print(f"assignments: {assignments}")

        first_assigned_conflict = None
        for assign in reversed(assignments):
            if assign in conflicts_for_var:
                first_assigned_conflict = assign
                break

        if VERBOSE:
            print(f"first assign conflict for {var}: ", first_assigned_conflict)

        return None, (None, first_assigned_conflict)

    for i in range(9):
        for j in range(9):
            var = (i, j)
            arcs = [(xi, var) for xi in find_rule_neighbors(var)]
            valid, conflicts = ac3(game_state, arcs)
            if not valid:
                raise
    # print("performed arc consistency: ", game_state)

    with profiler("backtrack out call"):
        backtrack(game_state, assigned, [conflicts])

    if VERBOSE:
        print("Puzzle:")
        print_grid(puzzle)
        print("Solutions:")
        for r in results:
            print_grid(r)
            print_grid(diff_grid(results[0], r))

    if not results:
        raise Exception("No result found.")

    if len(results) != 1:
        #         for r in results:
        #             print_grid(r)
        raise Exception("Multiple result found.")

    return results[0]


if ENABLE_PROFILE:
    _sudoku_solver = sudoku_solver

    def sudoku_solver(puzzle):
        N = 2
        M = 5
        for _ in range(N):
            for _ in range(M):
                _sudoku_solver(puzzle)
            print_profilers()
            reset_profilers()

        return _sudoku_solver(puzzle)
