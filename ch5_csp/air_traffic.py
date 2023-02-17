import copy


class Queue:
    "A container with a first-in-first-out (FIFO) queuing policy."

    def __init__(self):
        self.list = []

    def push(self, item):
        "Enqueue the 'item' into the queue"
        self.list.insert(0, item)

    def pop(self):
        """
        Dequeue the earliest enqueued item still in the queue. This
        operation removes the item from the queue.
        """
        return self.list.pop()

    def isEmpty(self):
        "Returns true if the queue is empty"
        return len(self.list) == 0


def select_variable(csp, assignment):
    """
    Minimal Remaining Values
    """
    free_vars = set(csp.variables) - set(assignment.keys())

    sorted_by_domain_length = sorted(
        [(len(csp.domains[var]), var) for var in free_vars if var not in assignment]
    )
    mrv = sorted_by_domain_length[0][1]
    return mrv


def ordered_domain_values(csp, var):
    # TODO: decide ordering
    return csp.domains[var]


def forward_check(csp, var, assignment):
    """Update domains according to rules and return consistency property."""
    xi = var
    if xi in csp.graph:
        for xj in csp.graph[xi].keys():  # check all constraints
            xj_values = csp.domains[xj]
            to_remove = []
            for val_j in xj_values:  # prune all invalid values
                for policy in csp.graph.get(xi, {}).get(xj, []):
                    policy_test_result = [
                        policy(val_i, val_j) for val_i in csp.domains[xi]
                    ]
                    if not any(policy_test_result):
                        if val_j in xj_values:
                            csp.domains[xj].remove(val_j)  # Update domains

            # Do it in reverse policy
            for val_j in xj_values:
                for policy in csp.graph.get(xj, {}).get(xi, []):
                    policy_test_result = [
                        policy(val_j, val_i) for val_i in csp.domains[xi]
                    ]
                    if not any(policy_test_result):
                        if val_j in xj_values:
                            csp.domains[xj].remove(val_j)  # Update domains

            if len(csp.domains[xj]) == 0:
                return False

    return True


def revise(csp, xi, xj):
    """
    Make xj arc-consistent in respect to xi.
    """
    revised = False

    xj_values = csp.domains[xj].copy()

    for val_j in xj_values:
        for policy in csp.graph.get(xi, {}).get(xj, []):
            policy_test_result = [policy(val_i, val_j) for val_i in csp.domains[xi]]
            if not any(policy_test_result):
                if val_j in csp.domains[xj]:
                    csp.domains[xj].remove(val_j)  # Update domains
                revised = True

        # Also do reverse policy check
        for policy in csp.graph.get(xj, {}).get(xi, []):
            policy_test_result = [policy(val_j, val_i) for val_i in csp.domains[xi]]
            if not any(policy_test_result):
                if val_j in csp.domains[xj]:
                    csp.domains[xj].remove(val_j)  # Update domains
                revised = True

    return revised


def arc3(csp, var, assignment):
    """update csp using arc consistency starting with var"""
    queue = []
    queue.insert(0, var)

    while queue:
        xi = queue.pop()
        # xi is the assignment changed
        # xj is assignment has constraint depends on xi
        if xi in csp.graph:
            for xj in csp.graph[xi].keys():
                if revise(csp, xi, xj):
                    if len(csp.domains[xj]) == 0:
                        return False
                    if xj in csp.graph:
                        for n in csp.graph[xj]:
                            if n != xi:
                                queue.insert(0, xj)

    return True


def backtracking(csp, assignment):
    """
    csp.variables = ['x1', 'x2']
    csp.domains = {
        'x1': [],
        'x2': [],
        # ...
    }
    # assume binary constrain
    csp.graph = {
        'x1': {
           'x2': lambda x1, x2: x1 > x2
        }
    }
    """
    # check all assignment
    if all(assignment.values()) and len(assignment.keys()) == len(csp.variables):
        return assignment

    var = select_variable(csp, assignment)

    for val in ordered_domain_values(csp, var):
        old_domains = copy.deepcopy(csp.domains)

        print(f"Assigning var {var} with value {val}.")
        assignment[var] = val
        csp.domains[var] = [val]
        is_valid = arc3(csp, var, assignment)  # update domains
        # is_valid = forward_check(csp, var, assignment)
        if is_valid:
            result = backtracking(csp, assignment)
            if result:
                return result

        csp.domains = old_domains
        del assignment[var]

    return None


class Struct:
    def __init__(self, **entries):
        self.__dict__.update(entries)


if __name__ == "__main__":
    csp = Struct(
        **{
            "variables": ["x1", "x2", "x3", "x4", "x5"],
            "domains": {
                "x1": [1, 2],
                "x2": [1],
                "x3": [1, 2, 3, 4],
                "x4": [3, 4],
                "x5": [1, 2, 3, 4],
            },
            "graph": {
                "x1": {
                    "x2": [lambda x1, x2: x1 != x2],
                    "x3": [lambda x1, x3: x1 != x3],
                },
                "x2": {
                    "x1": [lambda x2, x1: x1 != x2],
                    "x3": [lambda x2, x3: x2 != x3],
                },
                "x3": {
                    "x1": [lambda x3, x1: x1 != x3],
                    "x4": [lambda x3, x4: x4 < x3],
                },
                "x4": {"x3": [lambda x4, x3: x4 < x3], "x5": [lambda x4, x5: x4 != x5]},
                "x5": {"x4": [lambda x5, x4: x4 != x5]},
            },
        }
    )

    print(backtracking(csp, {}))
