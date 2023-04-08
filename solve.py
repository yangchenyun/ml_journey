from book.vec import Vec


def solve(m, v):
    """A general linear system solver."""
    assert m.D[0] == v.D

    D0 = sorted(list(m.D[0]))
    D1 = sorted(list(m.D[1]))

    N0 = len(D0)
    N1 = len(D1)

    # Convert matrix to a list of lists
    m_list = [[0 for _ in range(N1)] for _ in range(N0)]
    for key, value in m.f.items():
        row, col = key
        m_list[D0.index(row)][D1.index(col)] = value

    # Convert vector to a list
    v_list = [0 for _ in range(N0)]
    for key, value in v.f.items():
        v_list[D0.index(key)] = value

    # Solve the linear system
    solution = _gaussian_elimination(m_list, v_list)

    # converts back to vector
    return Vec(m.D[1], {D1[i]: solution[i] for i in range(N1)})


# Gaussian elimination function
def _gaussian_elimination(matrix, vector):
    epsilon = 1e-20
    assert len(matrix) == len(vector)
    n = len(matrix)
    m = len(matrix[0])

    for i in range(min(m, n)):
        # Search for maximum in this column
        max_element = abs(matrix[i][i])
        max_row = i
        for k in range(i + 1, n):
            if abs(matrix[k][i]) > max_element:
                max_element = abs(matrix[k][i])
                max_row = k

        # Swap maximum row with current row
        matrix[i], matrix[max_row] = matrix[max_row], matrix[i]
        vector[i], vector[max_row] = vector[max_row], vector[i]

        # Make all rows below this one 0 in current column
        for k in range(i + 1, n):
            if matrix[i][i] == 0:
                continue
            c = -matrix[k][i] / (matrix[i][i] + epsilon)
            for j in range(i, m):
                if i == j:
                    matrix[k][j] = 0
                else:
                    matrix[k][j] += c * matrix[i][j]
            vector[k] += c * vector[i]

    # Solve equation matrix[n-1][n-1]*x[n-1] = vector[n-1]
    x = [0 for _ in range(n)]

    # NOTE: learning free degree variables as 0
    n = min(m, n)
    x[n - 1] = vector[n - 1] / (matrix[n - 1][n - 1] + epsilon)
    for i in range(n - 2, -1, -1):
        x[i] = vector[i]
        for k in range(i + 1, n):
            x[i] -= matrix[i][k] * x[k]
        x[i] /= matrix[i][i] + epsilon

    return x
