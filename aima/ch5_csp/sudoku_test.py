# pip install git+https://github.com/codewars/python-test-framework.git#egg=codewars_test
#
from sudoku import sudoku_solver
import codewars_test as test


@test.describe("Fixed tests")
def fixed():
    @test.it("Single solutions")
    def fff():
        problems = [
            [
                [0, 9, 6, 5, 0, 4, 0, 7, 1],
                [0, 2, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 4, 0, 9, 0, 6, 2, 3],
                [0, 0, 3, 0, 6, 0, 0, 8, 0],
                [0, 0, 8, 0, 5, 0, 4, 0, 0],
                [9, 0, 0, 4, 0, 0, 0, 0, 5],
                [7, 0, 0, 0, 0, 9, 0, 0, 0],
                [0, 0, 1, 0, 7, 5, 3, 4, 9],
                [2, 3, 0, 0, 4, 8, 1, 0, 7],
            ],
            [
                [6, 0, 0, 0, 0, 0, 0, 0, 2],
                [0, 0, 3, 6, 0, 1, 7, 0, 0],
                [0, 7, 0, 0, 4, 0, 0, 1, 0],
                [0, 5, 0, 9, 0, 4, 0, 3, 0],
                [0, 0, 9, 0, 0, 0, 1, 0, 0],
                [0, 6, 0, 7, 0, 8, 0, 2, 0],
                [0, 3, 0, 0, 6, 0, 0, 5, 0],
                [0, 0, 5, 3, 0, 9, 4, 0, 0],
                [7, 0, 0, 0, 0, 0, 0, 0, 3],
            ],
            [
                [7, 0, 0, 0, 0, 0, 0, 0, 3],
                [0, 0, 3, 1, 0, 5, 7, 0, 0],
                [0, 2, 0, 0, 9, 0, 0, 8, 0],
                [0, 8, 0, 3, 0, 1, 0, 6, 0],
                [0, 0, 1, 0, 0, 0, 8, 0, 0],
                [0, 7, 0, 9, 0, 8, 0, 4, 0],
                [0, 3, 0, 0, 4, 0, 0, 7, 0],
                [0, 0, 7, 5, 0, 2, 9, 0, 0],
                [9, 0, 0, 0, 0, 0, 0, 0, 5],
            ],
            [
                [0, 0, 6, 3, 0, 0, 0, 0, 2],
                [0, 3, 0, 0, 4, 0, 0, 6, 0],
                [7, 0, 0, 0, 0, 1, 9, 0, 0],
                [2, 0, 0, 0, 0, 8, 7, 0, 0],
                [0, 1, 0, 0, 5, 0, 0, 4, 0],
                [0, 0, 9, 1, 0, 0, 0, 0, 5],
                [0, 0, 7, 4, 0, 0, 0, 0, 8],
                [0, 9, 0, 0, 1, 0, 0, 2, 0],
                [3, 0, 0, 0, 0, 5, 6, 0, 0],
            ],
            [
                [0, 7, 0, 0, 3, 0, 0, 5, 0],
                [0, 0, 0, 9, 0, 2, 0, 0, 0],
                [1, 0, 6, 0, 0, 0, 4, 0, 2],
                [0, 0, 4, 0, 0, 0, 8, 0, 0],
                [7, 0, 0, 0, 4, 0, 0, 0, 5],
                [0, 0, 1, 0, 0, 0, 6, 0, 0],
                [8, 0, 5, 0, 0, 0, 7, 0, 3],
                [0, 0, 0, 8, 0, 9, 0, 0, 0],
                [0, 6, 0, 0, 7, 0, 0, 1, 0],
            ],
            [
                [9, 0, 0, 0, 4, 0, 0, 0, 6],
                [0, 0, 5, 2, 0, 0, 4, 0, 0],
                [0, 3, 0, 0, 1, 0, 0, 5, 0],
                [0, 0, 0, 0, 0, 0, 0, 8, 0],
                [3, 0, 4, 0, 9, 0, 7, 0, 5],
                [0, 7, 0, 0, 0, 0, 0, 0, 0],
                [0, 2, 0, 0, 3, 0, 0, 1, 0],
                [0, 0, 8, 0, 0, 6, 3, 0, 0],
                [6, 0, 0, 0, 7, 0, 0, 0, 9],
            ],
            [
                [2, 0, 8, 3, 4, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 7, 1, 0, 0],
                [4, 0, 0, 0, 0, 0, 0, 0, 7],
                [0, 0, 0, 0, 7, 5, 3, 6, 0],
                [0, 3, 0, 0, 0, 0, 2, 0, 0],
                [5, 0, 0, 1, 0, 0, 0, 0, 0],
                [1, 0, 0, 8, 0, 0, 0, 0, 0],
                [0, 5, 2, 0, 0, 0, 0, 3, 9],
                [0, 0, 0, 0, 0, 6, 5, 0, 0],
            ],
            [
                [0, 9, 1, 0, 0, 0, 7, 0, 0],
                [0, 0, 8, 0, 0, 6, 0, 0, 0],
                [0, 0, 6, 0, 4, 3, 0, 2, 0],
                [0, 4, 0, 0, 0, 0, 3, 7, 0],
                [0, 0, 3, 0, 7, 8, 0, 1, 0],
                [0, 0, 0, 0, 9, 0, 0, 8, 0],
                [7, 6, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 9, 0, 0, 0, 0, 4, 0],
                [0, 0, 0, 0, 0, 0, 5, 0, 1],
            ],
            [
                [0, 9, 0, 0, 7, 1, 0, 0, 4],
                [2, 0, 0, 0, 0, 0, 0, 7, 0],
                [0, 0, 3, 0, 0, 0, 2, 0, 0],
                [0, 0, 0, 9, 0, 0, 0, 3, 5],
                [0, 0, 0, 0, 1, 0, 0, 8, 0],
                [7, 0, 0, 0, 0, 8, 4, 0, 0],
                [0, 0, 9, 0, 0, 6, 0, 0, 0],
                [0, 1, 7, 8, 0, 0, 0, 0, 0],
                [6, 0, 0, 0, 2, 0, 7, 0, 0],
            ],
            [
                [8, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 3, 6, 0, 0, 0, 0, 0],
                [0, 7, 0, 0, 9, 0, 2, 0, 0],
                [0, 5, 0, 0, 0, 7, 0, 0, 0],
                [0, 0, 0, 0, 4, 5, 7, 0, 0],
                [0, 0, 0, 1, 0, 0, 0, 3, 0],
                [0, 0, 1, 0, 0, 0, 0, 6, 8],
                [0, 0, 8, 5, 0, 0, 0, 1, 0],
                [0, 9, 0, 0, 0, 0, 4, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 2, 7, 5, 0],
                [0, 1, 8, 0, 9, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [4, 9, 0, 0, 0, 0, 0, 0, 0],
                [0, 3, 0, 0, 0, 0, 0, 0, 8],
                [0, 0, 0, 7, 0, 0, 2, 0, 0],
                [0, 0, 0, 0, 3, 0, 0, 0, 9],
                [7, 0, 0, 0, 0, 0, 0, 0, 0],
                [5, 0, 0, 0, 0, 0, 0, 8, 0],
            ],
        ]

        solutions = [
            [
                [3, 9, 6, 5, 2, 4, 8, 7, 1],
                [8, 2, 7, 1, 3, 6, 5, 9, 4],
                [5, 1, 4, 8, 9, 7, 6, 2, 3],
                [4, 5, 3, 7, 6, 1, 9, 8, 2],
                [1, 7, 8, 9, 5, 2, 4, 3, 6],
                [9, 6, 2, 4, 8, 3, 7, 1, 5],
                [7, 4, 5, 3, 1, 9, 2, 6, 8],
                [6, 8, 1, 2, 7, 5, 3, 4, 9],
                [2, 3, 9, 6, 4, 8, 1, 5, 7],
            ],
            [
                [6, 1, 8, 5, 9, 7, 3, 4, 2],
                [4, 9, 3, 6, 2, 1, 7, 8, 5],
                [5, 7, 2, 8, 4, 3, 9, 1, 6],
                [2, 5, 7, 9, 1, 4, 6, 3, 8],
                [3, 8, 9, 2, 5, 6, 1, 7, 4],
                [1, 6, 4, 7, 3, 8, 5, 2, 9],
                [9, 3, 1, 4, 6, 2, 8, 5, 7],
                [8, 2, 5, 3, 7, 9, 4, 6, 1],
                [7, 4, 6, 1, 8, 5, 2, 9, 3],
            ],
            [
                [7, 5, 9, 2, 8, 4, 6, 1, 3],
                [8, 4, 3, 1, 6, 5, 7, 9, 2],
                [1, 2, 6, 7, 9, 3, 5, 8, 4],
                [5, 8, 4, 3, 7, 1, 2, 6, 9],
                [3, 9, 1, 4, 2, 6, 8, 5, 7],
                [6, 7, 2, 9, 5, 8, 3, 4, 1],
                [2, 3, 5, 8, 4, 9, 1, 7, 6],
                [4, 6, 7, 5, 1, 2, 9, 3, 8],
                [9, 1, 8, 6, 3, 7, 4, 2, 5],
            ],
            [
                [1, 5, 6, 3, 8, 9, 4, 7, 2],
                [9, 3, 2, 5, 4, 7, 8, 6, 1],
                [7, 8, 4, 2, 6, 1, 9, 5, 3],
                [2, 4, 5, 9, 3, 8, 7, 1, 6],
                [8, 1, 3, 7, 5, 6, 2, 4, 9],
                [6, 7, 9, 1, 2, 4, 3, 8, 5],
                [5, 6, 7, 4, 9, 2, 1, 3, 8],
                [4, 9, 8, 6, 1, 3, 5, 2, 7],
                [3, 2, 1, 8, 7, 5, 6, 9, 4],
            ],
            [
                [2, 7, 8, 1, 3, 4, 9, 5, 6],
                [4, 5, 3, 9, 6, 2, 1, 8, 7],
                [1, 9, 6, 5, 8, 7, 4, 3, 2],
                [6, 2, 4, 3, 9, 5, 8, 7, 1],
                [7, 8, 9, 6, 4, 1, 3, 2, 5],
                [5, 3, 1, 7, 2, 8, 6, 4, 9],
                [8, 4, 5, 2, 1, 6, 7, 9, 3],
                [3, 1, 7, 8, 5, 9, 2, 6, 4],
                [9, 6, 2, 4, 7, 3, 5, 1, 8],
            ],
            [
                [9, 8, 1, 7, 4, 5, 2, 3, 6],
                [7, 6, 5, 2, 8, 3, 4, 9, 1],
                [4, 3, 2, 6, 1, 9, 8, 5, 7],
                [2, 5, 9, 4, 6, 7, 1, 8, 3],
                [3, 1, 4, 8, 9, 2, 7, 6, 5],
                [8, 7, 6, 3, 5, 1, 9, 4, 2],
                [5, 2, 7, 9, 3, 4, 6, 1, 8],
                [1, 9, 8, 5, 2, 6, 3, 7, 4],
                [6, 4, 3, 1, 7, 8, 5, 2, 9],
            ],
            [
                [2, 7, 8, 3, 4, 1, 9, 5, 6],
                [6, 9, 5, 2, 8, 7, 1, 4, 3],
                [4, 1, 3, 5, 6, 9, 8, 2, 7],
                [9, 2, 1, 4, 7, 5, 3, 6, 8],
                [7, 3, 4, 6, 9, 8, 2, 1, 5],
                [5, 8, 6, 1, 3, 2, 7, 9, 4],
                [1, 6, 9, 8, 5, 3, 4, 7, 2],
                [8, 5, 2, 7, 1, 4, 6, 3, 9],
                [3, 4, 7, 9, 2, 6, 5, 8, 1],
            ],
            [
                [4, 9, 1, 2, 8, 5, 7, 6, 3],
                [2, 3, 8, 7, 1, 6, 9, 5, 4],
                [5, 7, 6, 9, 4, 3, 1, 2, 8],
                [8, 4, 5, 6, 2, 1, 3, 7, 9],
                [9, 2, 3, 5, 7, 8, 4, 1, 6],
                [6, 1, 7, 3, 9, 4, 2, 8, 5],
                [7, 6, 4, 1, 5, 9, 8, 3, 2],
                [1, 5, 9, 8, 3, 2, 6, 4, 7],
                [3, 8, 2, 4, 6, 7, 5, 9, 1],
            ],
            [
                [5, 9, 8, 2, 7, 1, 3, 6, 4],
                [2, 4, 6, 3, 8, 5, 9, 7, 1],
                [1, 7, 3, 4, 6, 9, 2, 5, 8],
                [8, 6, 2, 9, 4, 7, 1, 3, 5],
                [9, 3, 4, 5, 1, 2, 6, 8, 7],
                [7, 5, 1, 6, 3, 8, 4, 9, 2],
                [4, 2, 9, 7, 5, 6, 8, 1, 3],
                [3, 1, 7, 8, 9, 4, 5, 2, 6],
                [6, 8, 5, 1, 2, 3, 7, 4, 9],
            ],
            [
                [8, 1, 2, 7, 5, 3, 6, 4, 9],
                [9, 4, 3, 6, 8, 2, 1, 7, 5],
                [6, 7, 5, 4, 9, 1, 2, 8, 3],
                [1, 5, 4, 2, 3, 7, 8, 9, 6],
                [3, 6, 9, 8, 4, 5, 7, 2, 1],
                [2, 8, 7, 1, 6, 9, 5, 3, 4],
                [5, 2, 1, 9, 7, 4, 3, 6, 8],
                [4, 3, 8, 5, 2, 6, 9, 1, 7],
                [7, 9, 6, 3, 1, 8, 4, 5, 2],
            ],
            [
                [9, 4, 6, 1, 8, 2, 7, 5, 3],
                [3, 1, 8, 5, 9, 7, 4, 2, 6],
                [2, 7, 5, 6, 4, 3, 8, 9, 1],
                [4, 9, 2, 3, 1, 8, 5, 6, 7],
                [6, 3, 7, 2, 5, 4, 9, 1, 8],
                [8, 5, 1, 7, 6, 9, 2, 3, 4],
                [1, 2, 4, 8, 3, 5, 6, 7, 9],
                [7, 8, 3, 9, 2, 6, 1, 4, 5],
                [5, 6, 9, 4, 7, 1, 3, 8, 2],
            ],
        ]

        for p, s in zip(problems, solutions):
            test.assert_equals(sudoku_solver(p), s, "Incorrect solution!")

    @test.it("Invalid grids")
    def fff():
        problems = [
            [
                [1, 1, 3, 4, 5, 6, 7, 8, 9],
                [4, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 1, 3, 4, 5, 6, 7, 8, 9],
                [4, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [4, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 1, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [4, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8],
                [4, 0, 6, 7, 8, 9, 1, 2],
                [7, 8, 9, 1, 2, 3, 4, 5],
                [2, 3, 4, 5, 6, 7, 8, 9],
                [5, 6, 7, 8, 9, 1, 2, 3],
                [8, 9, 1, 2, 3, 4, 5, 6],
                [3, 4, 5, 6, 7, 8, 9, 1],
                [6, 7, 8, 9, 1, 2, 3, 4],
                [9, 1, 2, 3, 4, 5, 6, 7],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, "a"],
                [4, 0, 6, 7, 8, 9, 1, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
            ],
            [
                [1, 1, 1, 1, 1, 1, 1, 1, 1],
                [2, 2, 2, 2, 2, 2, 2, 2, 2],
                [3, 3, 3, 3, 3, 3, 3, 3, 3],
                [4, 4, 4, 4, 4, 4, 4, 4, 4],
                [5, 5, 5, 5, 5, 5, 5, 5, 5],
                [6, 6, 6, 6, 6, 6, 6, 6, 6],
                [7, 7, 7, 7, 7, 7, 7, 7, 7],
                [8, 8, 8, 8, 8, 8, 8, 8, 8],
                [9, 9, 9, 9, 9, 9, 9, 9, 9],
            ],
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [4, 5, 6, 7, 8, 9, 1, 2, 3],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
        ]
        for p in problems:
            test.expect_error(
                "Invalid grid should raise an error.", lambda: sudoku_solver(p)
            )

    @test.it("Unsolvable ones")
    def fff():
        problems = [
            [
                [0, 2, 3, 4, 5, 6, 7, 8, 9],
                [1, 5, 6, 7, 8, 9, 0, 2, 3],
                [7, 8, 9, 1, 2, 3, 4, 5, 6],
                [2, 3, 4, 5, 6, 7, 8, 9, 1],
                [5, 6, 7, 8, 9, 1, 2, 3, 4],
                [8, 9, 1, 2, 3, 4, 5, 6, 7],
                [3, 4, 5, 6, 7, 8, 9, 1, 2],
                [6, 7, 8, 9, 1, 2, 3, 4, 5],
                [9, 1, 2, 3, 4, 5, 6, 7, 8],
            ],
            [
                [0, 9, 6, 5, 0, 4, 0, 7, 1],
                [0, 2, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 4, 0, 9, 0, 6, 2, 3],
                [0, 0, 3, 0, 6, 0, 0, 8, 0],
                [0, 0, 8, 0, 5, 0, 4, 0, 0],
                [9, 0, 0, 4, 1, 0, 0, 0, 5],
                [7, 0, 0, 0, 0, 9, 0, 0, 0],
                [0, 0, 1, 0, 7, 5, 3, 4, 9],
                [2, 3, 0, 0, 4, 8, 1, 0, 7],
            ],
        ]

        for p in problems:
            test.expect_error(
                "Invalid grid should raise an error.", lambda: sudoku_solver(p)
            )


@test.it("Random tests")
def rndTests():
    multiples = [
        [
            [0, 8, 0, 0, 0, 9, 7, 4, 3],
            [0, 5, 0, 0, 0, 8, 0, 1, 0],
            [0, 1, 0, 0, 0, 0, 0, 0, 0],
            [8, 0, 0, 0, 0, 5, 0, 0, 0],
            [0, 0, 0, 8, 0, 4, 0, 0, 0],
            [0, 0, 0, 3, 0, 0, 0, 0, 6],
            [0, 0, 0, 0, 0, 0, 0, 7, 0],
            [0, 3, 0, 5, 0, 0, 0, 8, 0],
            [9, 7, 2, 4, 0, 0, 0, 5, 0],
        ],
        [
            [9, 0, 6, 0, 7, 0, 4, 0, 3],
            [0, 0, 0, 4, 0, 0, 2, 0, 0],
            [0, 7, 0, 0, 2, 3, 0, 1, 0],
            [5, 0, 0, 0, 0, 0, 1, 0, 0],
            [0, 4, 0, 2, 0, 8, 0, 6, 0],
            [0, 0, 3, 0, 0, 0, 0, 0, 5],
            [0, 3, 0, 7, 0, 0, 0, 5, 0],
            [0, 0, 7, 0, 0, 5, 0, 0, 0],
            [4, 0, 5, 0, 1, 0, 7, 0, 8],
        ],
    ]

    unsolvables = [
        [
            [6, 0, 0, 0, 0, 0, 0, 0, 2],
            [0, 0, 3, 6, 0, 1, 7, 0, 0],
            [0, 7, 0, 0, 4, 0, 0, 1, 0],
            [0, 5, 0, 9, 0, 4, 0, 3, 0],
            [0, 0, 9, 0, 0, 0, 1, 0, 0],
            [0, 6, 0, 7, 0, 8, 0, 2, 0],
            [0, 3, 0, 0, 6, 0, 0, 5, 0],
            [0, 0, 5, 3, 0, 9, 4, 0, 0],
            [7, 2, 0, 0, 0, 0, 0, 0, 3],
        ],
        [
            [7, 0, 0, 0, 0, 0, 0, 0, 3],
            [0, 0, 3, 1, 0, 5, 7, 0, 0],
            [0, 2, 0, 0, 9, 0, 0, 8, 0],
            [0, 8, 0, 3, 0, 1, 0, 6, 0],
            [0, 0, 1, 0, 0, 0, 8, 0, 0],
            [0, 7, 0, 9, 0, 8, 0, 4, 0],
            [0, 3, 0, 0, 4, 0, 0, 7, 0],
            [0, 0, 7, 5, 0, 2, 9, 0, 4],
            [9, 0, 0, 0, 0, 0, 0, 0, 5],
        ],
    ]

    solvables = [
        [
            [0, 0, 0, 7, 0, 0, 0, 0, 0],
            [0, 0, 8, 0, 4, 9, 0, 1, 0],
            [0, 0, 5, 8, 0, 0, 0, 6, 0],
            [8, 0, 9, 0, 0, 0, 6, 0, 0],
            [0, 0, 0, 9, 0, 3, 0, 0, 0],
            [0, 0, 2, 0, 0, 0, 4, 0, 9],
            [0, 6, 0, 0, 0, 2, 7, 0, 0],
            [0, 2, 0, 5, 1, 0, 9, 0, 0],
            [0, 0, 0, 0, 0, 4, 0, 0, 0],
        ],
        [
            [0, 1, 0, 5, 0, 0, 2, 0, 0],
            [0, 0, 8, 0, 1, 0, 4, 0, 0],
            [0, 5, 0, 3, 0, 4, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 0],
            [0, 6, 4, 0, 8, 0, 3, 7, 0],
            [0, 0, 0, 0, 0, 0, 0, 4, 0],
            [0, 0, 0, 8, 0, 6, 0, 5, 0],
            [0, 0, 1, 0, 4, 0, 6, 0, 0],
            [0, 0, 6, 0, 0, 7, 0, 9, 0],
        ],
        [
            [0, 0, 0, 1, 0, 0, 2, 0, 0],
            [0, 0, 7, 0, 8, 6, 0, 0, 3],
            [0, 0, 6, 0, 0, 5, 0, 0, 0],
            [0, 3, 0, 0, 2, 0, 5, 0, 7],
            [0, 5, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 9, 0, 4, 0, 0, 8, 0],
            [0, 0, 0, 4, 0, 0, 7, 0, 0],
            [9, 0, 0, 7, 1, 0, 4, 0, 0],
            [0, 0, 8, 0, 0, 9, 0, 0, 0],
        ],
        [
            [4, 7, 0, 3, 0, 2, 0, 6, 0],
            [0, 0, 9, 0, 0, 0, 2, 0, 0],
            [0, 8, 0, 0, 0, 0, 7, 0, 0],
            [0, 5, 0, 0, 1, 9, 0, 0, 0],
            [0, 0, 0, 6, 0, 5, 0, 0, 0],
            [0, 0, 0, 2, 8, 0, 0, 5, 0],
            [0, 0, 3, 0, 0, 0, 0, 9, 0],
            [0, 0, 2, 0, 0, 0, 8, 0, 0],
            [0, 4, 0, 8, 0, 6, 0, 7, 2],
        ],
        [
            [8, 0, 3, 0, 7, 0, 0, 0, 0],
            [2, 0, 0, 0, 0, 0, 1, 8, 0],
            [0, 0, 1, 8, 0, 0, 0, 0, 0],
            [0, 0, 0, 5, 0, 0, 2, 0, 0],
            [0, 1, 0, 9, 8, 4, 0, 6, 0],
            [0, 0, 9, 0, 0, 2, 0, 0, 0],
            [0, 0, 0, 0, 0, 6, 4, 0, 0],
            [0, 4, 5, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 9, 0, 5, 0, 3],
        ],
        [
            [0, 0, 0, 0, 0, 2, 7, 5, 0],  # the hardest one
            [0, 1, 8, 0, 9, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            [4, 9, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 8],
            [0, 0, 0, 7, 0, 0, 2, 0, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 9],
            [7, 0, 0, 0, 0, 0, 0, 0, 0],
            [5, 0, 0, 0, 0, 0, 0, 8, 0],
        ],
    ]

    def randomizeGrid(p):
        switch = rand(16)

        if switch & 1:  # swap vertically the whole grid
            for x in range(4):
                p[x], p[8 - x] = p[8 - x], p[x]

        if switch & 2:  # swap horizontally the whole grid
            for r in p:
                for y in range(4):
                    r[y], r[8 - y] = r[8 - y], r[y]

        if switch & 4:  # mix groups of 3 consecutive rows
            i, j, k = (3 * z for z in sample(range(3), 3))
            for x in range(3):
                p[x], p[3 + x], p[6 + x] = p[i + x], p[j + x], p[k + x]

        if switch & 8:  # transpose
            p[:] = map(list, zip(*p))

    BASE = set(range(1, 10))

    def cccheckSquare(board, n):
        x, y = (n // 3) * 3, (n % 3) * 3
        return BASE == {board[x + i][y + j] for i in range(3) for j in range(3)}

    def validSolution(board, time=1):
        return (
            time == 1
            and validSolution(list(zip(*board)), 2)
            and all(cccheckSquare(board, n) for n in range(9))
            or time == 2
        ) and all(set(r) == BASE for r in board)

    def display(grid):
        print("\n".join(map(str, grid)))
        print("--------------------------")

    def isConsistent(actual, grid):
        return all(
            v == actual[x][y] for x, r in enumerate(grid) for y, v in enumerate(r) if v
        )

    def checker(p):
        actual = sudoku_solver([r[:] for r in p])

        print("Your result:")
        display(actual)

        test.expect(
            {v for r in actual for v in r} == BASE,
            "Your solution should only contain integer between 1 and 9",
        )
        test.assert_equals(len(actual), 9, "Your solution should have 9 rows")
        test.assert_equals(
            set(map(len, actual)), {9}, "All rows should have 9 elements"
        )
        test.expect(validSolution(actual), "Invalid solution!")
        test.expect(
            isConsistent(actual, p),
            "You cannot modify the values known at the beginning",
        )

    doneBig = 0
    nFailed = 0

    from random import randrange as rand, shuffle, sample

    for rr in range(40):
        isFail = rand(5) == 0 and nFailed < 6
        nFailed += isFail
        pool = solvables if not isFail else multiples if rand(2) == 0 else unsolvables

        iP = rand(len(pool) - (doneBig > 2 and not isFail))
        if rr >= 37 and doneBig < 3:
            isFail, iP, pool = 0, len(solvables) - 1, solvables
        doneBig += not isFail and iP == len(solvables) - 1

        p = pool[iP]
        randomizeGrid(p)  # (by mutation only)
        print("Input:")
        display(p)

        if isFail:
            test.expect_error(
                "Invalid grid should raise an error.", lambda: sudoku_solver(p)
            )
        else:
            checker(p)