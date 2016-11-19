from puzzle import Puzzle
from copy import deepcopy

class MNPuzzle(Puzzle):
    """
    An nxm puzzle, like the 15-puzzle, which may be solved, unsolved,
    or even unsolvable.
    """

    def __init__(self, from_grid, to_grid):
        """
        MNPuzzle in state from_grid, working towards
        state to_grid

        @param MNPuzzle self: this MNPuzzle
        @param tuple[tuple[str]] from_grid: current configuration
        @param tuple[tuple[str]] to_grid: solution configuration
        @rtype: None
        """
        # represent grid symbols with letters or numerals
        # represent the empty space with a "*"
        assert len(from_grid) > 0
        assert all([len(r) == len(from_grid[0]) for r in from_grid])
        assert all([len(r) == len(to_grid[0]) for r in to_grid])
        self.n, self.m = len(from_grid), len(from_grid[0])
        self.from_grid, self.to_grid = from_grid, to_grid

    # TODO
    # implement __eq__ and __str__
    # __repr__ is up to you

    def __eq__(self, other):
        """
        Return whether MNPuzzle self is equivalent to other.

        @type self: MNPuzzle
        @type other: MNPuzzle | Any
        @rtype: bool

        >>> target_grid = (("1", "2", "3"), ("4", "5", "*"))
        >>> start_grid = (("*", "2", "3"), ("1", "4", "5"))
        >>> another_grid = (("*", "1", "3"), ("1", "4", "5"))
        >>> mn1 = MNPuzzle(start_grid, target_grid)
        >>> mn2 = MNPuzzle(start_grid, another_grid)
        >>> mn1 == mn2
        False
        """
        return (type(self) == type(other) and
                self.from_grid == other.from_grid and
                self.to_grid == other.to_grid)

    def __str__(self):
        """
        Return a human-readable string representation of MNPuzzle self.

        @param MNPuzzle self: this MNPuzzle
        @rtype: str

        >>> target_grid = (("1", "2", "3"), ("4", "5", "*"))
        >>> start_grid = (("*", "2", "3"), ("1", "4", "5"))
        >>> mn = MNPuzzle(start_grid, target_grid)
        >>> print(mn)
        * 2 3
        1 4 5
        =====
        1 2 3
        4 5 *
        """
        r = ""
        r += "\n".join([" ".join(row) for row in self.from_grid])
        r += '\n{}\n'.format("="*(2*len(self.from_grid[0])-1))
        r += "\n".join([" ".join(row) for row in self.to_grid])
        return r

    # TODO
    # override extensions
    # legal extensions are configurations that can be reached by swapping one
    # symbol to the left, right, above, or below "*" with "*"
    def extensions(self):
        """
        Return list of extensions of MNPuzzle self.

        @type self: MNPuzzle
        @rtype: list[MNPuzzle]

        >>> target_grid = (("1", "2", "3"), ("4", "5", "*"))
        >>> start_grid = (("*", "2", "3"), ("1", "4", "5"))
        >>> mn = MNPuzzle(start_grid, target_grid)
        >>> expected1 = (("2", "*", "3"), ("1", "4", "5"))
        >>> expected2 = (("1", "2", "3"), ("*", "4", "5"))
        >>> ext1 = MNPuzzle(expected1, target_grid)
        >>> ext2 = MNPuzzle(expected2, target_grid)
        >>> e = mn.extensions()
        >>> ext1 and ext2 in e
        True
        """
        n, m = self.n, self.m
        fr, to = self.from_grid, self.to_grid
        e = []

        position = [(row, col) for row in range(len(fr))
                    for col in range(len(fr[row])) if fr[row][col] == "*"]
        assert len(position) == 1

        x, y = position[0][0], position[0][1]

        if not x == 0:
            n_grid = []
            [n_grid.append(list(x)) for x in fr]
            n_grid[x][y] = n_grid[x-1][y]
            n_grid[x-1][y] = '*'
            n_tuple = tuple([tuple(row) for row in n_grid])
            e.append(MNPuzzle(n_tuple, to))
        if not x == n-1:
            n_grid = []
            [n_grid.append(list(x)) for x in fr]
            n_grid[x][y] = n_grid[x+1][y]
            n_grid[x+1][y] = '*'
            n_tuple = tuple([tuple(row) for row in n_grid])
            e.append(MNPuzzle(n_tuple, to))

        if not y == 0:
            n_grid = []
            [n_grid.append(list(x)) for x in fr]
            n_grid[x][y] = n_grid[x][y-1]
            n_grid[x][y-1] = '*'
            n_tuple = tuple([tuple(row) for row in n_grid])
            e.append(MNPuzzle(n_tuple, to))

        if not y == m-1:
            n_grid = []
            [n_grid.append(list(x)) for x in fr]
            n_grid[x][y] = n_grid[x][y+1]
            n_grid[x][y+1] = '*'
            n_tuple = tuple([tuple(row) for row in n_grid])
            e.append(MNPuzzle(n_tuple, to))

        return e

    # TODO
    # override is_solved
    # a configuration is solved when from_grid is the same as to_grid

    def is_solved(self):
        """
        Return whether MNPuzzle self is solved.

        @type self: MNPuzzle
        @rtype: bool

        >>> target_grid = (("1", "2", "3"), ("4", "5", "*"))
        >>> start_grid = (("*", "2", "3"), ("1", "4", "5"))
        >>> mn1 = MNPuzzle(start_grid, target_grid)
        >>> mn1.is_solved()
        False
        >>> mn2 = MNPuzzle(target_grid, target_grid)
        >>> mn2.is_solved()
        True
        """
        return self.from_grid == self.to_grid


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    target_grid = (("1", "2", "3"), ("4", "5", "*"))
    start_grid = (("*", "2", "3"), ("1", "4", "5"))
    from puzzle_tools import breadth_first_solve, depth_first_solve
    from time import time
    start = time()
    solution = breadth_first_solve(MNPuzzle(start_grid, target_grid))
    end = time()
    print("BFS solved: \n\n{} \n\nin {} seconds".format(
        solution, end - start))
    start = time()
    solution = depth_first_solve((MNPuzzle(start_grid, target_grid)))
    end = time()
    print("DFS solved: \n\n{} \n\nin {} seconds".format(
        solution, end - start))
