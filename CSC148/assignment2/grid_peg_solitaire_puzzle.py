from puzzle import Puzzle
from copy import deepcopy

class GridPegSolitairePuzzle(Puzzle):
    """
    Snapshot of peg solitaire on a rectangular grid. May be solved,
    unsolved, or even unsolvable.
    """

    def __init__(self, marker, marker_set):
        """
        Create a new GridPegSolitairePuzzle self with
        marker indicating pegs, spaces, and unused
        and marker_set indicating allowed markers.

        @type marker: list[list[str]]
        @type marker_set: set[str]
                          "#" for unused, "*" for peg, "." for empty
        """
        assert isinstance(marker, list)
        assert len(marker) > 0
        assert all([len(x) == len(marker[0]) for x in marker[1:]])
        assert all([all(x in marker_set for x in row) for row in marker])
        assert all([x == "*" or x == "." or x == "#" for x in marker_set])
        self._marker, self._marker_set = marker, marker_set

    # TODO
    # implement __eq__, __str__ methods
    # __repr__ is up to you
    def __eq__(self, other):
        """
        Return whether GridPegSolitairePuzzle self is equivalent to other.

        @type self: GridPegSolitairePuzzle
        @type other: GridPegSolitairePuzzle | Any
        @rtype: bool

        >>> grid1 = [["*", "*", "*", "*", "*"]]
        >>> grid1.append(["*", "*", "*", "*", "*"])
        >>> grid1.append(["*", "*", "*", "*", "*"])
        >>> grid1.append(["*", "*", ".", "*", "*"])
        >>> grid1.append(["*", "*", "*", "*", "*"])
        >>> gpsp1 = GridPegSolitairePuzzle(grid1, {"*", ".", "#"})
        >>> grid2 = [["*", "*", "*", "*", "*"]]
        >>> grid2.append(["*", "*", "*", "*", "*"])
        >>> grid2.append(["*", "*", "*", "*", "*"])
        >>> grid2.append(["*", "*", "*", ".", "*"])
        >>> grid2.append(["*", "*", "*", "*", "*"])
        >>> gpsp2 = GridPegSolitairePuzzle(grid2, {"*", ".", "#"})
        >>> grid3 = [["*", "*", "*", "*", "*"]]
        >>> grid3.append(["*", "*", "*", "*", "*"])
        >>> grid3.append(["*", "*", "*", "*", "*"])
        >>> grid3.append(["*", "*", ".", "*", "*"])
        >>> grid3.append(["*", "*", "*", "*", "*"])
        >>> gpsp3 = GridPegSolitairePuzzle(grid3, {"*", ".", "#"})
        >>> gpsp1.__eq__(gpsp2)
        False
        >>> gpsp1.__eq__(gpsp3)
        True
        """
        return (type(self) == type(other) and
                self._marker == other._marker and
                self._marker_set == other._marker_set)

    def __str__(self):
        '''
        Return a human-readable string representation
        of GridPegSolitairePuzzle self.

        @param GridPegSolitairePuzzle self: this GridPegSolitairePuzzle
        @rtype: str

        >>> grid = [["*", "*", "*", "*", "*"]]
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> grid.append(["*", "*", ".", "*", "*"])
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> gpsp = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> print(gpsp)
        * * * * *
        * * * * *
        * * * * *
        * * . * *
        * * * * *
        '''
        marker = self._marker
        rows = [" ".join(l) for l in marker]
        return "\n".join(rows)

    # TODO
    # override extensions
    # legal extensions consist of all configurations that can be reached by
    # making a single jump from this configuration

    def extensions(self):
        """
        Return list of extensions of GridPegSolitairePuzzle self.

        @type self: GridPegSolitairePuzzle
        @rtype: list[GridPegSolitairePuzzle]

        >>> grid = [["*", "*", "*", "*", "*"]]
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> grid.append(["*", "*", ".", "*", "*"])
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> grid.append(["*", "*", "*", "*", "*"])
        >>> gpsp = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> e = gpsp.extensions()
        >>> grid[0][2] = "."
        >>> grid[1][2] = "."
        >>> grid[2][2] = "*"
        >>> next = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> next in e
        True
        """

        marker, marker_set = self._marker, self._marker_set
        rpuzzle = []

        for i,row in enumerate(marker):
            for j in range(len(row) - 2):
                if (row[j], row[j+1], row[j+2]) == ("*", "*", "."):
                    n_marker = []
                    [n_marker.append(e[:]) for e in marker]
                    n_marker[i] = row[:j] + [".", ".", "*"] + row[j+3:]
                    rpuzzle.append(
                        GridPegSolitairePuzzle(n_marker, {"*", ".", "#"}))
                if (row[j], row[j+1], row[j+2]) == (".", "*", "*"):
                    n_marker = []
                    [n_marker.append(e[:]) for e in marker]
                    n_marker[i] = row[:j] + ["*", ".", "."] + row[j+3:]
                    rpuzzle.append(
                        GridPegSolitairePuzzle(n_marker, {"*", ".", "#"}))
        for i in range(len(marker)-2):
            for j in range(len(marker[i])):
                if (marker[i][j] == "*" and marker[i+1][j] == "*"
                    and marker[i+2][j] == "."):
                    n_marker = []
                    [n_marker.append(e[:]) for e in marker]
                    n_marker[i][j] = "."
                    n_marker[i+1][j] = "."
                    n_marker[i+2][j] = "*"
                    rpuzzle.append(
                        GridPegSolitairePuzzle(n_marker, {"*", ".", "#"}))
                if (marker[i][j] == "." and marker[i+1][j] == "*"
                    and marker[i+2][j] == "*"):
                    n_marker = []
                    [n_marker.append(e[:]) for e in marker]
                    n_marker[i][j] = "*"
                    n_marker[i+1][j] = "."
                    n_marker[i+2][j] = "."
                    rpuzzle.append(
                        GridPegSolitairePuzzle(n_marker, {"*", ".", "#"}))
        return rpuzzle

    # TODO
    # override is_solved
    # A configuration is solved when there is exactly one "*" left

    def is_solved(self):
        """
        Return whether GridPegSolitairePuzzle self is solved.

        @type self: GridPegSolitairePuzzle
        @rtype: bool

        >>> grid = [[".", ".", ".", ".", "."]]
        >>> grid.append([".", ".", ".", ".", "."])
        >>> grid.append([".", ".", ".", ".", "."])
        >>> grid.append([".", ".", "*", "*", "."])
        >>> grid.append([".", ".", ".", ".", "."])
        >>> gpsp = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> r = gpsp.is_solved()
        >>> r
        False
        >>> grid[3][3] = "."
        >>> gpsp2 = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> t = gpsp2.is_solved()
        >>> t
        True
        """
        import collections
        l = []
        [l.extend(r) for r in self._marker]
        c = collections.Counter(l)
        return c["*"] == 1

    def fail_fast(self):
        """
        Return True if there is a lone peg on the board. Neighboring positions
        are all '.'

        @type self: GridPegSolitairePuzzle
        @rtype: bool

        >>> grid = [[".", ".", ".", ".", "."]]
        >>> grid.append([".", "*", ".", ".", "."])
        >>> grid.append([".", ".", ".", ".", "."])
        >>> grid.append([".", ".", "*", ".", "."])
        >>> grid.append([".", ".", ".", ".", "."])
        >>> gpsp = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
        >>> r = gpsp.fail_fast()
        >>> r
        True
        >>> grid2 = [[".", ".", ".", ".", "."]]
        >>> grid2.append([".", "*", "*", ".", "."])
        >>> grid2.append([".", ".", ".", ".", "."])
        >>> grid2.append([".", ".", ".", ".", "."])
        >>> grid2.append([".", ".", ".", ".", "."])
        >>> gpsp2 = GridPegSolitairePuzzle(grid2, {"*", ".", "#"})
        >>> r2 = gpsp2.fail_fast()
        >>> r2
        False
        """
        marker, marker_set = self._marker, self._marker_set
        n, m = len(marker), len(marker[0])

        marker_list = []
        [marker_list.extend(row) for row in marker]

        position = []
        [position.append(i) for i,x in enumerate(marker_list) if x == "*"]

        for p in position:
            row_neighbor = set(self.get_row_neighbor(p, marker_list))
            col_neighbor = set(self.get_col_neighbor(p, marker_list))
            immediate = row_neighbor | col_neighbor

            if '*' not in immediate:
                return True
        return False

    def get_row_neighbor(self, i, marker_list):
        n, m = len(self._marker), len(self._marker[0])

        r = i // m
        c = i % m

        if c == 0:
            return [marker_list[i+1]]
        elif c == m-1:
            return [marker_list[i-1]]
        else:
            return [marker_list[i-1], marker_list[i+1]]

    def get_col_neighbor(self, i, marker_list):
        n, m = len(self._marker), len(self._marker[0])

        r = i // m
        c = i % m

        if r == 0:
            return [marker_list[i+m]]
        elif r == n-1:
            return [marker_list[i-m]]
        else:
            return [marker_list[i-m], marker_list[i+m]]




if __name__ == "__main__":
    import doctest

    doctest.testmod()
    from puzzle_tools import depth_first_solve

    grid = [["*", "*", "*", "*", "*"],
            ["*", "*", "*", "*", "*"],
            ["*", "*", "*", "*", "*"],
            ["*", "*", ".", "*", "*"],
            ["*", "*", "*", "*", "*"]]
    gpsp = GridPegSolitairePuzzle(grid, {"*", ".", "#"})
    import time

    start = time.time()
    solution = depth_first_solve(gpsp)
    end = time.time()
    print("Solved 5x5 peg solitaire in {} seconds.".format(end - start))
    print("Using depth-first: \n{}".format(solution))
