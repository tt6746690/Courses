from puzzle import Puzzle


class SudokuPuzzle(Puzzle):
    """
    A sudoku puzzle that may be solved, unsolved, or even unsolvable.
    """

    def __init__(self, n, symbols, symbol_set):
        """
        Create a new nxn SudokuPuzzle self with symbols
        from symbol_set already selected.

        @type self: SudokuPuzzle
        @type n: int
        @type symbols: list[str]
        @type symbol_set: set[str]
        """
        assert n > 0
        assert round(n ** (1 / 2)) * round(n ** (1 / 2)) == n  # n are squares
        # character in symbols are defined
        assert all([d in (symbol_set | {"*"}) for d in symbols])
        assert len(symbol_set) == n       # symbol_set are unique characters
        assert len(symbols) == n ** 2     # symbol size is n square
        self._n, self._symbols, self._symbol_set = n, symbols, symbol_set

    def __eq__(self, other):
        """
        Return whether SudokuPuzzle self is equivalent to other.

        @type self: SudokuPuzzle
        @type other: SudokuPuzzle | Any
        @rtype: bool

        >>> grid1 = ["A", "B", "C", "D"]
        >>> grid1 += ["D", "C", "B", "A"]
        >>> grid1 += ["*", "D", "*", "*"]
        >>> grid1 += ["*", "*", "*", "*"]
        >>> s1 = SudokuPuzzle(4, grid1, {"A", "B", "C", "D"})
        >>> grid2 = ["A", "B", "C", "D"]
        >>> grid2 += ["D", "C", "B", "A"]
        >>> grid2 += ["*", "D", "*", "*"]
        >>> grid2 += ["*", "*", "*", "*"]
        >>> s2 = SudokuPuzzle(4, grid2, {"A", "B", "C", "D"})
        >>> s1.__eq__(s2)
        True
        >>> grid3 = ["A", "B", "C", "D"]
        >>> grid3 += ["D", "C", "B", "A"]
        >>> grid3 += ["*", "D", "*", "*"]
        >>> grid3 += ["*", "A", "*", "*"]
        >>> s3 = SudokuPuzzle(4, grid3, {"A", "B", "C", "D"})
        >>> s1.__eq__(s3)
        False
        """
        return (type(other) == type(self) and
                self._n == other._n and self._symbols == other._symbols and
                self._symbol_set == other._symbol_set)

    def __str__(self):
        """
        Return a human-readable string representation of SudokuPuzzle self.

        @param SudokuPuzzle self: this SudokuPuzzle
        @rtype: str

        >>> grid = ["A", "B", "C", "D"]
        >>> grid += ["D", "C", "B", "A"]
        >>> grid += ["*", "D", "*", "*"]
        >>> grid += ["*", "*", "*", "*"]
        >>> s = SudokuPuzzle(4, grid, {"A", "B", "C", "D"})
        >>> print(s)
        AB|CD
        DC|BA
        -----
        *D|**
        **|**
        """

        def row_pickets(row):
            """
                    Return string of characters in row with | divider
            between groups of sqrt(n)

            @type row: list[str]
            @rtype: str
            """
            string_list = []
            r = round(self._n ** (1 / 2))
            for i in range(self._n):
                if i > 0 and i % r == 0:
                    string_list.append("|")
                string_list.append(row[i])
            return "".join(string_list)

        def table_dividers(table):
            """
            Return rows of strings in table with
            "-----" dividers between groups of sqrt(n) rows.

            @type table: list[str]
            @rtype: list[str]
            """
            r = round(self._n ** (1 / 2))
            t, divider = [], "-" * (self._n + r - 1)
            for i in range(self._n):
                if i > 0 and i % r == 0:
                    t.append(divider)
                t.append(table[i])
            return t

        rows = [row_pickets([self._symbols[r * self._n + c]
                             for c in range(self._n)])
                for r in range(self._n)] # r = row#
        rows = table_dividers(rows) # here rows = list[str]
        return "\n".join(rows)

    def is_solved(self):
        """
        Return whether SudokuPuzzle self is solved.

        @type self: SudokuPuzzle
        @rtype: bool

        >>> grid = ["A", "B", "C", "D"]
        >>> grid += ["C", "D", "A", "B"]
        >>> grid += ["B", "A", "D", "C"]
        >>> grid += ["D", "C", "B", "A"]
        >>> s = SudokuPuzzle(4, grid, {"A", "B", "C", "D"})
        >>> s.is_solved()
        True
        >>> grid[9] = "D"
        >>> grid[10] = "A"
        >>> s = SudokuPuzzle(4, grid, {"A", "B", "C", "D"})
        >>> s.is_solved()
        False
        """
        # convenient names
        n, symbols = self._n, self._symbols
        # no "*" left and all rows, column, subsquares have correct symbols
        return ("*" not in symbols and
                all([(self._row_set(i) == self._symbol_set and
                      self._column_set(i) == self._symbol_set and
                      self._subsquare_set(i) ==
                      self._symbol_set) for i in range(n ** 2)]))

    def extensions(self):
        """
        Return list of extensions of SudokuPuzzle self.

        @type self: SudokuPuzzle
        @rtype: list[SudokuPuzzle]

        >>> grid = ["A", "B", "C", "D"]
        >>> grid += ["C", "D", "A", "B"]
        >>> grid += ["B", "A", "D", "C"]
        >>> grid += ["D", "C", "B", "*"]
        >>> s = SudokuPuzzle(4, grid, {"A", "B", "C", "D"})
        >>> L1 = list(s.extensions())
        >>> grid[-1] = "A"
        >>> L2 = [SudokuPuzzle(4, grid, {"A", "B", "C", "D"})]
        >>> len(L1) == len(L2)
        True
        >>> all([s in L2 for s in L1])
        True
        >>> all([s in L1 for s in L2])
        True
        """
        # convenient names
        symbols, symbol_set, n = self._symbols, self._symbol_set, self._n
        if "*" not in symbols:
            # return an empty list
            return [_ for _ in []]
        else:
            # position of first empty position
            i = symbols.index("*")
            # allowed symbols at position i
            # A | B == A.union(B)
            allowed_symbols = (self._symbol_set -
                               (self._row_set(i) |
                                self._column_set(i) |
                                self._subsquare_set(i)))
            # list of SudokuPuzzles with each legal digit at position i
            return (
                [SudokuPuzzle(n,
                 symbols[:i] + [d] + symbols[i + 1:], symbol_set)
                 for d in allowed_symbols])

    # TODO
    # override fail_fast
    # Notice that it is not possible to complete a sudoku puzzle if there
    # is one open position that has no symbols available to put in it.  In
    # other words, if there is one open position where the symbols already used
    # in the same row, column, and subsquare exhaust the symbols available,
    # there is no point in continuing.
    def fail_fast(self):
        """
        Return True if there is at least one open position that has
        no symbols available to put in it.

        @type self: SudokuPuzzle
        @rtype: bool

        >>> grid = ["A", "B", "C", "D"]
        >>> grid += ["C", "D", "A", "B"]
        >>> grid += ["B", "A", "D", "C"]
        >>> grid += ["D", "C", "A", "*"]
        >>> s = SudokuPuzzle(4, grid, {"A", "B", "C", "D"})
        >>> r = s.fail_fast()
        >>> r
        True
        >>> grid[-2] = "B"
        >>> L2 = [SudokuPuzzle(4, grid, {"A", "B", "C", "D"})]
        >>> r = s.fail_fast()
        >>> r
        False
        """
        symbols, n, symbol_set = self._symbols, self._n, self._symbol_set

        if "*" not in symbols:
            return False
        else:
            # all position of next move
            i = [i for i,x in enumerate(symbols) if x == "*"]
            # test for no available move at all i
            return any([len(symbol_set - (self._row_set(x) |
                        self._column_set(x) | self._subsquare_set(x))) == 0
                        for x in i])

    # some helper methods
    def _row_set(self, m):
        #
        # Return set of symbols in row of SudokuPuzzle self's symbols
        # where position m occurs.
        #
        # @type self: SudokuPuzzle
        # @type m: int
        assert 0 <= m < self._n ** 2
        # convenient names
        n, symbols = self._n, self._symbols
        # first position in m's row
        r = (m // n) * n
        # set of elements from symbols[r] .. symbols[r+n-1]
        return set([symbols[r + i] for i in range(n)])

    def _column_set(self, m):
        # Return set of symbols in column of SudokuPuzzle self's symbols
        # where position m occurs.
        #
        # @type self: SudokuPuzzle
        # @type m: int
        assert 0 <= m <= self._n ** 2
        # convenient names
        symbols, n = self._symbols, self._n
        # first position in m's column
        c = m % n
        # set of elements from symbols[c], symbols[c + n],
        # ... symbols[c + (n * (n-1))]
        return set([symbols[c + (i * n)] for i in range(n)])

    def _subsquare_set(self, m):
        # Return set of symbols in subsquare of SudokuPuzzle self's symbols
        # where position m occurs.
        #
        # @type self: SudokuPuzzle
        # @type m: int
        assert 0 <= m < self._n ** 2
        # convenient names
        n, symbols = self._n, self._symbols
        # row, column where m occur
        row, col = m // n, m % n
        # length of subsquares
        ss = round(n ** (1 / 2))
        # upper-left position of m's subsquare
        ul = (((row // ss) * ss) * n) + ((col // ss) * ss)
        # return set of symbols from subsquare starting at ul
        return set(
            [symbols[ul + i + n * j] for i in range(ss) for j in range(ss)])


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    s = SudokuPuzzle(9,
                     ["*", "*", "*", "7", "*", "8", "*", "1", "*",
                      "*", "*", "7", "*", "9", "*", "*", "*", "6",
                      "9", "*", "3", "1", "*", "*", "*", "*", "*",
                      "3", "5", "*", "8", "*", "*", "6", "*", "1",
                      "*", "*", "*", "*", "*", "*", "*", "*", "*",
                      "1", "*", "6", "*", "*", "9", "*", "4", "8",
                      "*", "*", "*", "*", "*", "1", "2", "*", "7",
                      "8", "*", "*", "*", "7", "*", "4", "*", "*",
                      "*", "6", "*", "3", "*", "2", "*", "*", "*"],
                     {"1", "2", "3", "4", "5", "6", "7", "8", "9"})

    from time import time

    print("solving sudoku from July 9 2015 Star... \n\n{}\n\n".format(s))
    from puzzle_tools import depth_first_solve

    start = time()
    sol = depth_first_solve(s)
    print(sol)
    while sol.children:
        sol = sol.children[0]
    end = time()
    print("time to solve 9x9 using depth_first: "
          "{} seconds\n".format(end - start))
    print(sol)

    s = SudokuPuzzle(9,
                     ["*", "*", "*", "9", "*", "2", "*", "*", "*",
                      "*", "9", "1", "*", "*", "*", "6", "3", "*",
                      "*", "3", "*", "*", "7", "*", "*", "8", "*",
                      "3", "*", "*", "*", "*", "*", "*", "*", "8",
                      "*", "*", "9", "*", "*", "*", "2", "*", "*",
                      "5", "*", "*", "*", "*", "*", "*", "*", "7",
                      "*", "7", "*", "*", "8", "*", "*", "4", "*",
                      "*", "4", "5", "*", "*", "*", "8", "1", "*",
                      "*", "*", "*", "3", "*", "6", "*", "*", "*"],
                     {"1", "2", "3", "4", "5", "6", "7", "8", "9"})

    print("solving 3-star sudoku from \"That's Puzzling\","
          "November 14th 2015\n\n{}\n\n".format(s))
    start = time()
    sol = depth_first_solve(s)
    while sol.children:
        sol = sol.children[0]
    end = time()
    print("time to solve 9x9 using depth_first: {} seconds\n".format(
        end - start))
    print(sol)

    s = SudokuPuzzle(9,
                     ["5", "6", "*", "*", "*", "7", "*", "*", "9",
                      "*", "7", "*", "*", "4", "8", "*", "3", "1",
                      "*", "*", "*", "*", "*", "*", "*", "*", "*",
                      "4", "3", "*", "*", "*", "*", "*", "*", "*",
                      "*", "8", "*", "*", "*", "*", "*", "9", "*",
                      "*", "*", "*", "*", "*", "*", "*", "2", "6",
                      "*", "*", "*", "*", "*", "*", "*", "*", "*",
                      "1", "9", "*", "3", "6", "*", "*", "7", "*",
                      "7", "*", "*", "1", "*", "*", "*", "4", "2"],
                     {"1", "2", "3", "4", "5", "6", "7", "8", "9"})

    print(
        "solving 4-star sudoku from \"That's Puzzling\", "
        "November 14th 2015\n\n{}\n\n".format(
            s))
    start = time()
    sol = depth_first_solve(s)
    while sol.children:
        sol = sol.children[0]
    end = time()
    print("time to solve 9x9 using depth_first: {} seconds\n".format(
        end - start))
    print(sol)
