from puzzle import Puzzle

# if exis with >> Process finished with exit code -1073741571 (0xC00000FD)
# please run it again it happens time to time
class WordLadderPuzzle(Puzzle):
    """
    A word-ladder puzzle that may be solved, unsolved, or even unsolvable.
    """

    def __init__(self, from_word, to_word, ws):
        """
        Create a new word-ladder puzzle with the aim of stepping
        from from_word to to_word using words in ws, changing one
        character at each step.

        @type from_word: str
        @type to_word: str
        @type ws: set[str]
        @rtype: None
        """
        (self._from_word, self._to_word, self._word_set) = (from_word,
                                                            to_word, ws)
        # set of characters to use for 1-character changes
        self._chars = "abcdefghijklmnopqrstuvwxyz"

        # TODO
        # implement __eq__ and __str__
        # __repr__ is up to you

    def __str__(self):
        """
        Return a human-readable string representation of WordLadderPuzzle
        self.

        @param WordLadderPuzzles self: this WordLadderPuzzle

        >>> w = WordLadderPuzzle("hi", "yo", {"ni", "yi"})
        >>> print(w)
        hi --> yo
        """
        return self._from_word + ' --> ' + self._to_word

    def __eq__(self, other):
        """
        Return whether WordLadderPuzzle self is equivalent to other.

        @type self: WordLadderPuzzle
        @type other: WordLadderPuzzle | Any
        @rtype: bool

        >>> w1 = WordLadderPuzzle("hi", "yo", {"ni", "yi"})
        >>> w2 = WordLadderPuzzle("hi", "yo", {"ni", "yi"})
        >>> w1 == w2
        True
        >>> w3 = WordLadderPuzzle("hi", "yk", {"ni", "yi"})
        >>> w1 == w3
        False
        """
        return (type(self) == type(other) and
                self._from_word == other._from_word and
                self._to_word == other._to_word and
                self._word_set == other._word_set)

        # TODO
        # override extensions
        # legal extensions are WordLadderPuzzles that have a from_word that can
        # be reached from this one by changing a single letter to one of those
        # in self._chars

    def extensions(self):
        """
        Return list of extensions of WordLadderPuzzle self.

        @type self: WordLadderPuzzle
        @rtype: list[WordLadderPuzzle]

        >>> w = WordLadderPuzzle("hi", "yo", {"ni", "yi"})
        >>> r = w.extensions()
        >>> i = WordLadderPuzzle("ni", "yo", {"ni", "yi"})
        >>> j = WordLadderPuzzle("yi", "yo", {"ni", "yi"})
        >>> i in r
        True
        >>> j in r
        True
        """

        f, t, s, c = self._from_word, self._to_word, self._word_set, self._chars
        e = []

        if f == t:
            return []
        elif not len(f) == len(t):
            return []
        else:
            for i in range(len(f)):
                available_char = set(iter(c)) - set(f[i])
                for char in available_char:
                    n = f[:i] + char + f[i+1:]
                    if n in s:
                        e.append(WordLadderPuzzle(n, t, s))
        return e

        # TODO
        # override is_solved
        # this WordLadderPuzzle is solved when _from_word is the same as
        # _to_word

    def is_solved(self):
        """
        Return whether WordLadderPuzzle self is solved.

        @type self: WordLadderPuzzle
        @rtype: bool

        >>> w = WordLadderPuzzle("hi", "yo", {"yi", "yo"})
        >>> w.is_solved()
        False
        >>> w2 = WordLadderPuzzle("yo", "yo", {"yi", "yo"})
        >>> w2.is_solved()
        True
        """
        return self._from_word == self._to_word


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    from puzzle_tools import breadth_first_solve, depth_first_solve
    from time import time
    with open("words", "r") as words:
        word_set = set(words.read().split())
    w = WordLadderPuzzle("same", "cost", word_set)
    start = time()
    sol = breadth_first_solve(w)
    end = time()
    print("Solving word ladder from same->cost")
    print("...using breadth-first-search")
    print("Solutions: {} took {} seconds.".format(sol, end - start))
    start = time()
    sol = depth_first_solve(w)
    end = time()
    print("Solving word ladder from same->cost")
    print("...using depth-first-search")
    print("Solutions: {} took {} seconds.".format(sol, end - start))
