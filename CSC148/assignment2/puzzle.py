class Puzzle:
    """"
    Snapshot of a full-information puzzle, which may be solved, unsolved,
    or even unsolvable.
    """

    def fail_fast(self):
        """
        Return True iff Puzzle self can never be extended to a solution.

        Override this in a subclass where you can determine early that
        this Puzzle can't be solved.

        @type self: Puzzle
        @rtype: bool
        """
        return False

    def is_solved(self):
        """
        Return True iff Puzzle self is solved.

        This is an abstract method that must be implemented
        in a subclass.

        @type self: Puzzle
        @rtype: bool
        """
        raise NotImplementedError

    def extensions(self):
        """
        Return list of legal extensions of Puzzle self.

        This is an abstract method that must be implemented
        in a subclass.

        @type self: Puzzle
        @rtype: list[Puzzle]
        """
        raise NotImplementedError
