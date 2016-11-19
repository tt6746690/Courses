"""
Some functions for working with puzzles
"""
from puzzle import Puzzle
from collections import deque
import random

# set higher recursion limit
# which is needed in PuzzleNode.__str__
# uncomment the next two lines on a unix platform, say CDF
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
import sys
sys.setrecursionlimit(10**6)


# TODO
# implement depth_first_solve
# do NOT change the type contract
# you are welcome to create any helper functions
# you like
def depth_first_solve(puzzle):
    """
    Return a path from PuzzleNode(puzzle) to a PuzzleNode containing
    a solution, with each child containing an extension of the puzzle
    in its parent.  Return None if this is not possible.

    @type puzzle: Puzzle
    @rtype: PuzzleNode
    """
    l = set()
    d = deque()
    root = PuzzleNode(puzzle)
    d.append(root)

    while not len(d) == 0:
        puzzle_node = d.pop()   # as stack
        puzz = puzzle_node.puzzle
        if duplicate(puzz, l):
            continue
        elif puzz.is_solved():
            return build_correct_path(puzzle_node)
        elif len(puzz.extensions()) == 0:
            continue
        else:   # internal node with extension that is not solved
            ext = puzz.extensions()
            for e in ext:
                children_list = puzzle_node.children
                children_list.append(PuzzleNode(e, None, puzzle_node))
            for child in puzzle_node.children:
                d.append(child)

def build_correct_path(puzzle_node):
    """
    build the correct puzzle path from solution by tracing its parents.

    @param PuzzleNode puzzle_node: solution puzzle_node
    @rtype: PuzzleNode
    """
    current = puzzle_node
    d2 = deque()
    while current.parent:
        d2.append(current)
        current = current.parent
    return_node = d2.pop()  # as stack
    cur_node = return_node
    while not len(d2) == 0:
        next_node = d2.pop()
        cur_node.children = []
        cur_node.children = [next_node]
        cur_node = next_node
    return return_node

def duplicate(puzzle, l):
    """
    return True if puzzle is duplicate, otherwise store it in a set

    @parem PuzzleNode puzzle: this PuzzleNode
    @rtype: bool
    """
    if str(puzzle) in l:
        return True
    else:
        l.add(str(puzzle))
        return False

# TODO
# implement breadth_first_solve
# do NOT change the type contract
# you are welcome to create any helper functions
# you like
# Hint: you may find a queue useful, that's why
# we imported deque
def breadth_first_solve(puzzle):
    """
    Return a path from PuzzleNode(puzzle) to a PuzzleNode containing
    a solution, with each child PuzzleNode containing an extension
    of the puzzle in its parent.  Return None if this is not possible.

    @type puzzle: Puzzle
    @rtype: PuzzleNode
    """
    l = set()
    d = deque()
    root = PuzzleNode(puzzle)
    d.append(root)

    while not len(d) == 0:
        puzzle_node = d.popleft()   # as queue
        puzz = puzzle_node.puzzle
        if duplicate(puzz, l):
            continue
        elif puzz.is_solved():
            return build_correct_path(puzzle_node)
        elif len(puzz.extensions()) == 0:
            continue
        else:   # internal node with extension that is not solved
            ext = puzz.extensions()
            for e in ext:
                children_list = puzzle_node.children
                children_list.append(PuzzleNode(e, None, puzzle_node))
            for child in puzzle_node.children:
                d.append(child)



# Class PuzzleNode helps build trees of PuzzleNodes that have
# an arbitrary number of children, and a parent.
class PuzzleNode:
    """
    A Puzzle configuration that refers to other configurations that it
    can be extended to.
    """

    def __init__(self, puzzle=None, children=None, parent=None):
        """
        Create a new puzzle node self with configuration puzzle.

        @type self: PuzzleNode
        @type puzzle: Puzzle | None
        @type children: list[PuzzleNode]
        @type parent: PuzzleNode | None
        @rtype: None
        """
        self.puzzle, self.parent = puzzle, parent
        if children is None:
            self.children = []
        else:
            self.children = children[:]

    def __eq__(self, other):
        """
        Return whether PuzzleNode self is equivalent to other

        @type self: PuzzleNode
        @type other: PuzzleNode | Any
        @rtype: bool

        >>> from word_ladder_puzzle import WordLadderPuzzle
        >>> pn1 = PuzzleNode(WordLadderPuzzle("on", "no", {"on", "no", "oo"}))
        >>> pn2 = PuzzleNode(WordLadderPuzzle("on", "no", {"on", "oo", "no"}))
        >>> pn3 = PuzzleNode(WordLadderPuzzle("no", "on", {"on", "no", "oo"}))
        >>> pn1.__eq__(pn2)
        True
        >>> pn1.__eq__(pn3)
        False
        """
        return (type(self) == type(other) and
                self.puzzle == other.puzzle and
                all([x in self.children for x in other.children]) and
                all([x in other.children for x in self.children]))

    def __str__(self):
        """
        Return a human-readable string representing PuzzleNode self.

        # doctest not feasible.
        """
        return "{}\n\n{}".format(self.puzzle,
                                 "\n".join([str(x) for x in self.children]))
