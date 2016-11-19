from csc148_queue import Queue


class Tree:
    """
    A bare-bones Tree ADT that identifies the root with the entire tree.
    """

    def __init__(self, value=None, children=None):
        """
        Create Tree self with content value and 0 or more children

        @param Tree self: this tree
        @param object value: value contained in this tree
        @param list[Tree] children: possibly-empty list of children
        @rtype: None
        """
        self.value = value
        # copy children if not None
        self.children = children.copy() if children else []

    def __repr__(self):
        """
        Return representation of Tree (self) as string that
        can be evaluated into an equivalent Tree.

        @param Tree self: this tree
        @rtype: str

        >>> t1 = Tree(5)
        >>> t1
        Tree(5)
        >>> t2 = Tree(7, [t1])
        >>> t2
        Tree(7, [Tree(5)])
        """
        # Our __repr__ is recursive, because it can also be called
        # via repr...!
        return ('Tree({}, {})'.format(repr(self.value), repr(self.children))
                if self.children
                else 'Tree({})'.format(repr(self.value)))

    def __eq__(self, other):
        """
        Return whether this Tree is equivalent to other.

        @param Tree self: this tree
        @param object|Tree other: object to compare to self
        @rtype: bool

        >>> t1 = Tree(5)
        >>> t2 = Tree(5, [])
        >>> t1 == t2
        True
        >>> t3 = Tree(5, [t1])
        >>> t2 == t3
        False
        """
        return (type(self) is type(other) and
                self.value == other.value and
                self.children == other.children)

    def __str__(self, indent=0):
        """
        Produce a user-friendly string representation of Tree self,
        indenting each level as a visual clue.

        @param Tree self: this tree
        @param int indent: amount to indent each level of tree
        @rtype: str

        >>> t = Tree(17)
        >>> print(t)
        17
        >>> t1 = Tree(19, [t, Tree(23)])
        >>> print(t1)
        19
           17
           23
        >>> t3 = Tree(29, [Tree(31), t1])
        >>> print(t3)
        29
           31
           19
              17
              23
        """
        root_str = indent * " " + str(self.value)
        return '\n'.join([root_str] +
                         [c.__str__(indent + 3) for c in self.children])



def list_internal(t):
    """
    Return list of values in internal nodes of t.

    @param Tree t: tree to list internal values of
    @rtype: list[object]

    >>> t = Tree(0)
    >>> list_internal(t)
    []
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> L = list_internal(t)
    >>> L.sort()
    >>> L
    [0, 1, 2]
    """
    # non recursive
    # if len(t.children) == 0:
    #     return []
    # else:
    #     l = []
    #     l.append(t.value)
    #     for child in t.children:
    #         if len(child.children) != 0:
    #             l.append(child.value)
    #             for c in child.children:
    #                 if len(c.children) != 0:
    #                     l.append(c.value)
    #     return l

    if len(t.children) == 0:
        return []
    else:
        return [t.value] + gather_lists([list_internal(c) for c in t.children])

def arity(t):
    """
    Return the maximum branching factor (arity) of Tree t.

    @param Tree t: tree to find the arity of
    @rtype: int

    >>> t = Tree(23)
    >>> arity(t)
    0
    >>> tn2 = Tree(2, [Tree(4), Tree(4.5), Tree(5), Tree(5.75)])
    >>> tn3 = Tree(3, [Tree(6), Tree(7)])
    >>> tn1 = Tree(1, [tn2, tn3])
    >>> arity(tn1)
    4
    """
    # if len(t.children) == 0: # leaf node
    #     return 0
    # else:                    # internal node
    #     l = []
    #     l.append(len(t.children))
    #     for child in t.children:
    #         if len(child.children) == 0: # leaf node
    #             return 0
    #         else:                        # internal node
    #             l.append(len(child.children))
    #     return max(l)

    if len(t.children) == 0:   # leaf node
        return 0
    else:                      # internal node
        return max([len(t.children)] + [arity(c) for c in t.children])
        # calculate max of arity of current node
        # and call arity again for these nodes having children


def contains_test_passer(t, test):
    """
    Return whether t contains a value that test(value) returns True for.

    @param Tree t: tree to search for values that pass test
    @param (object)->bool test: predicate to check values with
    @rtype: bool

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4.5, 5, 6, 7.5, 8.5], 4)
    >>> def greater_than_nine(n): return n > 9
    >>> contains_test_passer(t, greater_than_nine)
    False
    >>> def even(n): return n % 2 == 0
    >>> contains_test_passer(t, even)
    True
    """
    if len(t.children) == 0:
        return test(t.value)
    else:
        return any([test(t.value)] + [contains_test_passer(c, test) for c in t.children])


# helper function that may be useful in the functions
# above
def gather_lists(list_):
    """
    Concatenate all the sublists of L and return the result.

    @param list[list[object]] list_: list of lists to concatenate
    @rtype: list[object]

    >>> gather_lists([[1, 2], [3, 4, 5]])
    [1, 2, 3, 4, 5]
    >>> gather_lists([[6, 7], [8], [9, 10, 11]])
    [6, 7, 8, 9, 10, 11]
    """
    new_list = []
    for l in list_:
        new_list += l
    return new_list


# helpful helper function
def descendants_from_list(t, list_, arity):
    """
    Populate Tree t's descendants from list_, filling them
    in in level order, with up to arity children per node.
    Then return t.

    @param Tree t: tree to populate from list_
    @param list list_: list of values to populate from
    @param int arity: maximum branching factor
    @rtype: Tree

    >>> descendants_from_list(Tree(0), [1, 2, 3, 4], 2)
    Tree(0, [Tree(1, [Tree(3), Tree(4)]), Tree(2)])
    """
    q = Queue()
    q.add(t)
    list_ = list_.copy()
    while not q.is_empty():  # unlikely to happen
        new_t = q.remove()
        for i in range(0, arity):
            if len(list_) == 0:
                return t  # our work here is done
            else:
                new_t_child = Tree(list_.pop(0))
                new_t.children.append(new_t_child)  # t.children is a list of child trees
                q.add(new_t_child)
    return t


if __name__ == '__main__':
    import doctest
    doctest.testmod()
