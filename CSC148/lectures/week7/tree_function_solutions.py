# module-level functions, alternatively these could be implemented as
# methods, but then they might not be appropriate for every Tree
from tree import Tree, descendants_from_list
from csc148_queue import Queue


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
    return max([len(t.children)] + [arity(n) for n in t.children])


def count(t):
    """
    Return the number of nodes in Tree t.


    @param Tree t: tree to find number of nodes in
    @rtype: int

    >>> t = Tree(17)
    >>> count(t)
    1
    >>> t4 = descendants_from_list(Tree(17), [0, 2, 4, 6, 8, 10, 11], 4)
    >>> count(t4)
    8
    """
    return 1 + sum([count(n) for n in t.children])


def height(t):
    """
    Return 1 + length of longest path of t.

    @param Tree t: tree to find height of
    @rtype: int

    >>> t = Tree(13)
    >>> height(t)
    1
    >>> t = descendants_from_list(Tree(13), [0, 1, 3, 5, 7, 9, 11, 13], 3)
    >>> height(t)
    3
    """
    # 1 more edge than the maximum height of a child, except
    # what do we do if there are no children?
    return 1 + max([height(c) for c in t.children]) if t.children else 1


def leaf_count(t):
    """
    Return the number of leaves in Tree t.

    @param Tree t: tree to count number of leaves of
    @rtype: int

    >>> t = Tree(7)
    >>> leaf_count(t)
    1
    >>> t = descendants_from_list(Tree(7), [0, 1, 3, 5, 7, 9, 11, 13], 3)
    >>> leaf_count(t)
    6
    """
    return (sum([leaf_count(c) for c in t.children]) +
            (0 if t.children else 1))


def list_all(t):
    """
    Return list of values in t.

    @param Tree t: tree to list values of
    @rtype: list[object]

    >>> t = Tree(0)
    >>> list_all(t)
    [0]
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> list_ = list_all(t)
    >>> list_.sort()
    >>> list_
    [0, 1, 2, 3, 4, 5, 6, 7, 8]
    """
    # implicit base case when len(t.children) == 0
    return [t.value] + gather_lists([list_all(c) for c in t.children])


def list_leaves(t):
    """
    Return list of values in leaves of t.

    @param Tree t: tree to list leaf values of
    @rtype: list[object]

    >>> t = Tree(0)
    >>> list_leaves(t)
    [0]
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> list_ = list_leaves(t)
    >>> list_.sort() # so list_ is predictable to compare
    >>> list_
    [3, 4, 5, 6, 7, 8]
    """
    return ([t.value] if len(t.children) == 0  # root is a leaf, base case
            else gather_lists([list_leaves(c) for c in t.children]))


def list_interior(t):
    """
    Return list of values in interior nodes of t.

    @param Tree t: tree to list interior values of
    @rtype: list[object]

    >>> t = Tree(0)
    >>> list_interior(t)
    []
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> L = list_interior(t)
    >>> L.sort()
    >>> L
    [0, 1, 2]
    """
    if len(t.children) == 0:
        return []
    else:
        return ([t.value] +
                gather_lists([list_interior(c) for c in t.children]))


def count_internal(t):
    """
    Return number of internal nodes of t.

    @param Tree t: tree to count internal nodes of
    @rtype: int

    >>> t = Tree(0)
    >>> count_internal(t)
    0
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> count_internal(t)
    3
    """
    if len(t.children) == 0:
        return 0
    else:
        return (1 +
                sum([count_internal(c) for c in t.children]))


def list_if(t, p):
    """
    Return a list of values in Tree t that satisfy predicate p(value).

    @param Tree t: tree to list values that satisfy predicate p
    @param (object)->bool p: predicate to check values with
    @rtype: list[object]

    >>> def p(v): return v > 4
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> list_ = list_if(t, p)
    >>> list_.sort()
    >>> list_
    [5, 6, 7, 8]
    >>> def p(v): return v % 2 == 0
    >>> list_ = list_if(t, p)
    >>> list_.sort()
    >>> list_
    [0, 2, 4, 6, 8]
    """
    return (([t.value] if p(t.value) else []) +
            gather_lists([list_if(c, p) for c in t.children]))


def count_if(t, p):
    """
    Return number of values in Tree t that satisfy predicate p(value).

    @param Tree t: tree to count values that satisfy predicate p
    @param (object)->bool p: predicate to check values with
    @rtype: list[object]

    >>> def p(v): return v > 4
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> count_if(t, p)
    4
    >>> def p(v): return v % 2 == 0
    >>> count_if(t, p)
    5
    """
    return (1 if p(t.value) else 0) + sum([count_if(c, p) for c in t.children])


def list_below(t, n):
    """
    Return list of values in t from nodes with paths no longer
    than n from root.

    @param Tree t: tree to list values from
    @param int n: limit on path lengths
    @rtype: list[object]

    >>> t = Tree(0)
    >>> list_below(t, 0)
    [0]
    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7, 8], 3)
    >>> L = list_below(t, 1)
    >>> L.sort()
    >>> L
    [0, 1, 2, 3]
    """
    return ([] if n < 0  # don't list this node or its children
            else [t.value] +  # list this node and possibly below it
            gather_lists([list_below(c, n - 1) for c in t.children]))


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
    return test(t.value) or any([test(c.value) for c in t.children])


def preorder_visit(t, act):
    """
    Visit each node of Tree t in preorder, and act on the nodes
    as they are visited.

    @param Tree t: tree to visit in preorder
    @param (Tree)->Any act: function to carry out on visited Tree node
    @rtype: None

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7], 3)
    >>> def act(node): print(node.value)
    >>> preorder_visit(t, act)
    0
    1
    4
    5
    6
    2
    7
    3
    """
    act(t)
    for c in t.children:
        preorder_visit(c, act)


def postorder_visit(t, act):
    """
    Visit each node of t in postorder, and act on it when it is visited.

    @param Tree t: tree to be visited in postorder
    @param (Tree)->Any act: function to do to each node
    @rtype: None

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7], 3)
    >>> def act(node): print(node.value)
    >>> postorder_visit(t, act)
    4
    5
    6
    1
    7
    2
    3
    0
    """
    for c in t.children:
        postorder_visit(c, act)
    act(t)


def levelorder_visit(t, act):
    """
    Visit every node in Tree t in level order and act on the node
    as you visit it.

    @param Tree t: tree to visit in level order
    @param (Tree)->Any act: function to execute during visit

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7], 3)
    >>> def act(node): print(node.value)
    >>> levelorder_visit(t, act)
    0
    1
    2
    3
    4
    5
    6
    7
    """
    q = Queue()
    q.add(t)
    while not q.is_empty():
        next_t = q.remove()
        act(next_t)
        for c in next_t.children:
            q.add(c)


def visit_level(t, n, act):
    """
    Visit nodes of t at level n, act on them, and return the number
    visited.

    @param Tree t: tree to visit level n of
    @param int n: level (depth) to visit
    @param (Tree)->object act: function to execute at level n
    @rtype: int

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7], 3)
    >>> def act(node): print(node.value)
    >>> visit_level(t, 1, act)
    1
    2
    3
    3
    """
    if n == 0:
        act(t)
        return 1
    else:
        return sum([visit_level(c, n-1, act) for c in t.children])


def levelorder_visit(t, act):
    """
    Visit Tree t in level order and act on its nodes.

    @param Tree t: Tree to visit in level order
    @param (Tree)->object act: function to execute on visit

    >>> t = descendants_from_list(Tree(0), [1, 2, 3, 4, 5, 6, 7], 3)
    >>> def act(node): print(node.value)
    >>> levelorder_visit(t, act)
    0
    1
    2
    3
    4
    5
    6
    7
    """
    visited, n = visit_level(t, 0, act), 0
    while visited > 0:
        n += 1
        visited = visit_level(t, n, act)


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


if __name__ == '__main__':
    import doctest

    doctest.testmod()
