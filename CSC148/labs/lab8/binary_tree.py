class BinaryTree:
    """
    A Binary Tree, i.e. arity 2.

    === Attributes ===
    @param object data: data for this binary tree node
    @param BinaryTree|None left: left child of this binary tree node
    @param BinaryTree|None right: right child of this binary tree node
    """

    def __init__(self, data, left=None, right=None):
        """
        Create BinaryTree self with data and children left and right.

        @param BinaryTree self: this binary tree
        @param object data: data of this node
        @param BinaryTree|None left: left child
        @param BinaryTree|None right: right child
        @rtype: None
        """
        self.data, self.left, self.right = data, left, right

    def __eq__(self, other):
        """
        Return whether BinaryTree self is equivalent to other.

        @param BinaryTree self: this binary tree
        @param Any other: object to check equivalence to self
        @rtype: bool

        >>> BinaryTree(7).__eq__("seven")
        False
        >>> b1 = BinaryTree(7, BinaryTree(5))
        >>> b1.__eq__(BinaryTree(7, BinaryTree(5), None))
        True
        """
        return (type(self) == type(other) and
                self.data == other.data and
                (self.left, self.right) == (other.left, other.right))

    def __repr__(self):
        """
        Represent BinaryTree (self) as a string that can be evaluated to
        produce an equivalent BinaryTree.

        @param BinaryTree self: this binary tree
        @rtype: str

        >>> BinaryTree(1, BinaryTree(2), BinaryTree(3))
        BinaryTree(1, BinaryTree(2, None, None), BinaryTree(3, None, None))
        """
        return "BinaryTree({}, {}, {})".format(repr(self.data),
                                               repr(self.left),
                                               repr(self.right))

    def __str__(self, indent=""):
        """
        Return a user-friendly string representing BinaryTree (self)
        inorder.  Indent by indent.

        >>> b = BinaryTree(1, BinaryTree(2, BinaryTree(3)), BinaryTree(4))
        >>> print(b)
            4
        1
            2
                3
        <BLANKLINE>
        """
        right_tree = (self.right.__str__(
            indent + "    ") if self.right else "")
        left_tree = self.left.__str__(indent + "    ") if self.left else ""
        return (right_tree + "{}{}\n".format(indent, str(self.data)) +
                left_tree)

    def __contains__(self, value):
        """
        Return whether tree rooted at node contains value.

        @param BinaryTree self: binary tree to search for value
        @param object value: value to search for
        @rtype: bool

        >>> BinaryTree(5, BinaryTree(7), BinaryTree(9)).__contains__(7)
        True
        """
        return (self.data == value or
                (self.left and value in self.left) or
                (self.right and value in self.right))


def parenthesize(b):
    """
    Return a parenthesized expression equivalent to the arithmetic
    expression tree rooted at b.

    Assume:  -- b is a binary tree
             -- interior nodes contain data in {'+', '-', '*', '/'}
             -- interior nodes always have two children
             -- leaves contain float data

    @param BinaryTree b: arithmetic expression tree
    @rtype: str

    >>> b1 = BinaryTree(3.0)
    >>> print(parenthesize(b1))
    3.0
    >>> b2 = BinaryTree(4.0)
    >>> b3 = BinaryTree(7.0)
    >>> b4 = BinaryTree("*", b1, b2)
    >>> b5 = BinaryTree("+", b4, b3)
    >>> print(parenthesize(b5))
    ((3.0 * 4.0) + 7.0)
    """

    if b.right is None and b.left is None:  # leaf node
        return b.data
    else:   # internal node alwyas has two children
        return "(" + str(parenthesize(b.left)) + " " + str(b.data) + " " \
                    + str(parenthesize(b.right)) + ")"

def list_longest_path(node):
    """
    List the data in a longest path of node.

    @param BinaryTree|None node: tree to list longest path of
    @rtype: list[object]

    >>> list_longest_path(None)
    []
    >>> list_longest_path(BinaryTree(5))
    [5]
    >>> b1 = BinaryTree(7)
    >>> b2 = BinaryTree(3, BinaryTree(2), None)
    >>> b3 = BinaryTree(5, b2, b1)
    >>> list_longest_path(b3)
    [5, 3, 2]
    """
    if node is None:
        return []
    elif node.left is None and node.right is None:
        return [node.data]
    else:
        left = [node.data]
        right = left.copy()
        left.extend(list_longest_path(node.left))
        right.extend(list_longest_path(node.right))
        return left if len(left) > len(right) else right




def insert(node, data):
    """
    Insert data in BST rooted at node if necessary, and return new root.

    Assume node is the root of a Binary Search Tree.

    @param BinaryTree node: root of a binary search tree.
    @param object data: data to insert into BST, if necessary.

    >>> b = BinaryTree(5)
    >>> b1 = insert(b, 3)
    >>> print(b1)
    5
        3
    <BLANKLINE>
    """
    return_node = node
    if not node:
        return_node = BinaryTree(data)
    elif data < node.data:
        node.left = insert(node.left, data)
    elif data > node.data:
        node.right = insert(node.right, data)
    else:  # nothing to do
        pass
    return return_node


def list_between(node, start, end):
    """
    Return a Python list of all values in the binary search tree
    rooted at node that are between start and end (inclusive).

    @param BinaryTree|None node: binary tree to list values from
    @param object start: starting value for list
    @param object end: stopping value for list
    @rtype: list[object]

    >>> list_between(None, 3, 13)
    []
    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> list_between(b, 2, 3)
    [2]
    >>> L = list_between(b, 3, 11)
    >>> L.sort()
    >>> L
    [4, 6, 8, 10]
    """
    if node is None:
        return []
    else:
        r = [node.data] if start <= node.data <= end else []
        return r + list_between(node.left, start, end) \
            + list_between(node.right, start, end)
        # not really efficient. add if statement to check if needs to
        # list between left and right child


if __name__ == "__main__":
    import doctest
    doctest.testmod()
