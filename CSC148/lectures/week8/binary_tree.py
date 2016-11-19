class BinaryTree:
    """
    A Binary Tree, i.e. arity 2.
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
                (self.left, self.right) == (other.left, other.right))   # use tuple to simplify comparison
                # equate calls __eq__() again if self.left / self.right are subtrees

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
        right_tree = self.right.__str__(indent + "    ") if self.right else ""
        left_tree = self.left.__str__(indent + "    ") if self.left else ""
        return (right_tree + "{}{}\n".format(indent, str(self.data)) +
                left_tree)

    #
    # def __contains__(self, value):
    #
    #     if self.data == value:
    #         return true
    #     elif self.left == None or self.right == None:
    #         return false
    #     else:
    #         return any([self.right.__contains__(value), self.left.__contains(value)])
    #


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
                (self.left and value in self.left) or   # check if left exists
                (self.right and value in self.right))   # value in calls __contains__()


def evaluate(b):
    """
    Evaluate the expression rooted at b.  If b is a leaf,
    return its float data.  Otherwise, evaluate b.left and
    b.right and combine them with b.data.

    Assume:  -- b is a non-empty binary tree
             -- interior nodes contain data in {"+", "-", "*", "/"}
             -- interior nodes always have two children
             -- leaves contain float data

     @param BinaryTree b: binary tree representing arithmetic expression
     @rtype: float

    >>> b = BinaryTree(3.0)
    >>> evaluate(b)
    3.0
    >>> b = BinaryTree("*", BinaryTree(3.0), BinaryTree(4.0))
    >>> evaluate(b)
    12.0
    """
    if b.left == None and b.right == None:
        return b.data
    else:
        return eval(str(evaluate(b.left)) + str(b.data) + str(evaluate(b.right)))


def preorder_visit(root, act):
    if root is not None:
        act(root)
        preorder_visit(root, left, act)
        preorder_visit(root, right, act)

def postorder_visit(root, act):
    """
    Visit BinaryTree t in postorder and act on nodes as you visit.

    @param BinaryTree|None t: binary tree to visit
    @param (BinaryTree)->Any act: function to use on nodes
    @rtype: None

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> def f(node): print(node.data)
    >>> postorder_visit(b, f)
    2
    6
    4
    10
    14
    12
    8
    """
    if root is not None:
        inorder_visit(root.left, act)
        inorder_visit(root.right, act)
        act(root)


def inorder_visit(root, act):
    """
    Visit each node of binary tree rooted at root in order and act.

    @param BinaryTree root: binary tree to visit
    @param (BinaryTree)->object act: function to execute on visit
    @rtype: None

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> def f(node): print(node.data)
    >>> inorder_visit(b, f)
    2
    4
    6
    8
    10
    12
    14
    """
    if root is not None:
        inorder_visit(root.left, act)
        act(root)
        inorder_visit(root.right, act)

def visit_level(t, n, act):
    """
    Visit each node of BinaryTree t at level n and act on it.  Return
    the number of nodes visited visited.

    @param BinaryTree|None t: binary tree to visit
    @param int n: level to visit
    @param (BinaryTree)->Any act: function to execute on nodes at level n
    @rtype: int

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> def f(node): print(node.data)
    >>> visit_level(b, 2, f)
    2
    6
    10
    14
    4
    """
    if t is None:
        return 0
    elif n == 0:
        act(t)
        return 1
    elif n < 0:
        return (visit_level(t.left, n-1, act) + visiv_level(t.right, n-1, act))
    else:
        return 0


def levelorder_visit(t, act):
    """
    Visit BinaryTree t in level order and act on each node.

    @param BinaryTree|None t: binary tree to visit
    @param (BinaryTree)->Any act: function to use during visit
    @rtype: None

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> def f(node): print(node.data)
    >>> levelorder_visit(b, f)
    8
    4
    12
    2
    6
    10
    14
    """
    # this approach uses iterative deepening
    visited, n = visit_level(t, 0, act), 0
    while visited > 0:
        n += 1
        visited = visit_level(t, n, act)


# assume binary search tree order property
def bst_contains(node, value):
    """
    Return whether tree rooted at node contains value.

    Assume node is the root of a Binary Search Tree

    @param BinaryTree|None node: node of a Binary Search Tree
    @param object value: value to search for
    @rtype: bool

    >>> bst_contains(None, 5)
    False
    >>> bst_contains(BinaryTree(7, BinaryTree(5), BinaryTree(9)), 5)
    True
    """
    if node is None:
        return False
    elif value < node.data:
        return bst_contains(node.left, value)
    elif value > node.data:
        return bst_contains(node.right, value)
    else:
        return True


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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
