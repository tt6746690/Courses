"""
BinaryTree class and associated functions.
"""
from csc148_queue import Queue


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


def contains(node, value):
    """
    Return whether tree rooted at node contains value.

    @param BinaryTree|None node: binary tree to search for value
    @param object value: value to search for
    @rtype: bool

    >>> contains(None, 5)
    False
    >>> contains(BinaryTree(5, BinaryTree(7), BinaryTree(9)), 7)
    True
    """
    # handling the None case will be trickier for a method
    if node is None:
        return False
    else:
        return (node.data == value or
                contains(node.left, value) or
                contains(node.right, value))


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
    if b.left is None and b.right is None:
        return b.data
    else:
        return eval(str(evaluate(b.left)) +
                    str(b.data) +
                    str(evaluate(b.right)))


def postorder_visit(t, act):
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
    if t is None:
        pass
    else:
        postorder_visit(t.left, act)
        postorder_visit(t.right, act)
        act(t)


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
    if root is None:
        pass
    else:
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
    if t is None:       # base case: when t is not a node
        return 0
    elif n == 0:        # base case: act on current node if reached level n
        act(t)
        return 1
    elif n > 0:
        return (visit_level(t.left, n-1, act) +
                visit_level(t.right, n-1, act))
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


def levelorder_visit2(t, act):
    """
    Visit BinaryTree t in level order and act on nodes as they are visited

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
    >>> levelorder_visit2(b, f)
    8
    4
    12
    2
    6
    10
    14
    """
    nodes = Queue()
    nodes.add(t)
    while not nodes.is_empty():
        next_node = nodes.remove()
        act(next_node)
        if next_node.left:
            nodes.add(next_node.left)
        if next_node.right:
            nodes.add(next_node.right)


# the following functions assume a binary search tree
def insert(node, data):
    """
    Insert data in BST rooted at node if necessary, and return new root.

    Assume node is the root of a Binary Search Tree.

    @param BinaryTree node: root of a binary search tree.
    @param object data: data to insert into BST, if necessary.

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> print(b)
            14
        12
            10
    8
            6
        4
            2
    <BLANKLINE>
    """
    return_node = node
    if not node:            # base case: if empty tree or bottom of tree
        return_node = BinaryTree(data)      # insertion will always happen at the bottom of the tree
    elif data < node.data:
        node.left = insert(node.left, data)
    elif data > node.data:
        node.right = insert(node.right, data)
    else:  # nothing to do
        pass
    return return_node


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
    else:       # where value = node.data
        return True

def find_max(node):
    """
    Find and return subnode with maximum data.

    Assume node is the root of a binary search tree.

    @param BinaryTree node: binary tree node to begin search from
    @rtype: BinaryTree

    >>> find_max(BinaryTree(5, BinaryTree(3), BinaryTree(7)))
    BinaryTree(7, None, None)
    """
    return find_max(node.right) if node.right else node

def delete(node, data):
    """
    Delete data from binary search tree rooted at node, if it exists,
    and return root of resulting tree.

    @param BinaryTree|None node: tree to delete data from
    @param object data: data to delete
    @rtype: BinaryTree|None

    >>> b = BinaryTree(8)
    >>> b = insert(b, 4)
    >>> b = insert(b, 2)
    >>> b = insert(b, 6)
    >>> b = insert(b, 12)
    >>> b = insert(b, 14)
    >>> b = insert(b, 10)
    >>> b = delete(b, 12)
    >>> print(b)
            14
        10
    8
            6
        4
            2
    <BLANKLINE>
    >>> b = delete(b, 14)
    >>> print(b)
        10
    8
            6
        4
            2
    <BLANKLINE>
    """
    return_node = node
    # Algorithm for delete:
    # 1. If this node is None, return that
    if node is None:
        pass
    # 2. If data is less than node.data, delete it from left child and
    #     return this node
    elif data < node.data:
        return_node = delete(node.left, data)
    # 3. If data is more than node.data, delete it from right child
    #     and return this node
    elif data > node.data:
        return_node = delete(node.right, data)
    # 4. If node with data has fewer than two children,
    #     and you know one is None, return the other one
    elif node.left is None:
        return_node = node.right
    elif node.right is None:
        return_node = node.left
    # 5. If node with data has two non-None children,
    #     replace data with that of its largest child in the left subtree,
    #     and delete that child, and return this node
    else:
        max_node = find_max(node.left)
        node.data = max_node.data
        node.left = delete(node.left, max_node.data)
    return return_node



def bst_delete(self):

    parent = None       # locate the node that contain the element and its parent
    current = root

    while current is not None and current.data != data:
        if data < current.data:
            parent = current
            current = current.left
        elif data > current.data:
            parent = current
            current = current.right
        else: pass
    if current is None: return False

    if current.left is None: # case 1
        if parent is None:
            current = current.right
        else:
            if data < parent.data:
                parent.left = current.right
            else:
                parent.right = current.right
    else:
        parent_of_right_most = current
        right_most = current.left
        while right_most.right is not None:
            parent_of_right_most = right_most
            right_most = right_most.right
        current.element = right_most.element

        if parent_of_right_most.right = right_most:
            parent_of_right_most.right = right_most.left
        else:
            parent_of_right_most.left = right_most.left
        return True

if __name__ == "__main__":
    import doctest
    doctest.testmod()
