#### Review
Depth-first traversal
+ inorder  
+ preorder  
+ posterorder

Breadth-first traversal
+ level-order  

_Binary Search Trees_       
data in the root is larger than all that is in left and less than all that is in right. Data stored in nodes are comparable


####BST
+ BST with 1 node has height 1
+ BST with 3 node may have height 2
+ BST with 7 nodes may have height 3  
+ BST with 15 nodes may have height 4  
+ BST with n nodes may have height [lgn]     

IF the BST is 'balanced', then we can check whether an element is present in about lgn node accesses. This is significantly faster than searching a list (O(n))      

```python
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
```

```python
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
```

#### bst_delete

```python
# Algorithm for delete:
# 1. If this node is None, return that
# 2. If data is less than node.data, delete it from left child and
# return this node
# 3. If data is more than node.data, delete it from right child
# and return this node
# 4. If node with data has fewer than two children,
# and you know one is None, return the other one
# 5. If node with data has two non-None children,
# replace data with that of its largest child in the left
# subtree and delete that child, and return this node
```

first locate the node that contains the element and also its parent node. Let __current__ point to the node that contains the element in the tree and __parent__ point to the parent of the current node.   

__Case 1__: the `current` node has _no_ `left child`
Simply connect the `parent` with the right child of the `current` node.   

__Case 2__: the `current` node has a `left child`
Let `right_most` point to the node that contains the largest element in the left subtree of the `current` node. And let `parent_of_right_most` point to the parent node of the `right_most` node.

Then, Replace the element value in the `current` node with the one in the `right_most` node. Connect the `parent_of_right_most` node with the `left child` of the `right_most` node.

```python
def bst_delete(self, data):
  parent = None
  current = root

  while current is not None and current.data != data:
      if data < current.data:
          parent = current
          current = current.left
      elif data > current.data:
          parent = current
          current = current.right
      else: pass # Element is in the tree pointed at by current
  if current is None: return False # Element is not in the tree

  # Case 1: current has no left child
  if current.left is None:
    # Connect the parent with the right child of the current node
    # Special case, assume the node being deleted is at root
    if parent is None:
        current = current.right
    else:
      # Identify if parent left or parent right should be connected
        if data < parent.data:  # parent.left is current
            parent.left = current.right
        else:                   # parent.right is current
            parent.right = current.right
  else: # Case 2: The current node has a left child
  # Locate the rightmost node in the left subtree of
  # the current node and also its parent
    parent_of_right_most = current
    right_most = current.left

    while right_most.right is not None:
      parent_of_right_most = right_most
      right_most = right_most.right # Keep going to the right

    # Replace the element in current by the element in rightMost
    current.element = right_most.element

    # Eliminate rightmost node
    if parent_of_right_most.right == right_most:
      parent_of_right_most.right = right_most.left  # right_most.left may be None
    else:
      # Special case: parent_of_right_most == current
      parent_of_right_most.left = right_most.left
    return True # Element deleted successfully
```

bst_delete may as well be implemented by discussing cases based on right instead of left. they are equivalent.
