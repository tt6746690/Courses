# Week 7 - Trees

While lists are a really useful data structure, not all data has a natural linear order. Family trees, corporate hierarchies, classification schemes like "Kingdom, Phylum, etc." and even file storage on computers all follow a _hierarchical structure_, in which an entity is linked to multiple entities "below it", forming a non-linear ordering on the different components.

In computer science, we have use a **tree** data structure to represent this type of data. Trees are a recursive data structure, with the following definition:

*   A tree can be empty.
*   A tree can have a **root value** connected to any number of other trees, called the **subtrees** of the tree.

Intuitively, the root of the tree is the item which is located at the top of the tree; the rest of the tree consists of subtrees which are attached to the root. Note that a tree can contain a root value but not have any subtrees: this would represent a tree that contains just a single item.

## Tree Terminology

A tree is either **empty** or **non-empty**. Every non-empty tree has a **root node** (which is generally drawn at the top), connected to zero or more **subtrees**. The root node of the above tree is labelled A. The **size** of a tree is the number of nodes in the tree. _What's the relationship between the size of a tree and the size of its subtrees?_

A **leaf** is a node with no subtrees. The leaves of the above tree are labelled E, F, G, I, and J. _What's the relationship between the number of leaves of a tree and the number of leaves of its subtrees?_

The **height** of a tree is the length of the _longest_ path from its root to one of its leaves, _counting the number of items on the path_. The height of the above tree is 4\. _What's the relationship between the height of a tree and the heights of its subtrees?_

The **children** of a node are all nodes directly connected underneath that node. The children of node A are nodes B, C, and D. Note that the number of children of a node is equal to the number of subtrees of a node, but that these two concepts are quite different. The **descendants** of a node are itself, its children, the children of its children, etc. _What's the relationship between the descendants of a node and the descendants of its children?_

Similarly, the **parent** of a node is the one immediately above and connected to it; each node has one parent, except the root, which has no parent. The **ancestors** of a node are itself, its parent, the parent of its parent, etc.

**Note**: sometimes, it will be convenient to say that descendants/ancestors of a node do onot include the node itself; we'll make it explicit whether to include the node or not when it comes up. Note that a node is **not** a child of itself, nor a parent of itself.

## Tree implementation

Here is a simple implementation of a tree in Python.

```python
class Tree:
    """A recursive tree data structure."""
    # === Private Attributes ===
    # @type _root: object | None
    #     The item stored at the tree's root, or None if the tree is empty.
    # @type _subtrees: list[Tree]
    #     A list of all subtrees of the tree

    # === Representation Invariants ===
    # - If _root is None then _subtrees is empty. This setting of attributes
    #   represents an empty Tree.
    # - (NEW) self._subtrees doesn't contain any empty trees

    def __init__(self, root):
        """Initialize a new Tree with the given root value.

        If <root> is None, the tree is empty.
        A new tree always has no subtrees.

        @type self: Tree
        @type root: object | None
        @rtype: None
        """
        self._root = root
        self._subtrees = []

    def is_empty(self):
        """Return True if this tree is empty.

        @type self: Tree
        @rtype: bool
        """
        return self._root is None

    def add_subtrees(self, new_trees):
        """Add the trees in <new_trees> as subtrees of this tree.

        Precondition: this tree is not empty
        Precondition: no tree in <new_trees> is already in self._subtrees

        @type self: Tree
        @type new_trees: list[Tree]
        @rtype: None
        """
        self._subtrees.extend(new_trees)
        # or, self._subtrees = self._subtrees + new_trees

```


Our constructor here always creates either an empty tree (when `root is None`), or a tree with just an item at the root and no children. These two cases are generally the base cases when dealing with trees. When you write your recursive functions, your base cases will generally be one or both of these types of trees.

For our convenience in building trees, we've also included an `add_subtrees` method that adds trees to the list of children, which you can find in the source code.

## Recursion on Trees

There's a reason I keep asking the same question: understanding the relationship between a tree and its subtrees is precisely understanding the recursive structure of the trees. Understand this and you'll be able to write extremely simple and elegant code for processing trees.

Here's a quick example: the size of a non-empty tree is the sum of the sizes of its subtrees, plus 1 for the root; the size of an empty tree is, of course, 0\. This single observation immediately lets us write the following recursive function for computing the size of a tree:

```python
def __len__(self):
    if self.is_empty():
        return 0
    else:
        size = 0
        for subtree in self._subtrees:
            size += subtree.__len__()
        return size + 1
```


We can generalize this nicely to a template for recursive methods on trees:

```python
def f(self):
    if self.is_empty():
        ...
    else:
        ...
        for subtree in self._subtrees:
            ... subtree.f() ...
        ...
```


## Traversing a Tree

**Note: not covered in lecture, but useful.**

Because lists have a natural order of their elements, they're pretty straightforward to traverse, meaning (among other things) that it's easy to print out all of the elements. How might we accomplish the same with a tree?

Here's an idea: print out the root, then recursively print out all of the subtrees. That's pretty easy to implement. Note that in our implementation, the base case is when the tree is empty, and in this case the method does nothing.

```python
def print_tree(self):
    if not self.is_empty():
        print(self._root)
        for subtree in self._subtrees:
            subtree.print_tree()


>>> t1 = Tree(1)
>>> t2 = Tree(2)
>>> t3 = Tree(3)
>>> t4 = Tree(4)
>>> t4.add_subtrees([t1, t2, t3])
>>> t5 = Tree(5)
>>> t5.add_subtrees([t4])
>>> t5.print_tree()
5
4
1
2
3

```



We know that 5 is the root of the tree, but it's ambiguous how many children it has. Let's try to use _indentation_ to differentiate between the root node and all of its subtrees. But how do we do this? Looking at the previous code, what we want is for when `subtree.print_tree()` is called, that everything that's printed out in the recursive call is indented.

>But remember, we _aren't_ tracing the code, and in particular, we can't "reach in" to make the recursive calls act differently. However, this problem - we want some context from where a method is called to influence what happens inside the method - is **exactly** the problem that parameters are meant to solve.

So we'll pass in an extra parameter for the _depth_ of the current tree, which will be reflected by adding a corresponding number of whitespace characters to printing.

```python
def print_tree_indent(self, depth=0):
    if not self.is_empty():
        print(depth * '  ' + str(self._root))
        for subtree in self._subtrees:
            subtree.print_tree_indent(depth + 1)

```

Now we can implement `print_tree` simply by making a call to `print_tree_indent`:

```python
def print_tree(self):
    self.print_tree_indent()
```


### Side Note: Optional Parameters

In `print_tree_indent`, we used an _optional parameter_ that could either be included or not included when the function is called.

So we can call `t.print_tree_indent(5)`, which sets its `depth` parameter to `5`, as we would expect. However, we can also call the method as `t.print_tree_indent()`, in which case it sets the `depth` parameter to `0`.

Optional parameters are a powerful Python feature that allows us to write more flexible functions and methods to be used in a variety of situations. Two important points to keep in mind, though:

*   All optional parameters must appear after all of the required parameters in the function signature
*   Do not use mutable values like lists for your optional parameters; instead, use immutable values like ints, strings, and `None`


[__MUTATING TREES__](http://www.cs.toronto.edu/~david/csc148/content/trees/mutating_trees.html)  
[__BINARY SEARCH TREE__](http://www.cs.toronto.edu/~david/csc148/content/trees/bst.html)
