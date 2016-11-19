# Recursive Delete
`bst_del_rec`<br>Defined as deleting a node (if exists) from the BST and return the resulting BST.<br>Example

```
+  `t = bst_del_rec(t, 10)` deletes 10 from BST t and returns the reference to the tree
```

_base case_ if the tree is none return none  

```python
if not tree:
    return None
```

_recursive case_ if data is less than the tree data, delete it from left child

```python
if data < tree.data:
    tree.left = bst_del_rec(tree.left, data)
```

```python
if data > tree.data:
    tree.right = bst_del_rec(tree.right, data)
```

`bst_del_rec(tree, data)`<br>what does it mean if none of the above if have been true?  
- we have located the tree node to be deleted  
- What next?  
- case 1: if the tree node does not have a left child  
- case 2: if the tree node does not have a left child
  - find the largest node of the left child
  - replace the tree node data with the largest just found
  - delete the largest

```python
if tree.left is not None:
    largest = findmax(tree.left)
    tree.data = largest.data
    tree.left = bst_del_rec(tree.left, largest.data)    
    return largest.left
```

```python
def bst_del_rec(tree, data):
    if not tree:                # base cases
        return None
    elif data < tree.data:      # recursive cases
        tree.left = bst_del_rec(tree.left, data)
    elif data > tree.data:
        tree.right = bst_del_rec(tree.right, data)
    elif tree.left is None:    # left child is empty
        return tree.right
    else:   # left child is not empy
        largest = findmax(tree.left)
        tree.data = largest.data
        tree.left = bst_del_rec(tree.left, largest.data)    
        return tree

    def findmax(tree):
        return tree if not tree.right else findmax(tree.right)
```

# Efficiency of algorithms
BST: iterative delete vs. recursive deletes?
- extra memory?   
  - constant vs. in order of height of tree   
  - O(1) vs. O(lg n) if balanced or O(n) otherwise (since every recursive call requires additional memory)  
  - similar complexity but iterative runs faster

- Time?
  - Although both in order of height of tree, the latter requires more work

**Fibonacci: iteration vs. recursion**  
- extra memory?  
  - iterative: O(1)   >> no matter how big is n, fixed amount of memory required   
  - recursive: O(n)   >> memory required depends on the size of input  

- Time?   
  - iterative: O(n)  
  - recursive: O(2^n)

```python
def fib_rec(n):
    if n == 0 or 1:
        return n
    else return fib_rec(n-1) + fib_rec(n-2)

def fib_iter(n):
    a, b = 0, 1
    for i in range(0, n):
        a, b = b, a + b
    return a
```

**Recursive vs. iterative**
- recursive functions impose a loop
- the loop is implicit and the complier / interpreter

> Every recursive function can be written iteratively (by explicit loops) and may require staks too. yet the when the nature of a problem is recursive, writing it iteratively can be time consuming and less readable.

- recursion is a very powerful technique for problems that are naturally recursive
