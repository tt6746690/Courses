Binary Trees, BST
=================

#### Binary trees
Change tree so that we have _two_ named children, _left_ and _right_, and can represent an empty tree with `None`  


#### Arithmetic expression trees
Binary arithmetic expressions can be represented as binary trees  


lower down the BST will have precedents. operator should be at root, each has two children - numbers.


Evaluating a binary expression tree
1. there are no empty expressions
2. if it's a leaf, just return the value
3. otherwise
  + evaluate the left tree
  + evaluate the right tree
  + combine the left and right with the binary operator



  > Inorder: L N R       # A + B  
    Preorder: N L R      # + AB  
    Postorder: L R N     # AB +  
    Left children: L; Right children: R; node: N  

#### Inorder

A recursive definition:
+ visit the left subtree inorder
+ visit the node itself
+ visit the right subtree inorder

The code is almost identical to the definition


```python
# tree

# left of tree
    B

A       
        D
    C
        E
# right of tree

# inorder: BADCE    OR  B|A|(D|C|E)
# preorder: ABCDE   OR  A|B|(C|D|E)
# postorder: BDECA  OR  B(D|C|E)A  having order L R N
```


#### level other
view this node  
view this node's children
view this node's grandchildren
view this node's great grandchildren

let's have a helper function   

```python
def visit_level(tree, level, act):

  2
5

8
  1
3    
  5
0   1 2  # visit levels
```


#### Binary search tree
 add ordering conditinos to a binry tree:
+ data are comparable
+ data in the left subtree are less than node.data
+ data in right subtree are more than node.data


```python
8,5,3,12,4,2,11
       2
     3
       4
   5


8

     11
  12

```


BST: searching can be extremely efficient

BST with 1 one node has height 1
BST with 3 nodes may have height 2
BST with 7 nodes may have height 3
BST with 15 nodes may have height 4
BST with n nodes may have height (log n)  

If the BST is balanced, then we can check
