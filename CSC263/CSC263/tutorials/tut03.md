TUTORIAL 3: Deletion operation for BSTs and AVL trees
=====================================================

You will need the AVL handout for the tutorial this week.
It's on the course web page under lecture notes.
(AVL trees are not in the textbook.)

DELETING FROM AN UNBALANCED BINARY SEARCH TREE:
-----------------------------------------------

Recall that we use the following algorithm to delete a node from a BST:

Find the node x that contains the key k:
1) If x has no children, delete x.
2) If x has one child, delete x and link x's parent to x's child
3) If x has two children,
   -find x's successor z [the leftmost node in the rightsubtree of x]
   -replace x's contents with z's contents, and
   -delete z.
   (Note: z does not have a left child, but may have a right child)
   [since z has at most one child, so we use case (1) or (2) to delete z]

Code to find x's successor:
     z = RIGHT_CHILD(x)
     while LEFT_CHILD(z) \neq NULL
         z = LEFT_CHILD(z)


Example: Delete(T,10)

          15
     5          18
  2    10
     8     13
    6    11   14
          12

Find z:
              15
     5               18
  2    x=10
     8       13
    6    z=11   14
            12

Replace contents:
              15
     5               18
  2    x=11
     8       13
    6    z=11   14
          12

Delete z:
              15
     5               18
  2    x=11
     8       13
    6    12    14

Recall that the worst-case running time of DELETE is \Theta(n)

DELETING IN AVL TREES:
----------------------

Delete(T,k) means delete a node with the key k from the AVL tree T
[Aside: Delete(T,x) is safer, and needed if we allow keys to appear multiple
times. This way is a little simpler.]

I) First: find the node x where k is stored

II) Second: delete the contents of node x

Claim: Deleting a node in an AVL tree can be reduced to deleting a leaf

There are three possible cases (just like for BSTs):
1) If x has no children (i.e., is a leaf), delete x.
2) If x has one child, let x' be the child of x.
    Notice: x' cannot have a child, since subtrees of T can differ in
    height by at most one
    -replace the contents of x with the contents of x'
    -delete x' (a leaf)
3) If x has two children,
   -find x's successor z (which has no left child)
   -replace x's contents with z's contents, and
   -delete z.
   [since z has at most one child, so we use case (1) or (2) to delete z]

    In all 3 cases, we end up removing a leaf.

III) Third: Go from the deleted leaf towards the
      root and at each ancestor of that leaf:
      -update the balance factor
      -rebalance with rotations if necessary.

Example #1: Single rotation
---------------------------

AVL tree T:
                  13 (0)
               /         \
         9 (0)            15(+)
        /    \            /    \
     6 (-)  12 (-)     14 (0)   20 (0)
     /      /                   /  \
    2 (0)  11(0)            18 (0)  30 (0)

Delete(T, 13):

Before rotation, but after updating balance factors:
                  14 (0)
               /         \
         9 (0)            15(++)
        /    \            /    \
     6 (-)  12 (-)       X   20 (0)
     /      /                   /  \
    2 (0)  11(0)            18 (0)  30 (0)

After rotation:

                  14 (0)
               /         \
         9 (0)            20 (-)
        /    \          /     \
     6 (-)  12 (-)     15 (+)  30 (0)
     /      /            \    
    2 (0)  11(0)          18 (0)

Example #2: Double rotation
---------------------------

AVL tree T:
                  9 (+)
               /         \
         6 (-)              15(-)
        /    \            /      \
     2 (-)    7 (0)     13 (-)     20 (0)
     /                 /   \        /     \
    1 (0)           12 (-) 14 (0) 18 (0)  30 (0)
                      /
                    11 (0)

Delete(T, 2):

Before rotation, but after updating balance factors:
                  9 (++)
               /         \
         6 (0)              15(-)
        /    \            /      \
     1 (0)    7 (0)     13 (-)     20 (0)
      /                 /   \        /     \
     X              12 (-) 14 (0) 18 (0)  30 (0)
                      /
                    11 (0)

After first rotation:
                  9 (++)
               /         \
         6 (0)              13(+)
        /    \            /      \
     1 (0)    7 (0)     12 (-)     15 (+)
                        /          /      \
                    11 (0)       14 (0)  20 (0)
                                         /     \
                                      18 (0)  30 (0)

After second rotation:
                  13 (0)
               /         \
         9 (0)              15(+)
        /       \           /     \       
      6 (0)      12 (-)    14 (0)  20 (0)
     /    \        /               /     \
  1 (0)    7 (0)  11 (0)          18 (0)  30 (0)
