

##### Review

Efficiency of iterative algorithms  
+ time complexity
+ calculate/estimate a function denoting the number of operations (comparisons), and we focus on the dominant term. We mean by discarding irrelevant coefficients as well as all non-dominant terms
+ We focus on loops  
  + the way the loop invariant is changed
  + if the loops are nested or sequential
+ We also watch _function calls_   


#### example 1

```python
def bst_contains(node, value):
  if node is None:
    return False
  elif value < node.data:
    return bst_contains(node.left, value)
  elif value > node.data:
    return bst_contains(node.right, value)
  else:
    return True
```

Assume `T(n)` as the number of operations for a tree with `n` nodes. Assume we have balanced tree. Then `T(n) = T(n/2) + epsilon` where `epsilon` is a constant time. NOTE for a general BST, the worst case scenario has `T(n) = T(n-1)`

#### Example 2  

```python
Qsort(A, i, j):
if (i<j):
  p := partition(A)
  Qsort(A, i, p-1)
  Qsort(A, p+1, j)
```
denote T(n) as the number of operations in `Qsort` for a list with `n` items. Partition requires to traverse the whole list, i.e. `n` iterations.  Assume we have the best partition function: i.e. `p` is roughly at the middle of the list.

`T(n) = n + 2T(n/2) + epsilon`

for example

```python
8 10 2 5 6 3 12 1
# Qsort(A, 0, 7)
# say first item is pivot, put all item less than pivot on left of pivot
# and put all item larger than pivot on right of pivot, return index of pivot
1 3 6 5 2 8 10 12
# calls Qsort(A, 0, 4) ## sort everything before pivot 8
#       Qsort(A, 6, 7) ## sort everything after pivot 8
```


#### Example 3


```python
Msort(A, i, j)

if (i < j):
  S1 := Msort(A, i, (i+j)/2)
  S2 := Msort(A, (i+j)/2, j)
  Merge(S1, S2, i, j)
end
```

Denote `T(n)` as the number operations in `Msort` for a list with `n` items. Merge is to merge two sorted list into one, the result will always have `n` items, hence Merge requires `n` operations.

```python
S1 = 4, 6, 20, 21, 30, 50, 52
S2 = 1, 3, 19, 40, 42

# Merge
S = 1, 3, 4, 6, 19, 20, 21, 30, 40, 42, 50, 52
```

`T(n) = 2T(n) + n + epsilon`


#### More insights to big O

when we say an algorithm f(n) is in O(g(n)), we mean f(n) is bound by g(n). In other words, g(n) is an upper bound for f(n). This means, there are positive constants `c` and `B` such that f(n) <= cg(n) for all n greater than B.


```python
1
n
loglogn
logn
nlogn
n^2
n^2logn
n^3
n^4
2^n
3^n
n^n
n!
```
To find an upper bound, find the tightest bound   


##### Python list and liked lists

A python list is a contiguous structure.
+ _lookup_ is faster
+ _insertion_ and _deletion_ is slow

linked list is not a contiguous data structure
+ _lookup_ is slow
+ _insertion_ and _deletion_ is fast


|  | lookup  |insert| delete|
| :------------- | :------------- |
| lists        | O(1)     |O(n)|O(n)|
| Linked Lists | O(n)|O(1)|O(1)|
|BST| O(logn)| O(logn)|O(logn)|
|Hash Table| O(1)*|O(1)*|O(1)*|

\* assume no collision
(say ADAM or MADA is exactly the same when they are not - __collision__)



#### Recall balanced BST
BST can be implemented by linked lists. Yet it has a property that makes it more efficient when it comes to lookup

##### Hash function

+ input a _key_
+ output is _index_ value in a list

A hash function first converts a key to an integer value, then compresses that value into an index. Just as a simple example:

> the conversion can be done by applying some function to the binary values of the characters of the key

And the compression can be done by some modular operations


#### Example

Say a class roster of up to 10 students.

To insert:
  + we want to enroll 'ANA'
  + hash function:
    + conversion component returns `208 = 65 + 78 + 65 `
    + compression component, for instance, returns `8`, which is `208` modular operation (remainder) `10`  
  + so we insert 'ANA' at index `8` of roster

  + similarly we want to enroll 'ADAM'
  + hash function:
    + conversion returns `208 + 77 = 285`
    + compression component returns `5`  
  + so we insert 'ADAM' at index `5` of roster  

To lookup


__Collision__

How collision happen?  

What can we do when there is a collision?  
+ chaining
  + building a linked list at every index
+ probing
  + each cell of a hash table stores a single key–value pair. When the hash function causes a collision by mapping a new key to a cell of the hash table that is already occupied by another key, linear probing searches the table for the closest following free location and inserts the new key there. Lookups are performed in the same way, by searching the table sequentially starting at the position given by the hash function, until finding a cell with a matching key or an empty cell.
+ double hashing  
  + Like linear probing, it uses one hash value as a starting point and then repeatedly steps forward an interval until the desired value is located, an empty location is reached, or the entire table has been searched; but this interval is decided using a second, independent hash function (hence the name double hashing). Unlike linear probing and quadratic probing, the interval depends on the data, so that even values mapping to the same location have different bucket sequences; this minimizes repeated collisions and the effects of clustering.
