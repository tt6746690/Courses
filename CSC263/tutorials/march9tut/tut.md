midterm ...
#question 1 

```
nothing(A)
    A[i] = 1
    n = A.size()
    for i = 1 to n do
        for j = 1 to n-1 do
            if A[j] != -A[j + 1]: return
            if i + j > n - 1: return
    return
```

upper bound is `O(n)` because when `i = 1` and `j = n -1`, so then `i + j = n > n - 1` function returns 
lower bound give a bad input say `A = [1, -1, ...]` alternating between 1 and -1


#question 5 

Algo process a sequence of distinct integer keys as input to algo one at a time 
    1. process: process input 
    2. print: returns the smallest key among all keys before print 
    3. median: return the median of the `m` smallest keys

AVL-tree: 
    each node contains the size of subtree

process
1. insert the first `m` key to the tree 
2. when insert any other input, we remove the max element in the tree
AVL-insert delete `O(log m)` so together `O(log m)`

print
1. returns in-order traversal 
`O(m)`

median
1. Search((m+1)/2) by using size of subtree..
`O(log m)`




# Amortized complexity 

Consider a data structure representing a set I of integers 
store all integers in linked list of arrays 

1. each element in I occurs exactly once
2. Each array is in increasing order 
3. size of each array is a power of 2 
4. the arrays in linked list are kept in order of increasing size


ex. I = { 3, 5, 1, 17, 10 }

3   ->  1 
        5
        10
        17

Insert 
    1. create an array containing x 
    2. Merge the array to the linked list
        just like binomial heap 


Aggregate analysis 
Let y be the first position of missing array in the sorted linked list of arrays

i-th array      cost        total 
1               1           n / 2
2               2           n / 2
3               4           n / 2 

total cost `O(nlog n)`:
    n operations 

Amortized `O(log n)` 
