RECURSION
=========


### Example 1: sum of a list

```python
>>> L1 = [1, 9, 8, 15]
>>> sum(L1)
33

>>> L2 = [[1, 5], [9, 8], [1, 2, 3, 4]]
>>> sum(L2)
error message here        # since the first element in the list is not a number

>>> sum([sum(row) for row in L2])
33

>>> L3 = [[1, 5], 9, [8, [1, 2], 3, 4]]     # complexity increase for higher nesting
```

#### Solution

a function sum_list() that adds all the numbers in a nested list shouldn't ignore built-in sum. except sum wouldn't work properly on nested lists, so make a list comprehension of their sum_list

```python
def sum_list(L):
  ''' (list or in) -> int
  return L if it is an int, or sum of the
  numbers in possibly nested list L

  >>> sum_list(17)
  17
  >>> sum_list([1, 2, 3])
  6
  >>>
  '''
  # reuse: isinstance, sum, sum_list
  if isinstance(L, list):
    return sum([sum_list(x) for x in L)])
  else:  # L is an int
    return L      # base case: the part of the code that does not call onto itself

```

#### Tracing `sum_list()`

to understand recursion, trace from simple to complex  

Trace   
`sum_list(7)`=>    
7   

Trace  
`sum_list([1, 2, 3])` =>  
`sum([sum_list(1), sum_list(2), sum_list(3)])` =>  
`sum(1, 2, 3)` =>  
`6`  

Trace  
`sum_list([1, [2, 3], 4, [2, 3]])` =>   
`sum([sum_list(1), sum_list([2, 3]), sum_list(4), sum_list([2, 3])])` =>   
`sum([1, sum(sum_list(2), sum_list(3)), 4, sum(sum_ilst(2), sum_list(3))])` =>  
`sum([1, sum(2, 3), 4, sum(2, 3)])` =>  
`sum([1, 5, 4, 5])` =>   
`15`  


### Example II: depth of a list  

define the depth of L as follows
if L is a list, 1 plus the maximum depth of L's elements, otherwise 0


```python
>>> L1 = [1, 9, 8, 15]
>>> depth(L1)
1

>>> L2 = [[1, 5], [9, 8], [1, 2, 3, 4]]
>>> depth(L2)
2

>>> L3 = [[1, 5], 9, [8, [1, 2], 3, 4]]
>>> depth(L3)
3
```

#### Solution     

```python

def depth(L):
  '''(list or int) -> int
  return 0 if its empty or an int
  otherwise 1 + max of L's element
  >>> depth(7)
  0
  >>> depth([17])
  1
  >>> depth([1, [2, 3, [4]], 5])
  3
  '''
  #reuse isinstance, max, depth
  if instance(L, list):
    if len(L) == 0:
      return 0  # base case: empty list
    else:
      return 1 + max([depth(x) for x in L])   # recursive case
  else # L is not a list
    return 0    # base case: non-list

```


#### Tracing depth  
Trace in increasing complexity; at each step fill in values for recursive calls that have already been traced

Trace  
`depth([])` =>  
`0`

Trace     
`depth(17)` =>     
`0`

Trace   
`depth([3, 17, 1])` =>   
`1 + max([depth(3), depth(17), depth(1)])` =>   
`1 + max([0, 0, 0])` =>   
`1`


###Recursion
every resursive function have _at least_ one base case  


### Example III: find maximum in nested list
to find the max of non-nested list  ->  `max()`    


#### Solution

```python
def max_list(L):
  if isinstance(L, list):
    return max([max_list(x) for x in L])    # recursive case
  else: # L is in int
    return L      # base case
```

#### Trace

Trace    
`max_list([3, 5, 2, 17, 3])` =>     
`max([max_list(3), max_list(5), max_list(2), max_list(17), max_list(3)])` =>   
`max([3, 5, 2, 17, 3])` =>
`17`

Trace
`max_list([3, [2, 7], 4])` =>  
`max([max_list(3), max_list([2,7]), max_list(4)])` =>
`max([3, max([max_list(2), max_list(7)]), 4])` =>
`max([3, max(2, 7), 4])` =>
`max([3, 7, 4])` =>  
`7`


### Example IV get some turtles to draw  
spawn some turtles, point them in diff
