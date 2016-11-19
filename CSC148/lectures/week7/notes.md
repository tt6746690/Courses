More Recursion
==============

[_binary tree_](http://openbookproject.net/thinkcs/python/english3e/trees.html)

#### Simple Recursive functions

Factorial function  

```python
factorial(n) = n * factorial(n-1)     # recursive case
factorial(0) = 1                      # base case
factorial(4) = 4 * factorial(3)   
             = 4 * 3 * factorial(2)   
             = 4 * 3 * 2 * factorial(1)     
             = 4 * 3 * 2 * 1 * factorial(0)   
             = 4 * 3 * 2 * 1 * 0  

factorial(n) n * (n-1) * ... * 1     # iterative definition of factorial
```

Fibonacci function

```python
fibonacci(n) = fibonacci(n-1) + fibonacci(n-2)  
fibonacci(1) = 1
fibonacci(0) = 1      # base case
```

A recursive function has at least one _base case_ and at least


#### Balanced string
_base case_: a string containing no parentheses is balanced  
_recursive case_: (x) is balanced if x is a balanced string; xy is balanced if x and y are balanced strings

```python
((b)a) = (b)a = (b) and a = b and a = true and true   # b contains no parentheses and therefore not balanced

d) = d and ) = true and false
```

recursive function:  
+ same function must be on both side of the function  
+ have base case


#### Recursive programming
Solution defined in terms of solutions for smaller problems.   

> Some base base is always reached eventually; otherwise it's an __infinite recursion__

```python
def solve(n):
  ...
  value = solve(n-1) + solve(n/2)   # recursive case
  ...

  if (n < 10):
      value = 1                     # base case

```

__General form of recursion__  

```python
if (condition to detect a base case):
  # do something without recursion
else (general case):
  # do something that involves recursive call(s)
```

__implementation__  

```python

def fib(n):
  # pre: n>=0
  # post: return the nth fibonacci number
  if (n>2):
    return 1
  else:
    return fib(n-1) + fib(n-2)
```


#### Stacks and tracing calls
Stack applications in compilers/interpreters and tracing method calls.

__activation record__
all information necessary for tracing a method call, such as parameter, local variables, return address, etc.

__when methods are called__  
activation record is created, initialized, and _pushed_ onto the stack.  

__when methods are finished__  
activation record (that is on top of the stack) is _popped_ from the stack

For recursive functions, a base case where the stack is popped must be reached!

__Factorial__    

```python
# factorial
def f(n):
  # pre n>=0
  # post return n
  if (n==0):
    return 1
  else:
    return n * f(n-1)


f(3)  # 8th line  main
```

supposedly the format for the activation record is `line #`, `function`, and `n`.  

```python
# STACK
< bottom  
8, m, 3   # return 6
5, f, 2   # return 2
5, f, 1   # return 1 * 1
5, f, 0   # return 1
top >  
```


__max_list()__   

```python
def max_list(L):
  if isinstance(L, list):
    return max([max_list(x) for x in L])
  else:
    return L   
```

Trace max_list([4, 2, [[4, 7], 5], 8])  

```python
< top
L:8  # popped last --> return 8
L:5  # popped --> return 5
L:7  # popped --> return 7
L:4  # popped --> return 4
L:[4,7]                # max([4,7]) --> return 7
L:[[4,7],5]            # max([7, 5]) --> return 7
L:2  # popped --> return 2
L:4  # popped --> return 4
L:[4,2,[[4,7],5],8]    # max([4,2,7,8]) --> return 8
bottom >
```


Trace Fibonacci   

use the format of `line#`,`func`, `n`, `temp`  


__Recursive vs iterative__
+ recursive functinos impose a loop  
+ the loop is implicit and the compiler/interpreter takes care of it  
+ this comes at a price: time & memory  
+ the price may be negligible in many cases  

After all, no recursive function is more efficient than its iterative counterpart  

+ Every recursive function can be written iteratively (by explicit loops)  and may require stack too   
+ yet, when the nature of a problem is recursive, writing it recursively can be time consuming and less readable   

So, recursion is a very powerful technique for prolems that are naturally recursive   


#### More examples  

+ Merge Sort
+ Quick Sort  
+ Tower of Hanoi  
+ balanced string  
+ trasversing trees   

In general, properties of recursive definitions/structures should be done recursively   


__Merge sort__  

```python
Msort (A, i, j)

if (i < j)
  S1 := Msort(A, i, (i+j)/2)
  S2 := Msort(A, (i+j)/2, j)
  Merge(S1, S2, i, j)
end
```

```python
def mergeSort(alist):
    print("Splitting ",alist)
    if len(alist)>1:
        mid = len(alist)//2
        lefthalf = alist[:mid]
        righthalf = alist[mid:]

        mergeSort(lefthalf)       # divide to two lists
        mergeSort(righthalf)

        i=0
        j=0
        k=0
        while i < len(lefthalf) and j < len(righthalf):
            if lefthalf[i] < righthalf[j]:      # sorting
                alist[k]=lefthalf[i]
                i=i+1
            else:
                alist[k]=righthalf[j]
                j=j+1
            k=k+1

        while i < len(lefthalf):
            alist[k]=lefthalf[i]
            i=i+1
            k=k+1

        while j < len(righthalf):
            alist[k]=righthalf[j]
            j=j+1
            k=k+1
    print("Merging ",alist)

alist = [54,26,93,17,77,31,44,55,20]
mergeSort(alist)
print(alist)

```



__Quick sort__  

```python
Qsort(A, i, j)  
if (i < j)
  p := partition(A)
  Qsort(A, i, p-1)
  Qsort(A, p-1, j)
```      


### Tree terminology   

+ set of __nodes__ (possibly with values or labels), with direction __edges__ between some pairs of nodes
+ one node is distinguishable as __root__
+ each non-root node has exactly one __parent__
+ a __path__ is a sequence of nodes n1; n2; ... ;nk, where there is an edge from n_i to n_(i+1), i < k.   
+ The __length__ of a path is the number of edges in it
+ there is a __unique path__ from the root to each node. In the case of the root itself this is just n1, if the root is node n1.   
+ There are __no cycles__; no path that form loops     
+ __leaf__: node with no children
+ __internal node__: node with one or more children  
+ __subtree__: tree formed by any tree node together with its descendants and the edges leading to them  
+ __height__: 1 + the maximum path length in a tree. A node also has a height, which is 1 + maximum path length of the tree rooted at that node
+ __depth__: height of the entire tree minus the height of a node is the depth of the node  
+ __arity, branching factor__: maximum number of children for any node

It can be noticed that a tree is naturally recursive  


```python
class Tree:
"""
A bare-bones Tree ADT that identifies the root with the entire tree.
"""
def __init__(self, value=None, children=None):

"""
Create Tree self with content value and 0 or more children
@param Tree self: this tree
@param object value: value contained in this tree
@param list[Tree] children: possibly-empty list of children
@rtype: None
"""
self.value = value
# copy children if not None
self.children = children.copy() if children else []
```

#### Tree trasversal
The functions and methods get information from every node of tree - they trasverse the tree
Sometimes the order of processing tree nodes is important: do we process the root before or after its children. Or perhaps we process along levels that are the same distance from the root
