TUTORIAL 12: Decision trees and lower bounds
============================================

This week the tutorial is about *problem*
(not algo...) complexity and the Decision Tree Model.

In class, I illustrated this by explaining:

(a) how a comparison-based sorting algorithm can be modelled
with the decision tree model, and 
(b) how to use this fact to derive the Omega(n log n)-comparisons
lower bound for the sorting problem (for comparison-based
sorting algorithms).

You will:
A) illustrate the use of the decision tree model
to derive lower bounds for other problems than sorting
(for comparison-based algos)

B) show that sometimes the decision tree technique to
determine a lower bound is not powerful enough to give
you a *good* lower bound by looking at search on an
*unsorted* array of n distinct elements.


Decision tree example: 
---------------------
Draw the decision tree for binary search in a sorted array of size 5.
(Assume that we search for some arbitrary key "key".)

The algorithm for binary search is:

low = 1
high = size(A)

repeat
  mid = floor((high-low)/2) + low
  
  if A[mid] = key 
     return mid
  else if A[mid] < key
     low = mid + 1
  else if A[mid] > key
     high = mid -1

until (high < low)

return "not found"

The decision tree is as follows: 
(Let the students figure it out.) SPEND A LOT OF TIME ON THIS EXAMPLE!

(It's hard to draw the tree in text, so here's some explanation of the
tree:
-rectangular boxes should be around comparisons (i.e., key:A[i])
-the high, low values are outside the nodes, and are only included to make
 computing the next level of the tree easier
-each edge in the tree has one of the following labels on it: key=A[i], 
 key<A[i] or key>A[i]  
-return values should have ovals around them    
)

                       key:A[3]  
                    (high=5;low=1)
           /              |              \
key=A[3]  /      key<A[3] |               \ key>A[3] 
         /                |                \
  return 3            key:A[1]             key:A[4]
                   (high=2;low=1)        (high=5;low=4)
               /          |     \        /      |            \
    key=A[1]  /  key<A[1] | key>A[1]  key=A[4]  | key<A[4]    \ key>A[4]
             /            |     |       |       |              \
      return 1       return    key:A[2]  return 4  return        key:A[5]
                  "not found" (high=2;             "not found"   (high=5; 
                                low=2)                           low 4)
                         /         |    \                 /      |     \
               key=A[2] / key<A[2] | key>A[2]   key=A[5] /  key<A[5]   key>A[5]    


Things to notice:
1) some leaves are duplicates (i.e., return "not found")
2) there are n+1 unique leaves  (we can return each of the 5 keys, and
   "not found" if the value is not in the array 
3) Each non-leaf node has three children

Lower bound for comparison-based search algorithms
-------------------------------------------------
For ANY comparison-based search algorithm A, we can prove a lower bound of
\Omega(log n) as follows:

the number of distinct outcomes is n+1 in the worst-case (i.e., when
   all values in the array are unique)
-> the  corresponding decision tree T_A has at least n+1 leafs

there are at most three outcomes at any tree level (=, <, >).
-> decision tree has height at least log_3 n
(since a ternary tree of height h has at most 3^h leaves)

Since log_3 n \in \Omega(log n), we have a lower bound of \Omega(log n).

Aside #1:
To show log_3 n \in \Omega(log n):

let k= log_3 n         <- take this to the power of 3
   3^k = n             <- take the log_2 of each side
   k log_2 3 = log_2 n <- substitute k= log_3 n and divide by log_2 3
   log_3 n   = log_2 n / log_2 3 
-> log_3 n \in \Theta (log_2 n)

Aside #2: For comparison-based algorithms, we can have the following 
outcomes (or labels on the branches):
(<= , >) or (<, =>) or (<=, =>) or (<, =, >) or (=, \not =)
In each case, there are at most 3 possible outcomes


Lower bound for searching on an unsorted array
----------------------------------------------

Sometimes the decision tree technique to determine a lower bound is not
powerful enough to give a good lower bound:

In particular, consider any algorithm for searching on an unsorted array.

The decision tree for an unsorted array has 
-n+1 distinct leafs
-3 outcomes at each level (<, =, >) [not just (=, \neq) because we want our
  bound to apply to algorithms that compare values of elements too]
-> the tree has height log_3 n
-> we have a lower bound of \Omega(log n)

BUT this lower bound is too low!
Search on an unsorted array actually has a lower bound of \Omega(n)
(Intuitively, if we only look at n-1 elements, we can't be sure if an
element is not in the array or if it is the element we haven't looked at.
This is a simple "adversary" argument.)
