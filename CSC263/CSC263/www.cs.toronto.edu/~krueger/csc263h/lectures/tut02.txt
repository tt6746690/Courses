TUTORIAL 2: Heaps
=================

PART A: Maintaining the heap property
-------------------------------------

Heap-order property:
 -root has maximum value key
 -key stored at a non-root is at most the value of its parent

Therefore:
-any path from root to leaf is in nonincreasing order
-left and right sub-trees are unrelated

Example #1:
Does this satisfy the head-order property?

                   20
           5              10
       12    15        8      2
   6     2  9


[Note that this can also be written an array format as:
 20 5 10 12 15 8 2 6 2 9 ]

No, but   12      15    are heaps.
         6  2    9

We will use a function called MaxHeapify to fix the heap.

PART B: MaxHeapify(A,i) procedure
---------------------------------
Overview of the algorithm:
Inputs: Array A (or binary tree)
        index i in array
Precondition: the binary trees rooted at LEFT(i) and RIGHT(i) are heaps
              Note: A[i] may be smaller than its children
Postcondition: The subtree rooted at index i is a heap

MaxHeapify(A,i)
 l = LEFT(i)
 r = RIGHT(i)

 if l \leq heap_size(A) and A[l] > A[i]
    then largest = l
    else largest = i
 if r \leq heap_size(A) and A[r] > A[largest]
    then largest = r
 if largest \neq i
    then swap(A[i],A[largest])
         MaxHeapify(A,largest)

Back to the example: A = [20 5 10 12 15 8 2 6 2 9]
Call MaxHeapify(A,2)

Step #1: Swap A[2] with A[5]

                   20
           15              10
       12    5        8      2
   6     2  9


Step #2: Swap A[5] with A[10]

                   20
           15              10
       12    9        8      2
   6     2  5


What is the worst-case running-time of MaxHeapify?
-MaxHeapify is recursively called **at most** h times, where h is the
 height of the subtree starting at i
-Each call to MaxHeapify takes c steps, for some constant c

Therefore:
for all inputs, **at most** ch steps are needed where h is the
 height of the subtree starting at i

So, the worst case time complexity is O(h) where h is the subtree with
root i OR O(log n) since h \leq log n = the height of the entire tree

PART C: Build-Max-Heap
----------------------
Question: How do we build a heap in linear time?

Idea:
FIRST create a binary tree (stick each element into a node of the tree)
OR put all the elements in an array

THEN use MaxHeapify on non-leaf nodes

BUILD-MAX-HEAP
 heap-size = length(A)
 for i = \lfloor lenght(A)/2 \rfloor downto 1
     MaxHeapify(A,i)


Example #2:
Apply BUILD-MAX-HEAP to the binary tree below:

                  10
           5             2
       6     12        15     1

Note: Since lenght(A) = 7
\lfloor lenght(A)/2 \rfloor = \lfloor 7/2 \rfloor = 3

Step 1: MaxHeapify(A,3)

                  10
           5            15
       6     12        2     1

Step 2: MaxHeapify(A,2)

                  10
           12            15
       6     5        2     1

Step 3: MaxHeapify(A,1)

                  15
           12            10
       6     5        2     1

Show that the worst-case running time is O(n):
---------------------------------------------
First try: -MaxHeapify has a worst-case running time of O(log n)
           -there are at most n calls to MaxHeapify
So, the worst-case time complexity is O(n log n)

Better answer:
the running time needed at each level of the tree
= worst-case running time of a node at level i * # nodes at level i

Aside: there are **at most** \lceil n / 2^(h+1) \rceil nodes at height h
      in the tree

                  15            height = 2
           12            10     height = 1
       6     5        2     1   height = 0

worst-case running time of a node at height h * # nodes at height h
= ch * \lceil n / 2^(h+1) \rceil

Sum this over all the nodes in the tree:

Sum_{h=0}^{height of the tree}  ch * \lceil n / 2^(h+1) \rceil
\leq Sum_{h=0}^{\lfloor log n \rfloor} ch * \lceil n / (2*2^h) \rceil
\leq cn Sum_{h=0}^{\infinity} h/(2^h)
=c/2 n (1/2 + 2/4 + 3/8 + ...)
\leq c/2 n d, for some contant d  (see textbook page 135 for details)

Therefore, for all inputs of size n, the time needed is at most cd/2 n

So, the worst case time complexity in O(n)


Show that the worst-case running time is Omega(n):
--------------------------------------------------
 Pick any input of size n:
  The loop goes from length(A/2) downto 1.
  Since there are **at least** n/2 iterations of the loop, and each such
  iteration takes **at least** a constant amount of time
  (bcs each execution of MAX-HEAPIFY takes at least a constant amount of time)

So the worst-case time complexity of the algorithm is Omega(n),

=> The worst-case complexity of BUILD-MAX-HEAP is Theta(n).

Part D: Heapsort
---------------
We can use max-heaps to sort an array A.

Idea:
-put elements into a max-heap
-element A[1] is the maximum element
-exchange A[1] with the last element in the array
-decrease the size of the heap by one
-maintain the heap order property by calling Max-Heapify

Here's the code:

HEAPSORT(A)
 BUILD-MAX-HEAP(A)
 for i = length[A] downto 2
    do exchange A[1] with A[i]
       heap-size[A] = heap-size[A] - 1
       MAX-HEAPIFY(A, 1)

 BUILD-MAX-HEAP(A) takes O(n) time,
 MAX-HEAPIFY(A, 1) takes O(log n) time,
 and the for loop executes n-1 times

Therefore heapsort takes O(n log n) time in the worst-case.

