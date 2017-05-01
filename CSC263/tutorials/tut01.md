TUTORIAL 1: Asymptotic bounds
=============================

Part (A):
---------

QUICK REMINDER OF:

(1) Definition and intuitive meaning of T(n),
       the worst-case time complexity of an algo A.

(2) Definition and intuitive meaning of

	f(n) is O(g(n)),
 	f(n) is Omega(g(n)), and
	f(n) is Theta(g(n)).

	In this course we are interested
        in the special case where f(n) = T(n),
	the worst-case time complexity of some algorithm.

(3) Explain what needs to be done to prove:

	T(n) is O(g(n)),
 	T(n) is Omega(g(n)), and
	T(n) is Theta(g(n)).

(1), (2) and (3) above are reminders of things
they should have seen in a past course,
and also seen in my lecture.

[For (1) (2) and (3) you can use these notes,
and the "asymptotic bounds" handout on the course web.
You can also use the textbook CLRS Chapter 3.]

Part (B):
---------

example 1: Time complexity analysis of SequentialSearch
-------------------------------------------------------

Let T(n) be the worst-case time complexity of the Sequential Search (SS)
algorithm (the sequential search of an item x in an array of size n).

- Show that T(n) is O(n)
     [easy: for every n, for EVERY input of size n,
            SS takes AT MOST n steps]

- Show that T(n) is Omega(1)
     [trivial: for every n, there is SOME input of size n (actually any one),
               such that SS takes AT LEAST 1 steps]

- Show that T(n) is Omega(n)
     [less trivial: for every n, there is SOME input of size n,
                    such that SS takes AT LEAST n steps]

Conclusion:  since T(n) is both O(n) and Omega(n), it is Theta(n).


More details:
-------------

SeqSearch returns true if x is in A[1..n] and false otherwise:

SeqSearch(A[1..n],x)

i := 1
found := false
while i <= n and not found do
  if A[i] = x then found := true end if
  i := i+1
end while
if found then
  return true
else
  return false
end if


Upper bound: showing that T(n) is O(n)
---------------------------------------

For every n, and EVERY input of size n, the following is true:

While loop is executed at most n times.
(Why?  Each time i is incremented, so after n iterations i > n.)
Each iteration takes c steps for some constant c.
(The detailed value of c depends on what we count as a "step",
but we don't care because we will express the time complexity in
asymptotic terms, using the O/Omega notation, so constant factors
are unimportant --- in fact, this is one of the reasons for
doing time complexity analysis using O/Omega notation.)

So for ALL inputs of size n, the time needed for the entire
algorithm is at most cn+d (for some constant d, whose
detailed value depends on the steps executed in statements
outside the loop).

So the worst case time complexity of the algorithm T(n) is O(n).

Lower bound: showing that T(n) is Omega(n)
------------------------------------------

***First explain that, for every n, there are inputs of size n,
for which the algorithm is very fast (executes few steps)***

E.g., suppose x is in A[1] (or A[15], for that matter).
-> the while loop is executed only once (or 15 times), since
the loop terminates after x is found.
-> The body of the loop takes c steps,
for some constant c.

So the time needed FOR THIS INPUT is *at least* c (or 15c).
On the basis of this input, we can say that the worst-case time
complexity of SeqSearch is Omega(1).

This is true, but it is a very weak statement.
(For full marks) we want to find the strongest result possible.

In fact, the w.c. time complexity is Omega(n).
To prove this, for every n, we have to identify a "bad" input
of size n that will cause the algorithm to take at least cn steps
(for some constant c).

A bad input is when the first appearence of x in A is in A[n].
(i.e., A[n]=x and A[i] \ne x for all i s.t. 1 <= i < n).
Note: We can't prove that this **is** a worst-case input, but we can
show that it is a "bad" input.

__For every n, and THIS input of size n__, the following is true:
-the while loop is executed n times,
-each iteration takes at least c steps (for some constant c),
So, for this input the algorithm executes at least cn steps.

Therefore:
for **ONE** input of size n, the time needed for the entire
algorithm is **at least** cn

So on the basis of THIS input, we conclude that the time complexity
of the algorithm is Omega(n).

Since the w.c. time complexity of the algorithm T(n) is both O(n) and Omega(n),
it is Theta(n^2).


Now, consider an input where the first instance of x is the "middle"
element of A (i.e, A[n/2]=x and A[i] \ne x for all i s.t.
1 <= i < n/2 --- for simplicity we assume that n is even, if not
we would take floor(n/2) instead of n/2).

In this case the loop is executed n/2 times, and each iteration
takes at least c steps for some constant c.  So for this input
the entire algorithm requires at least (c/2)n steps.

So on the basis of this input as well, the worst case time complexity
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
of the algorithm is Omega(n).

The same conclusion would be drawn for an input where the first
occurence of x in A is in position n/100.

The point of these other examples of "bad inputs" (where x is
"closer" to the beginning of the array) is to illustrate that
to argue about the worst case time complexity of an algorithm
in asymptotic terms (i.e., using O/Omega rather than computing
exact constants) we don't really have to find THE worst case input
(which in some cases can be quite tricky).  Rather it is enough to
find an input that is "bad enough", in the sense that it produces
a lower bound that matches (in asymptotic terms) the upper bound.

-------------------------------------------------------------------------
-------------------------------------------------------------------------

Example 2: Time complexity analysis of Bubblesort
-------------------------------------------------

Let T(n) be the worst-case time complexity of the BubbleSort (BBS) algorithm.

- Show that T(n) is O(n^2)
  [easy: for every n, for EVERY input of size n,
         BBS takes AT MOST n^2 steps]

- Show that T(n) is Omega(n)
  [easy: for every n, there is SOME input of size n (actually any one),
         such that BBS takes AT LEAST n steps]

- Show that T(n) is Omega(n^2)
  [harder: for every n, there is SOME input of size n,
           such that BBS takes AT LEAST n^2 steps]

Conclusion: since T(n) is both O(n^2) and Omega(n^2), it is Theta(n^2).


-------------------------------------------------------------------------------

More details:
-------------

BubbleSort(A[1..n])

last := n
sorted := false
while not sorted do
  sorted := true
  for j := 1 to last-1 do
    if A[j] > A[j+1] then
      swap A[j] and A[j+1]
      sorted := false
    end if
  end for
  last := last-1
end while

Explain informally how bubblesort works:
At the end of the i-th iteration of the while loop,
A[last+1..n] contains in sorted order the i largest
elements of A.
(This is a loop invariant that can be proved using
techniques students have seen in 238/B38.)

For example:
10 5 9 8 6
j=1: 5 9 8 6 10
j=2: 5 8 6 9 10
j=3: 5 6 8 9 10
j=4: 5 6 8 9 10

Upper bound: showing that T(n) is  O(n^2).
-----------------------------------------

For every n, and for EVERY input of size n the following is true:

1) While loop is executed at most n-1 times.
   Each time "last" is reduced by 1.
-> after n-1 iterations, last=1
-> no iterations of the for loop are performed on iteration n-1
-> "sorted" is true on iteration n-1
-> while loop ends on iteration n-1 (if not earlier)

2) Each iteration of the while loop takes at most cn time,
for some constant c
-for loop excutes at most n-1 times, since "last" is always at most n.

3) d steps are taken outside the while loop, for some constant d

Therefore:
for **ALL** inputs of size n, the time needed for the entire
algorithm is **at most** cn(n-1) + d

So the worst case time complexity of the algorithm T(n) is O(n^2).


Lower bound: showing that T(n) is Omega(n) and also Omega(n^2).
---------------------------------------------------------------

First explain that, for every n, there are inputs of size n
for which the algorithm is very fast (executes few steps).

E.g., suppose the input array A (of size n) is already sorted.
Then the while loop is executed only once (why?).
The body of the loop takes at least c(n-1) time,
for some constant c (because the for loop is executed n-1 times).

So the time needed FOR THIS INPUT is at least c(n-1).
On the basis of this input we can say that the worst-case time
complexity of BubbleSort T(n) is Omega(n).
This is true, but it is a very weak statement.

In fact, the w.c. time complexity of BubbleSort is Omega(n^2).
To prove this, for every n we have to identify a "bad" input
of size n that will cause the algorithm to take at least cn^2 steps
(for some constant c).

Such a bad input is when the input array A is in reverse sorted order.
Then we can see that

* After one iteration of the while loop, the largest element is in A[n]
  and the rest of the array A[1..n-1] is in reverse sorted order.
  This iteration of the while loop requires at least n-1 steps
  (because the inner for loop is is executed n-1 times).

* After two iterations of the while loop, the largest two elements
  are in A[n-1..n]
  and the rest of the array A[1..n-2] is in reverse sorted order.
  This iteration of the while loop requires at least n-2 steps
  (because the inner for loop is is executed n-2 times).

* After three iterations of the while loop, the largest three elements
  are in A[n-2..n]
  and the rest of the array A[1..n-3] is in reverse sorted order.
  This iteration of the while loop requires at least n-3 steps
  (because the inner for loop is is executed n-3 times).

	.
	.
	.

* After n-1 iterations of the while loop, the largest n-1 elements
  are in A[2..n]
  and the rest of the array A[1] is in reverse sorted order.
  (So now the array is sorted.)
  This iteration of the while loop requires at least 1 step
  (because the inner for loop is is executed 1 time).

So, for this input the while loop requires the execution of at least

	(n-1) + (n-2) + ... + 1 = (n-2)(n-1)/2 steps

On the basis of this, we conclude that the worst-case time complexity
of this algorithm is Omega(n^2).

Since the w.c. time complexity of BubbleSort is both O(n^2) and Omega(n^2),
it is Theta(n^2).

----------------------------------------------------------------------
