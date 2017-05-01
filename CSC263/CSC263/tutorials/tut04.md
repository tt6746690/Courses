Tutorial 4: Probability review
==============================

This week's tutorial will be a review of (very) basic
discrete probability theory + what "average time complexity" means.

[Note that some notation in this file is in LaTeX style: for example,
S_n means S subscript n, and { } groups long subscripts together.
The { } also means "a set", so try not to get confused.]

A) Probability definitions:
---------------------------
(This should be review for EVERYONE, so don't spend too much time on this.
Define each concept and use the example of rolling a dice to explain each one.
You may want to put these in a table.)

Term: Event 
Definition: one or more possible outcomes
Example: rolling a 1 or rolling an even number

Term: Sample space (S)    
Definition: a set of event
Example: {rolling a 1, rolling a 2, ..., rolling a 6}

Term: Probability distribution (Pr)
Definition: a mapping from events of S to real numbers such that the 
following probability axioms hold:
  Pr(A) \geq 0 for any event A
  \sum_{A \in S} Pr(A) = 1
Example: Pr(rolling an x) = 1/6  where 1 <= x <= 6 

Term: Probability of an event occurring (Pr(A))
Definition: Pr[A] = \sum_{s is an outcome belonging to event A} Pr[s]
Example: Pr(an even number is rolled) = Pr(rolling a 2) + Pr(rolling a 4) +
Pr(rolling a 6) = 1/2

Term: Random variable
Intuitively: a variable that associates a real number with each possible
outcome.
Definition: a function from a finite or countably infinite sample space S
to the real numbers
Example:
X = the number on the dice after a roll
sample space -> real number
rolling a 1 -> 1
rolling a 2 -> 2
...
rolling a 6 -> 6 

Term: Expected value of a random variable
Intuitively: the "average" value of a random variable
Definition: The expected value of X = E[X] = \sum_x x Pr[X=x]
Example: E[X] = \sum_{1 \leq x \leq 6} x Pr[X=x] = 1/6 (6)(7)/2 = 3.5

B) The definition of the average running time (T(n)) of an algo A:
------------------------------------------------------------------
(New material starts here, so slow down now.)
 
Let A be an algorithm.
Let S_n be the sample space of all inputs of size n.  

In order to talk about the "average" running time of A over S_n, we NEED 
to specify how likely each input is.  This is done by specifying a 
probability distribution over S_n.
 
Once we specify a probability distribution over S_n: 
let t_n(x) be the number of steps taken by A on input x, for x in S_n.  

t_n is a random variable (it assigns a numerical value to each element 
of our probability space).
 
From probability theory, we know that the "average" value of t_n(x) is
E[t_n], the expected number of steps taken by A on inputs of size n, and
is equal to
         E[t_n] =   sum    t_n(x) * Pr(x)
                  x in S_n
 
where Pr(x) is the probability of input x given by the probability
distribution.

Accordingly, we define the average case running time of A to be T(n) = E[t_n].

C) Illustrate the above concepts by doing the following question:
-----------------------------------------------------------------
 
Consider a linear linked list L of 8 elements with distinct keys.
What is the expected time-complexity of Search(k,L) (i.e., searching 
for the element with key k in the list L), under the following assumptions:
 
  1) The input k is random and follows the following probability 
  distribution:
    For each i, 1 <= i <= 8, the probability that k is the i-th element
    of the list L is 1/16;
    the probability that k is _not_ in L is 1/2.
 
  2) The time to execute Search(k,L) is 2i if k is the i-th element of L;
  and it is 17 if k is not in L.

Prove your answer rigorously using the relevant concepts and definitions
needed from probability theory.

Solution:
Sample space: S = {1,2,3,4,5,6,7,8,9} where element j \in S represents the
key at the j-th location in L, except for j=9, which represents all the keys
that are not in L

[OR S={k is in the first position, k is in the second position, ...,
k is in the 8th position, k is not in the list}]

Probability distribution: For j \in S, Pr(j) = 1/16 for 1 \leq j \leq 8
                                             = 1/2 for j=9

Random variable: Let T(j) be the random variable denoting the search time
associated with j \in S.

T(j) = 2j for 1 \leq j \leq 8
     = 17 for j=9

Expected value: We are seeking E(T). By definition,
E(T) = \sum_{j \in S} Pr(j) T(j)

Therefore:
E(T) = \sum_{j = 1}^{8} 1/16 2j + 1/2 17 
     = 2/16 \sum_{j = 1}^{8} j + 17/2
     = 2/16 (8)(9)/2 + 17/2
     = 26/2
     = 13
