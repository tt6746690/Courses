Tutorial 8: Amortized analysis
==============================

For this tutorial, go through two examples of amortized analysis.
(1) the PUSH/MULTIPOP example - go through both methods
(2) a variant on dynamic table expansion - see if they can work out the answer

The first example is from the textbook, and is good for intuition.
The second example was a final exam question from last winter.

--------------------------------------------------------------------------
Example 1:

operation           description                               actual cost
PUSH                pushes an item on to the stack            1
MULTIPOP(k)         pops k items from the stack               min(k,s)
where s is the number of items on the stack.
Note: we cannot pop items if the stack is empty.

What would be the worst-case sequence of operations? (Let the students guess).
One possible suggestion is push multipop(1) push multipop(1) ...
Here each operation has cost 1, so the total cost of n operations = n

Another suggestion is n-1 pushes followed by a multipop: push push push ...
multipop(n-1)
Each push has cost 1 and the multipop has cost n-1. Thus the total cost
is 2n-2.

This gives us some intuition into the expected amortized cost.

Method 1: Aggregate method
---------------------------

Consider the cost of the push and multipop operations separately.

cost of worst-case sequence = cost of all pushes in sequence + cost of all
multipops

The two keys things to notice here are:
1) In any sequence of n pushes and multipops, the total cost of pushes is at
most n.
2) An item cannot be popped from the stack without first being pushed on to the
stack, thus the cost of all multipops in any sequence <= cost of all pushes.

So, cost of worst-case sequence <= 2 x (cost of all pushes in sequence) <= 2n.
This gives us an amortized cost of 2n/n = 2 for each operation in the sequence.

Method 2: Accounting Method
---------------------------

Here we charge each operation as small cost as possible, but we *cannot* go
into debt by spending more than we have saved.

if actual cost < amortized cost, we credit the extra charge to the "bank"
if amortized cost < actual cost, we spend some savings in the bank... but we
cannot use more than we have saved.

Let amortized cost of push = 2, amortized cost of pop = 0.
When we push an item on to the stack, we use $1 to pay for the push operation,
and we put $1 in the bank. We use the saved $1 later when we pop the item from
the stack.


cost of worst-case sequence <= \sum amortized cost of pushes + \sum amortized
cost of multipops
                             = \sum (2) + \sum(0)
                            <= 2n
This gives us an amortized cost of 2n/n = 2 for each operation in the sequence.


-----------------------------------------------------------------------------

Example 2: Solve the following question with them
(see whether they can figure out the answer themselves):


In class we analysed Dynamic Table Expansion
under the assumption that if we want to insert a new element
in a table $T$ that is full,
we first copy all the elements of $T$ into a new table $T'$
of size $|T'| = 2 |T|$, and then enter the new element in $T'$.
In this question, we consider cases where the size of $T'$
is not double the size of $T$.

In the following, assume (as in class) that entering an element
in an empty slot of a table costs 1,
and copying an element from a table into a new table also costs 1.

\subquestion[10]
\textit{Suppose that $|T'| = |T| + 1000$, i.e, each new table has 1000 more
slots than the previous one.}

Starting with an empty table $T$ with 1000 slots, we insert
        a sequence of $n$ elements.
What is the \emph{amortized cost per insertion}?

Choose \emph{one} of the possible answers below and circle it.
Justify your answer.

\emph{Hint:} Use aggregate analysis.


\begin{enumerate}

\item At most 2.

\item More than 2 but at most 1000.

\item $\Theta(\sqrt{n} )$.

\item $\Theta(n)$.		%this is the answer

\item $\Theta(n \sqrt{n} )$.

\item $\Theta(n^2)$.

\end{enumerate}


