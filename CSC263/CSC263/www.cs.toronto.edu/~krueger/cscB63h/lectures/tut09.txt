TUTORIAL 9: Dijkstra's algorithm
================================

Dijkstra's algorithm solves the single-source shortest-paths problem on a 
weighted, directed graph G = (V, E) for the case in which all edge weights 
are nonnegative.

In the algorithm below:
S = the set of vertices whose shortest path from the source have been found
Q = V-S (at the start of each iteration of the while loop)

DIJKSTRA(G, w, s)
  INITIALIZE-SINGLE-SOURCE(G, s)
  S = \emptyset 
  Q = V[G]
  while Q \neq \emptyset  
      do u = EXTRACT-MIN(Q)
         S = S \cup {u}
         for each vertex v \in Adj[u]
             do RELAX(u, v, w)

-------------------------------------
INITIALIZE-SINGLE-SOURCE initializes all the parent variables (pi[v]) 
to NIL and the shortest distance from the source (d[v]) to infinity.
The distance from s to s (d[s]) is initialized to 0.

INITIALIZE-SINGLE-SOURCE(G, s)
  for each vertex v \in V[G]
       do d[v] = infinity 
          pi[v] = NIL
  d[s] = 0

-------------------------------------
RELAX tests whether we can improve the shortest path to v 
found so far by going through u and, if so, updates d[v] and pi[v].

RELAX(u, v, w)
  if d[v] > d[u] + w(u, v)
     then d[v] = d[u] + w(u, v)
          pi[v] = u

-------------------------------------

Example:
Compute the shortest path from s from all vertices in the graph in Figure
24.6 of the textbook.

Show S, Q and the distances from the source for each iteration of the while
loop. 

-------------------------------------

DIJKSTRA's algorithm uses a priority queue. Which operations are performed
where? (Let the students guess them all.)

1) Q = V[G]: initializes the queue using BUILD-MIN-HEAP  (or INSERT)
   i.e. inserts all |V| vertices into the queue

2) u = EXTRACT-MIN(Q) 
   we extract one vertex during each iteration of the while loop
   (after |V| iterations Q is empty)

3) d[v] = d[u] + w(u, v):  DECREASE-KEY

What is the worst-case running time of DIJKSTRA's algorithm using the 
min-heap implementation of the priority queue?

Let n = |V| and m = |E|.

operation                         worst-case running time
----------                        -----------------------
BUILD-MIN-HEAP                       O(n)
EXTRACT-MIN                          O(log n)
DECREASE-KEY                         O(log n)

-Each vertex is initially inserted into Q.
-On each iteration of the while loop exactly one vertex is removed from Q
-The while loop terminates when Q is empty
Therefore, the while loop iterates n times and the function EXTRACT-MIN
executes n times.

In addition, since each vertex is removed from Q exactly once, the adjacency
list of each vertex is scanned exactly once.
The operation RELAX(u,v,w) is only preformed when the adjacency list of u is
scanned.
-> RELAX(u,v,w) is only performed once for each edge (u,v).
-> RELAX is performed at most m times

Therefore, the total worst-case running time is:
O(n+ nlog n + mlog n) = O((n+m)log n) = O(m log n)

-------------------------

Theorem 24.6: Correctness of DIJKSTRA's algorithm

If Dijkstra's algorithm, is run on a weighted, directed graph G = (V, E) with 
a non-negative weight function w and source s, then when it terminates d[u] 
is the shortest path from s to u.

Proof:
------
We will show that the following loop invariant always holds:
At the start of each iteration of the while loop, d[v] is the shortest path 
from s to v for each vertex v \in S.

Initialization: Initially, S = \emptyset, so the invariant is true.
Maintenance: Assume, for a contradiction, that there is a u \in S such that 
d(u) is not the shortest path from s to u.

Without loss of generality, assume u was the first vertex added to S, such that
the loop invariant did not hold.

Fact:
1) u \neq s because s has distance 0 from itself, d[s] is set to 0, and s
 is the first node added to S
2) S \neq \emptyset just before u was added to S, since s is in S.
3) There must be a path from s to u, because if not: 
   d[u] = infinity and the shortest path from s to u = infinity.   
   (i.e. d[u] = the shortest path from s to u)

By 3, we know that there is some path from s to u, so there must be a 
shortest path from s to u, say p

Draw p just prior to adding u to S (see figure 24.7):


Consider the first vertex y along path p that is not in S. (Note that y 
might be u.)

Let x \in S be the parent/predecessor of y 

Claim: When u is added to S, d[y] is the shortest path from s to y
--------
Proof of Claim:
----------------
 d[x] is the shortest path from s to x, since x is in S and u is the 
 first vertex added to S such that d[u] is not the shortest path from s to u.

 We call the function RELAX on the edge (x,y) when x was added to S
 -> either d[y] = d[x] + w(x,y) is the shortest path from s to y 
 OR
 -> there is a shorter path p' from s to y that does not go through x.
    In this case, consider the path p'' from s to u that is the same as
    p' from s to y and the same as p from y to u. Path p'' provides a 
    contradiction, since p'' is shorter than p and we defined p to be the 
    shortest path from s to u
Thus, d[y] = d[x] + w(x,y) is the shortest path from s to y [end of Claim]

y occurs before u on the shortest path from s to u and all edge weights 
are non-negative -> shortest path from s to y <= shortest path from s to u

Therefore (*):
d[y] = shortest path from s to y
   <= shortest path from s to u
   <= d[u] (since d[u] cannot be less that the shortest path from s to u)

But u and y are both in V-S when u is chosen, so d[u] <= d[y]
(since we extract the vertex with the minimum d value).

Therfore since d[y] <= d[u] and d[u] <= d[y], we have d[u] = d[y]. (**)

Thus, by (*) and by (**):
d[y] = the shortest path from s to y = shortest path from s to u = d[u]
-> d[u] = shortest path from s to u

This contradicts the fact that d[u] is not the shortest path from s to u.

Termination: At the termination of the algorithm, Q = \emptyset
Since Q = V-S, S = V.

Therefore, d[u] = shortest path from s to u for all u \in V.
