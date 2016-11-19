

def dot_prod(u, v):
    """
    Return the dot product of u and v

    @param list[float] u: vector of floats
    @param list[float] v: vector of floats
    @rtype: float

    assume u, v are vectors of same length

    >>> dot_prod([1.0, 2.0], [3.0, 4.0])
    11.0
    """
    # sum of products of pairs of corresponding coordinates of u and v
    s = 0

    for i in range(len(u)):
        s += u[i] * v[i]

    return s

def matrix_vector_prod(m, u):
    """
    Return the matrix-vector product of m x u

    @param list[list[float]] m: matrix
    @param list[float] u: vector
    @rtype: list[float]
    >>> matrix_vector_prod([[1.0, 2.0], [3.0, 4.0]], [5.0, 6.0])
    [17.0, 39.0]
    """
    # list of dot products of vectors in m with v

    v = []
    for vector in m:
        v.append(dot_prod(vector, u))
    return v


def pythagorean_triples(n):
    """
    Return list of pythagorean triples as non-descending tuples
    of ints from 1 to n.

    Assume n is positive.

    @param int n: upper bound of pythagorean triples

    >>> pythagorean_triples(5)
    [(3, 4, 5)]
    """
    l = []

    # range stops at n-1 so have to add 1
    for i in range(1, n+1):
        for j in range(1, n+1):
            for k in range(1, n+1):
                t = (i, j, k)
                if (t[0] <= t[1] <= t[2]) and (t[0]**2 + t[1]**2) == t[2]**2:
                    l.append(t)

    return l
