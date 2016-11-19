# some definitions using comprehensions


def dot_prod(u, v):
    """
    Return the dot product of u and v

    @param list[float] u: vector of floats
    @param list[float] v: vector of floats
    @rtype: float

    >>> dot_prod([1.0, 2.0], [3.0, 4.0])
    11.0
    """
    # sum of products of pairs of corresponding coordinates of u and v
    return sum([u_coord * v_coord for u_coord, v_coord in zip(u, v)])


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
    return [dot_prod(v, u) for v in m]


def pythagorean_triples(n):
    """
    Return list of pythagorean triples as non-descending tuples
    of ints from 1 to n.

    Assume n is positive.

    @param int n: upper bound of pythagorean triples

    >>> pythagorean_triples(5)
    [(3, 4, 5)]
    """
    # helper to check whether a triple is pythagorean and non_descending
    # you could also use a lambda instead of this nested function def.
    def _ascending_pythagorean(t):
        # """
        # Return whether t is pythagorean and non-descending.
        #
        # @param tuple[int] t: triple of integers to check
        # """
        return (t[0] <= t[1] <= t[2]) and (t[0]**2 + t[1]**2) == t[2]**2

    # filter just the ones that satisfy ascending_pythagorean
    # produce a list from the filter for ascending_pythagoreans from...
    return list(filter(_ascending_pythagorean,
                       # ...list of all triples in the range 1..n
                       [(i, j, k)
                        for i in range(1, n + 1)
                        for j in range(1, n + 1)
                        for k in range(1, n + 1)]))


if __name__ == '__main__':
    import doctest
    doctest.testmod()
