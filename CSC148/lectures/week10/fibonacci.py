# some recursive explorations
# import resource
# resource.setrlimit(resource.RLIMIT_STACK, (2**29, -1))
# import sys
# sys.setrecursionlimit(10**6)
from random import shuffle


def fibonacci(n):
    """
    Return the nth fibonacci number, that is n if n < 2,
    or fibonacci(n-2) + fibonacci(n-1) otherwise.

    @param int n: a non-negative integer
    @rtype: int

    >>> fibonacci(0)
    0
    >>> fibonacci(1)
    1
    >>> fibonacci(3)
    2
    """
    if n < 2:
        return n
    else:
        return fibonacci(n - 2) + fibonacci(n - 1)


def fib_memo(n, seen):
    """
    Return the nth fibonacci number reasonably quickly.

    @param int n: index of fibonacci number
    @param dict[int, int] seen: already-seen results
    """
    if n not in seen:
        seen[n] = (n if n < 2
                   else fib_memo(n - 2, seen) + fib_memo(n - 1, seen))
    return seen[n]

# quick sort
def qs(list_):
    """
    Return a new list consisting of the elements of list_ in
    ascending order.

    @param list list_: list of comparables
    @rtype: list

    >>> qs([1, 5, 3, 2])
    [1, 2, 3, 5]
    """
    if len(list_) < 2:
        return list_[:]
    else:
        return (qs([i for i in list_ if i < list_[0]]) +
                [list_[0]] +
                qs([i for i in list_[1:] if i >= list_[0]]))


def merge(L1, L2):
    """return merge of L1 and L2

    >>> merge([1, 3, 5], [2, 4, 6])
    [1, 2, 3, 4, 5, 6]
    >>> merge([1, 2, 3], [0, 4, 5])
    [0, 1, 2, 3, 4, 5]
    >>> merge([0], [1, 2, 3, 4])
    [0, 1, 2, 3, 4]
    """
    L = []
    i1, i2 = 0, 0
    while i1 < len(L1) and i2 < len(L2):
        if L1[i1] < L2[i2]:
            L.append(L1[i1])
            i1 += 1
        else:
            L.append(L2[i2])
            i2 += 1
    return L + L1[i1:] + L2[i2:]


def merge_sort(L):
    """Produce copy of L in non-decreasing order

    >>> merge_sort([1, 5, 3, 4, 2])
    [1, 2, 3, 4, 5]
    >>> L = list(range(20))
    >>> shuffle(L)
    >>> merge_sort(L) == list(range(20))
    True
    """
    if len(L) < 2 :
        return L[:]
    else :
        return merge(merge_sort(L[:len(L) // 2]),
                     merge_sort(L[len(L) // 2 :]))


if __name__ == "__main__":
    import doctest

    doctest.testmod()
    print(fib_memo(150, {}))
