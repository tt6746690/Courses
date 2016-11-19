# recursion exercises with nested lists


def gather_lists(list_):
    """
    Return the concatenation of the sublists of list_.

    @param list[list] list_: list of sublists
    @rtype: list

    >>> list_ = [[1, 2], [3, 4]]
    >>> gather_lists(list_)
    [1, 2, 3, 4]
    """
    # this is a case where list comprehension gets a bit unreadable
    new_list = []
    for sub in list_:
        for element in sub:
            new_list.append(element)
    return new_list


def list_all(obj):
    """
    Return a list of all non-list elements in obj or obj's sublists, if obj
    is a list.  Otherwise, return a list containing obj.

    @param list|object obj: object to list
    @rtype: list

    >>> obj = 17
    >>> list_all(obj)
    [17]
    >>> obj = [1, 2, 3, 4]
    >>> list_all(obj)
    [1, 2, 3, 4]
    >>> obj = [[1, 2, [3, 4], 5], 6]
    >>> list_all(obj)
    [1, 2, 3, 4, 5, 6]
    >>> all([x in list_all(obj) for x in [1, 2, 3, 4, 5, 6]])
    True
    >>> all([x in [1, 2, 3, 4, 5, 6] for x in list_all(obj)])
    True
    """
    if isinstance(obj, list):
        return gather_lists([list_all(x) for x in obj])
    else:
        return [obj]


def max_length(obj):
    """
    Return the maximum length of obj or any of its sublists, if obj is a list.
    otherwise return 0.

    @param object|list obj: object to return length of
    @rtype: int

    >>> max_length(17)
    0
    >>> max_length([1, 2, 3, 17])
    4
    >>> max_length([[1, 2, 3, 3], 4, [4, 5]])
    4
    """

    if isinstance(obj, list):
            return max(len(obj), max([max_length(x) for x in obj]))
    else:
        return 0



def list_over(obj, n):
    """
    Return a list of strings of length greater than n in obj, or sublists of obj, if obj
    is a list.  Otherwise, if obj is a string return a list containing obj if obj has
    length greater than n, otherwise an empty list.

    @param str|list obj: possibly nested list of strings, or string
    @param int n: non-negative integer
    @rtype: list[str]

    >>> list_over("five", 3)
    ['five']
    >>> list_over("five", 4)
    []
    >>> L = list_over(["one", "two", "three", "four"], 3)
    >>> L
    ['three', 'four']
    >>> all([x in L for x in ["three", "four"]])
    True
    >>> all([x in ["three", "four"] for x in L])
    True
    """

    if isinstance(obj, list):
        return gather_lists([list_over(x, n) for x in obj])
    else:
        if len(obj) > n:
            return [obj]
        return []

def list_even(obj):
    """
    Return a list of all event integers in obj or sublists of obj, if obj is a list.
    Otherwise, if obj is an even integer return a list containing obj, and if obj
    is an odd integer, return an empty list.

    @param int|list obj: possibly nested list of ints, or int
    @rtype: list[int]

    >>> list_even(3)
    []
    >>> list_even(16)
    [16]
    >>> L = list_even([1, 2, 3, 4, 5])
    >>> all([x in L for x in [2, 4]])
    True
    >>> all([x in [2, 4] for x in L])
    True
    >>> L = list_even([1, 2, [3, 4], 5])
    >>> all([x in L for x in [2, 4]])
    True
    >>> all([x in [2, 4] for x in L])
    True
    >>> L = list_even([1, [2, [3, 4]], 5])
    >>> all([x in L for x in [2, 4]])
    True
    >>> all([x in [2, 4] for x in L])
    True
    """
    if isinstance(obj, list):
        return gather_lists([list_even(x) for x in obj])
    elif obj % 2 == 0:
        return [obj]
    else:
        return []


def count_even(obj):
    """
    Return the number of even numbers in obj or sublists of obj
    if obj is a list.  Otherwise, if obj is a number, return 1
    if it is an even number and 0 if it is an odd number.

    @param int|list obj: object to count even numbers from
    @rtype: int

    >>> count_even(3)
    0
    >>> count_even(16)
    1
    >>> count_even([1, 2, [3, 4], 5])
    2
    """
    if isinstance(obj, list):
        return sum([count_even(x) for x in obj])
    elif obj % 2 == 0:
        return 1
    else:
        return 0


def count_all(obj):
    """
    Return the number of elements in obj or sublists of obj if obj is a list.
    Otherwise, if obj is a non-list return 1.

    @param object|list obj: object to count
    @rtype: int

    >>> count_all(17)
    1
    >>> count_all([17, 17, 5])
    3
    >>> count_all([17, [17, 5], 3])
    4
    """
    if isinstance(obj, list):
        return sum([count_all(x) for x in obj])
    else:
        return 1


def count_above(obj, n):
    """
    Return tally of numbers in obj, and sublists of obj, that are over n, if
    obj is a list.  Otherwise, if obj is a number over n, return 1.  Otherwise
    return 0.

    >>> count_above(17, 19)
    0
    >>> count_above(19, 17)
    1
    >>> count_above([17, 18, 19, 20], 18)
    2
    >>> count_above([17, 18, [19, 20]], 18)
    2
    """
    if isinstance(obj, list):
        return sum([count_above(x, n) for x in obj])
    elif obj > n:
        return 1
    else:
        return 0

if __name__ == "__main__":
    import doctest
    doctest.testmod()
