def max_depth(obj):
    """
    Return 1 + the maximum depth of obj's elements if obj is a list.
    Otherwise return 0.

    @param object|list obj: list or object to return depth of
    @rtype: int

    >>> max_depth(17)
    0
    >>> max_depth([])
    1
    >>> max_depth([1, "two", 3])
    1
    >>> max_depth([1, ["two", 3], 4])
    2
    >>> max_depth([1, [2, ["three", 4], 5], 6])
    3
    """
    if not isinstance(obj, list):
        return 0
    elif obj == []:
        return 1
    else:
        return 1 + max([max_depth(x) for x in obj])


if __name__ == "__main__":
    import doctest
    doctest.testmod()
