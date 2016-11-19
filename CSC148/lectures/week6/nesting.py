# some recursive functions on nested lists


def depth(obj):
    """
    Return 0 if obj is a non-list, or 1 + maximum
    depth of elements of obj, a possibly nested
    list of objects.

    Assume obj has finite nesting depth

    @param list[object] | object obj: possibly nested list of objects

    >>> depth(3)
    0
    >>> depth([])
    1
    >>> depth([[], [[]]])
    3
    >>> depth([1, 2, 3])
    1
    >>> depth([1, [2, 3], 4])
    2
    """
    if not isinstance(obj, list):
        # obj is not a list
        return 0
    elif obj == []:
        return 1
    else:
        # obj is a list
        return 1 + max([depth(x) for x in obj])



def rec_max(obj):
    """
    Return obj if it's an int, or the maximum int in obj,
    a possibly nested list of numbers.

    Assume: obj is an int or non-empty list with finite nesting depth,
    and obj doesn't contain any empty lists

    @param int|list[int|list[...]] obj: possibly nested list of int

    >>> rec_max([17, 21, 0])
    21
    >>> rec_max([17, [21, 24], 0])
    24
    >>> rec_max(31)
    31
    """
    if not isinstance(obj, list):
        # obj is not a list
        return obj
    else:
        # obj is a list
        return max([rec_max(x) for x in obj])



def concat_strings(string_list):
    """
    Concatenate all the strings in possibly-nested string_list.

    @param list[str]|str string_list
    @rtype: str

    >>> list_ = ["how", ["now", "brown"], "cow"]
    >>> concat_strings(list_)
    'hownowbrowncow'
    """
    if isinstance(string_list, str):
        # string_list is a str
        return string_list
    else:
        return "".join([concat_strings(x) for x in string_list])

    # equivalent to
    # return "".join([concat_strings(x) if isinstance(string_list, list) else x for x in string_list])


def nested_contains(list_, value):
    """
    Return whether list_, or any nested sub-list of list_ contains value.

    @param list list_: list to search
    @param object value: non-list value to search for
    @rtype: bool

    >>> list_ = ["how", ["now", "brown"], 1]
    >>> nested_contains(list_, "brown")
    True
    >>> nested_contains([], 5)
    False
    >>> nested_contains([5], 5)
    True
    """
    # check out Python built-in any
    return any([nested_contains(x, value)
                if isinstance(x, list) else x == value
                for x in list_])


def nested_count(list_):
    """
    Return the number of non-list elements of list_ or its nested sub-lists.

    @param list list_: possibly nested list to count elements of
    @rtype: int

    >>> list_ = ["how", ["now", "brown"], "cow"]
    >>> nested_count(list_)
    4
    """
    # functional if helps here
    return sum([nested_count(x)
                if isinstance(x, list) else 1
                for x in list_])




if __name__ == '__main__':
    import doctest
    doctest.testmod()
