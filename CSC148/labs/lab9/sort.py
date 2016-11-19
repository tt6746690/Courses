# === Selection sort === #


def find_min(list_, i):
    """
    Return the index of the smallest item in list_[i:].

    @param list list_: list to search
    @param int i: index to search from
    @rtype: int

    >>> find_min([1, 2, 3], 1)
    1
    """
    smallest = i
    length = len(list_)
    for j in range(i + 1, length):
        if list_[j] < list_[smallest]:
            smallest = j
    return smallest


# find_min using built-in min, tuple comparisons, and enumerate
# should be faster...
# def find_min(list_, i):
    # """Return index of smallest item in list_[i:]."""
    # return min([(k, j) for j, k in enumerate(list_[i:])])[1] + i


def selection_sort(list_):
    """
    Sort the items in list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    # i = 0
    # while i != len(list_) - 1:
    length = len(list_)
    for i in range(length - 1):
        # Find the smallest remaining item and move it to index i.
        smallest = find_min(list_, i)
        list_[smallest], list_[i] = list_[i], list_[smallest]
        # i += 1


# === Insertion sort 1 === #
def insert(list_, i):
    """
    Move list_[i] to where it belongs in list_[:i]

    @param list list_: list to modify
    @param int i: index of element to place
    @rtype: None
    """
    v = list_[i]
    while i > 0 and list_[i - 1] > v:
        list_[i] = list_[i - 1]
        i -= 1
    list_[i] = v


def insertion_sort_1(list_):
    """
    Sort the items in list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    i = 1
    # Insert each item i where it belongs in list_[0:i + 1]
    while i != len(list_):
        insert(list_, i)
        i += 1


# === Insertion sort 2 === #
def find_insertion_point(list_, i):
    """
    Return the index where list_[i] belongs in list_[:i + 1].

    @param list list_: list to find insertion point in
    @param int i: index of element to insert
    @rtype: int

    >>> find_insertion_point([1, 3, 2], 2 )
    1
    """
    v = list_[i]
    while i > 0 and list_[i - 1] > v:
        i -= 1
    return i


def insertion_sort_2(list_):
    """
    Sort the items in list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    i = 1
    # Insert each item i where it belongs in list_[0:i + 1]
    while i != len(list_):
        the_spot = find_insertion_point(list_, i)
        v = list_[i]
        del list_[i]
        list_.insert(the_spot, v)
        i += 1


# === Bubblesort 1 == #
def bubblesort_1(list_):
    """
    Sort the items in list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    j = len(list_) - 1
    while j != 0:
        # Swap every item that is out of order.
        for i in range(j):
            if list_[i] > list_[i + 1]:
                list_[i], list_[i + 1] = list_[i + 1], list_[i]
        j -= 1


# === Bubblesort 2 === #
def bubblesort_2(list_):
    """
    Sort the items in list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    j = len(list_) - 1
    swapped = True
    # Stop when no elements are swapped.
    while swapped and j != 0:
        swapped = False
        # Swap every item that is out of order.
        for i in range(j):
            if list_[i] > list_[i + 1]:
                swapped = True
                list_[i], list_[i + 1] = list_[i + 1], list_[i]
        j -= 1


# === Mergesort 1 === #
def _merge_1(left, right):
    """
    Merge sorted lists left and right into a new list and return that new
    list.

    @param list left: left elements of desired list
    @param list right: right elements of desired list
    @rtype: list

    >>> _merge_1([1, 3, 5], [2, 4,6])
    [1, 2, 3, 4, 5, 6]
    """
    result = []
    i, j = 0, 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result += left[i:]
    result += right[j:]
    return result


def _mergesort_1(list_):
    """
    Return a new list that is a sorted version of list list_.

    @param list list_: list to sort from
    @rtype: list

    >>> _mergesort_1([5, 1, 3])
    [1, 3, 5]
    """
    if len(list_) < 2:
        return list_[:]
    else:
        middle = len(list_) // 2
        left = _mergesort_1(list_[:middle])
        right = _mergesort_1(list_[middle:])
    return _merge_1(left, right)


def mergesort_1(list_):
    """
    Sort list list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    list_[:] = _mergesort_1(list_)


# === Mergesort 2 === #
def _merge_2(list_, i, mid, j):
    """
    Merge the two sorted halves list_[i:mid + 1] and
    list_[mid + 1:j + 1] and return them in a new list.
    Notice that list_[mid] belongs in the left half and list_[j]
    belongs in the right half -- the indices are inclusive.

    @param list list_: list to sort
    @param int i: starting index for first half of list to sort
    @param int mid: index for middle of list to sort
    @param int j: index for end of list to sort

    >>> _merge_2([1, 3, 5, 2, 4, 6], 0, 2, 5)
    [1, 2, 3, 4, 5, 6]
    """
    result = []
    left = i
    right = mid + 1
    # Done when left > mid or when right > j; i.e.,
    # when we've finished one of the halves.
    while left <= mid and right <= j:
        if list_[left] < list_[right]:
            result.append(list_[left])
            left += 1
        else:
            result.append(list_[right])
            right += 1
    # Items left: list_[left:mid + 1]
    #             list_[right:j + 1]
    return result + list_[left:mid + 1] + list_[right:j + 1]


def _mergesort_2(list_, i, j):
    """
    Sort the items in list_[i] through list_[j] in
    non-decreasing order.

    @param list list_: list to sort
    @param int i: index to begin sorting from
    @param int j: index to continue sorting until
    @rtype: None
    """
    if i < j:
        mid = (i + j) // 2
        _mergesort_2(list_, i, mid)
        _mergesort_2(list_, mid + 1, j)
        list_[i:j + 1] = _merge_2(list_, i, mid, j)


def mergesort_2(list_):
    """
    Sort list list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    _mergesort_2(list_, 0, len(list_) - 1)


# === Quicksort 1 === #
def _partition_1(list_):
    """
    Rearrange list_ so that items < list_[0] are at the beginning
    and items >= list_[0] are at the end, and return the new index
    for list_[0].

    @param list list_: list to partition
    @rtype: int

    >>> _partition_1([2, 1, 3])
    1
    """
    v = list_[0]
    i = 1
    j = len(list_) - 1
    while i <= j:
        if list_[i] < v:
            i += 1
        else:
            list_[i], list_[j] = list_[j], list_[i]
            j -= 1
    list_[0], list_[j] = list_[j], list_[0]
    return j


def _quicksort_1(list_):
    """
    Sort list_ by partitioning it around the first item,
    then recursing.

    @param list list_: list to sort
    @rtype: list

    >>> _quicksort_1([3, 1, 2, 5])
    [1, 2, 3, 5]
    """
    if len(list_) <= 1:
        return list_
    else:
        pivot = _partition_1(list_)
        left = list_[:pivot]
        right = list_[pivot + 1:]
        return _quicksort_1(left) + [list_[pivot]] + _quicksort_1(right)


def quicksort_1(list_):
    """
    Sort list list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    list_[:] = _quicksort_1(list_)


# === Quicksort 2 === #
def _partition_2(list_, i, j):
    """Rearrange list_[i:j] so that items < list_[i] are at
    the beginning and items >= list_[i] are at the end,
    and return the new index for list_[i].

    @param list list_: list to partition
    @param int i: beginning of partition slice
    @param int j: end of partition slice
    @rtype: int

    >>> _partition_2([1, 5, 2, 4, 3], 1, 4)
    3
    """
    v = list_[i]
    k = i + 1
    j -= 1
    while k <= j:
        if list_[k] < v:
            k += 1
        else:
            list_[k], list_[j] = list_[j], list_[k]
            j -= 1
    list_[i], list_[j] = list_[j], list_[i]
    return j


def _quicksort_2(list_, i, j):
    """
    Sort list_[i:j] by partitioning it around the first item,
    then recursing.

    @param list list_: list to sort
    @param int i: index to begin sorted slice
    @param int j: index to end sorted slice
    @rtype: None
    """
    if i < j:
        pivot = _partition_2(list_, i, j)
        _quicksort_2(list_, i, pivot)
        _quicksort_2(list_, pivot + 1, j)


def quicksort_2(list_):
    """
    Sort list list_ in non-decreasing order.

    @param list list_: list to sort
    @rtype: None
    """
    _quicksort_2(list_, 0, len(list_))
