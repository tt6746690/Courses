from sort import *
import random
import timeit
import cProfile


def is_sorted(list_):
    """
    Return True iff list_ is in non-decreasing order.

    @param list list_: list to inspect
    @rtype bool:

    >>> is_sorted([1, 3, 5])
    True
    >>> is_sorted([3, 1, 5])
    False
    """
    for j in range(1, len(list_)):
        if list_[j - 1] > list_[j]:
            return False
    return True


def time_sort(list_, sorted_list, which_sort):
    """
    Sort list_ using function which_sort, time it and print the results,
    and ensure that the elements are the same as list sorted_list,
    which is a sorted version of L obtained by
    using the built-in sort.

    @param list list_: list to sort
    @param list sorted_list: list to compare result to
    @param (list)->None which_sort: sorting algorithm
    """

    sorter_name = which_sort.__name__  # the function's name as a string

    # Verify that the sorting algorithm works correctly!
    new_list = list_[:]
    which_sort(new_list)
    error_string = sorter_name + "did not sort"
    assert is_sorted(new_list) and new_list == sorted_list, error_string

    # The timeit module provides accurate timing of code in seconds, by
    # running the code a number of times and adding up the total time.
    t = timeit.timeit('{}({})'.format(sorter_name, list_),
                      'from sort import ' + sorter_name,
                      number=4) / 4   # takes average of 4 

    # Print information about the results so far, before all of the output
    # generated by the cProfile module.
    print("{} {} items in {:.6f}\n".format(sorter_name, len(list_), t))


def generate_data(n, sorted_=False, reversed_=False):
    """
    Return a list of n ints. If sorted_, the list should be nearly sorted:
    only a few elements are out of order. If sorted_ and reversed_, the list
    should be nearly sorted in reverse. The list should otherwise be
    shuffled (in random order).

    @param int n: number of ints in the list to be returned
    @param bool sorted_: indicates whether or not to sort
    @param bool reversed_: indicates whether or not to reverse
    @rtype: list[int]
    """
    list_ = [2 * j for j in range(n)]
    if sorted_:
        j = random.randrange(5, 11)
        while j < n // 2:
            list_[j], list_[-j] = list_[-j], list_[j]
            j += random.randrange(5, 11)
        if reversed_:
            list_.reverse()
    else:
        random.shuffle(list_)
    return list_


def profile_comparisons(n):
    """
    Run cProfile to identify bottlenecks in algorithms.

    @param int n: size of list to run algorithms on.
    @rtype: None
    """
    for algo in [selection_sort, insertion_sort_1, bubblesort_1,
                 mergesort_1, quicksort_1]:
        list_ = generate_data(n)
        name = algo.__name__
        print("=== profiling {} ===".format(name))
        cProfile.run("{}({})".format(algo.__name__, list_), sort='calls')


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    for algo_ in [selection_sort, insertion_sort_1, bubblesort_1,
                  mergesort_1, quicksort_1]:
        for i in range(1, 7):
            L = generate_data(i * 100)
            time_sort(L, sorted(L), algo_)
    for i in range(1, 7):
        L = generate_data(i * 100)
        time = timeit.timeit('{}.sort()'.format(L), number=100) / 100
        print("built-in sort {} items in {:.6f}\n".format(len(L), time))

    # uncomment this call to profile-comparisons, and edit the list of
    # algorithms to see call-by-call comparison
    profile_comparisons(1000)
