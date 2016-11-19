
####Running Time  

_selection sort_  __O(n^2)__
Repeatedly selecting the smallest remaining item and putting them to where it belongs.


_insertion sort_ __O(n^2)__  
Repeatedly inserting the next item where it belongs among the sorted item at the front of the list.

_bubble sort_ __O(n^2)__
Repeatedly scanning the entire list and swapping adjacent items that are out of order. Exit early by checking if the last two items are swapped increase performace

_merge sort_ __O(nlogn)__  
Splits list in half and sorts the two halves, and then merge the two sorted halves. Efficiency favors sorting that minimizes list update, since all subsequent items have to be shifted down.

_quick sort_ __O(nlogn)__  
partitioning list into two halves: those items less than the first item and those greater than or equal to first item
More efficient on random data, or with randomizer to pick the pivot. If an item that is about middle of the set of data, comparison is halved

_built in sort_ __O(nlogn)__
uses tim sort
