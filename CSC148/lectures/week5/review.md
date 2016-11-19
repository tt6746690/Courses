linked list
===========

### linked list
Python implemention of lists has some drawbacks: inserting and deleting requires shifting of many elements in the array. More precisely, in the language of Big-Oh then we saw that inserting and deleting at the front of a Python list takes O(n) time, where n is the length of the list. (In other words, these operations take time proportional to the length of the list, because every item in the list needs to be shifted by 1 spot.)

The advantage of linked list is that its much easier to remove and delete elements, because the elements don't need to be stored in memory in order; insertion and deletion involves only changing a few links.

> linked list is an alternate implementation of _existing_ ADT: the familiar list. Thus our goal is to create a new class that behaves exactly the same as existing lists, changing only what goes on behind the scenes.


```python
class _Node:
    """A node in a linked list.

    Note that this is considered a "private class", one
    which is only meant to be used in this module by the
    LinkedList class, but not by client code.

    === Attributes ===
    @type item: object
        The data stored in this node.
    @type next: _Node | None
        The next node in the list, or None if there are
        no more nodes in the list.
    """
    def __init__(self, item):
        """Initialize a new node storing <item>, with no next node.

        @type self: _Node
        @type item: object
        @rtype: None
        """
        self.item = item
        self.next = None  # Initially pointing to nothing


class LinkedList:
    """A linked list implementation of the List ADT.
    """
    # === Private Attributes ===
    # @type first: _Node | None
    #    The first node in the list, or None if the list is empty.

    def __init__(self, items):
        """Initialize a new linked list containing the given items.

        The first node in the linked list contains the first item
        in <items>.

        @type self: LinkedList
        @type items: list
        @rtype: None
        """
        if len(items) == 0:  # No items, and an empty list!
            self._first = None
        else:
            self._first = _Node(items[0])
            current_node = self._first
            for item in items[1:]:
                current_node.next = _Node(item)
                current_node = current_node.next
```

+ The most common mistake students make is confusing an individual node object with the item stored in the list. Because each item needs to "know" where the next item in the list is, we really do need a Node class to store both an item and a reference to the next node in the list.

```python
>>> lst = LinkedList([1, 2, 3])
>>> lst._first
<_Node object at 0x000000000322D5F8>
>>> lst._first.item
1
>>> lst._first.next.next
<_Node object at 0x000000000322D198>
>>> lst._first.next.next.item
3
```

### Traversing a Linked List

```python
i = 0
while i < len(my_list):
    # Do something with my_list[i], e.g.:
    print(my_list[i])
    # Increment i
    i = i + 1
```
