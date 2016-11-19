from container import Container

class Stack(Container):
    """Stack implementation.

    Stores data in a first-in, last-out order.
    When removing an item from the stack, the most recently-added
    item is the one that is removed.
    """
    # === Private Attributes ===
    # @type _items: list
    #     The items stored in the stack.
    #     The back of the list will represent the top of the stack.
    def __init__(self):
        """Create a new empty stack.

        @type self: Stack
        @rtype: None
        """
        self._items = []

    def is_empty(self):
        """Return whether this stack contains no items.

        @type self: Stack
        @rtype: bool
        """
        return len(self._items) == 0

    def push(self, item):
        """Add a new element to the top of this stack.

        (Implementation of Container's 'add' method.)

        @type self: Stack
        @type item: object
        @rtype: None
        """
        self._items.append(item)

     def pop(self):
        """Remove and return the element at the top of this stack.

        Do nothing if <self> is empty.

        @type self: Stack
        @rtype: object

        >>> s = Stack()
        >>> s.add(5)
        >>> s.add(7)
        >>> s.remove()
        7
        """
        if not self.is_empty():
            return self._items.pop()



if __name__ == "__main__":
    import doctest
    doctest.testmod()
