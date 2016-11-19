class Stack:
    """
    Last-in, first-out (LIFO) stack.
    """

    def __init__(self):
        """
        Create a new, empty Stack self.

        @param Stack self: this stack
        @rtype: None
        """
        pass

    def add(self, obj):
        """
        Add object obj to top of Stack self.

        @param Stack self: this Stack
        @param Any obj: object to place on Stack
        @rtype: None
        """
        pass

    def remove(self):
        """
        Remove and return top element of Stack self.

        Assume Stack self is not empty.

        @param Stack self: this Stack
        @rtype: object

        >>> s = Stack()
        >>> s.add(5)
        >>> s.add(7)
        >>> s.remove()
        7
        """
        pass

    def is_empty(self):
        """
        Return whether Stack self is empty.

        @param Stack self: this Stack
        @rtype: bool
        """
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
