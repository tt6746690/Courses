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
        self._contents = []

    def add(self, obj):
        """
        Add object obj to top of Stack self.

        @param Stack self: this Stack
        @param object obj: object to place on Stack
        @rtype: None
        """
        self._contents.append(obj)

    def __str__(self):
        """
        Return a str representation of Stack self.

        @param Stack self: this Stack
        @rtype: str

        >>> s = Stack()
        >>> s.add(3)
        >>> s.add(2)
        >>> print(s)
        [3, 2]
        """
        return str(self._contents)

    def __eq__(self, other):
        """
        Return whether Stack self is equivalent to other.

        @param Stack self: this Stack
        @param object|Stack other: object to compare to self.
        @rtype: bool

        >>> s1 = Stack()
        >>> s1.add(3)
        >>> s2 = Stack()
        >>> s2.add(3)
        >>> s1 == s2
        True
        """
        return (type(self) == type(other) and
                self._contents == other._contents)

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
        return self._contents.pop()

    def is_empty(self):
        """
        Return whether Stack self is empty.

        @param Stack self: this Stack
        @rtype: bool
        """
        return len(self._contents) == 0


if __name__ == "__main__":
    import doctest
    doctest.testmod()
