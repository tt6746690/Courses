class Sack:
    """
    A Sack with elements in no particular order.
    """

    def __init__(self):
        """
        Create a new, empty Sack self.

        @param Sack self: this Sack
        @rtype: None
        """
        pass

    def add(self, obj):
        """
        Add object obj to top of Sack self.

        @param Sack self: this Sack
        @param Any obj: object to place on Sack
        @rtype: None
        """
        pass

    def remove(self):
        """
        Remove and return some random element of Sack self.

        Assume Sack self is not empty.

        @param Sack self: this Sack
        @rtype: object

        >>> s = Sack()
        >>> s.add(7)
        >>> s.remove()
        7
        """
        pass

    def is_empty(self):
        """
        Return whether Sack self is empty.

        @param Sack self: this Sack
        @rtype: bool
        """
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()
