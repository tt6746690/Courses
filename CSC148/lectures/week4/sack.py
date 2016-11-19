import random
from container import Container

class Sack(Container):
    """
    A Sack with elements in no particular order.
    """

    def __init__(self):
        """
        Create a new, empty Sack self.

        @param Sack self: this Sack
        @rtype: None
        """
        self._data = []
        self._random = random.Random()

    def add(self, obj):
        """
        Add object obj to top of Sack self.

        @param Sack self: this Sack
        @param Any obj: object to place on Sack
        @rtype: None
        """
        self._data.append(obj)
        self._random.shuffle(self._data)

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
        return self._data.pop()

    def is_empty(self):
        """
        Return whether Sack self is empty.

        @param Sack self: this Sack
        @rtype: bool
        """
        return self._data == []


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    s = Sack()
    for i in range(10):
        s.add(i)
    while not s.is_empty():
        print(s.remove())
