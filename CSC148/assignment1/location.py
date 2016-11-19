class Location:
    def __init__(self, row, column):
        """Initialize a location.

        @type self: Location
        @type row: int
        @type column: int
        @rtype: None
        """
        self.m, self.n = row, column

    def __str__(self):
        """Return a string representation.

        @rtype: str
        >>> l = Location(4, 5)
        >>> print(l)
        (4, 5)
        """
        return '({}, {})'.format(self.m, self.n)

    def __eq__(self, other):
        """Return True if self equals other, and false otherwise.

        @rtype: bool

        >>> l1 = Location(4, 5)
        >>> l2 = Location(4, 5)
        >>> l1.__eq__(l2)
        True
        """
        return (isinstance(other, Location) and
                self.m == other.m and
                self.n == other.n)


def manhattan_distance(origin, destination):
    """Return the Manhattan distance between the origin and the destination.

    @type origin: Location
    @type destination: Location
    @rtype: int

    >>> manhattan_distance(Location(1, 7), Location(5, 4))
    7
    """
    return abs(origin.m - destination.m) + abs(origin.n - destination.n)


def deserialize_location(location_str):
    """Deserialize a location.

    @type location_str: str
        A location in the format 'row,col'
    @rtype: Location

    >>> l = deserialize_location('10,11')
    >>> print(l)
    (10, 11)
    """
    row = int(location_str.split(',')[0])
    col = int(location_str.split(',')[1])
    return Location(row, col)



if __name__ == '__main__':
    import doctest
    doctest.testmod()
