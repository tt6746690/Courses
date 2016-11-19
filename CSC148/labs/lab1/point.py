class Point:
    """"
    A two-dimensional point

    Attributes:
    ===========
    @type x: float
        units to right of origin
    @type y: float
        units above origin
    """

    def __init__(self, x, y):
        """
        Create a new Point self that is x units to the right,
        and y units above, the origin.

        Units are non-negative floats.

        @type self: Point
        @type x: float | int
        @type y: float | int
        @rtype: None
        """
        self.x, self.y = float(x), float(y)

    def __eq__(self, other):
        """
        Return whether this point is equivalent to other.

        @type self: Point
        @type other: Point | Any
        @rtype: bool

        >>> p1 = Point(3, 4)
        >>> p2 = Point(4, 3)
        >>> p3 = Point(3.0, 4.0)
        >>> p1 == p2
        False
        >>> p1 == p3
        True
        """
        return (type(self) == type(other) and
                self.x == other.x and
                self.y == other.y)

    def __str__(self):
        """
        Return a user-friendly string representation of
        Point self.

        @type self: Point
        @rtype: str

        >>> p = Point(3, 4)
        >>> print(p)
        (3.0, 4.0)
        """
        return "({}, {})".format(self.x, self.y)

    def distance_to_origin(self):
        """
        Return distance from Point self to (0.0, 0.0)

        @type self: Point
        @rtype: float

        >>> p = Point(3, 4)
        >>> p.distance_to_origin()
        5.0
        """
        return (self.x ** 2 + self.y ** 2) ** (1 / 2)

    def distance(self, other):
        """
        Return the distance between Point self and Point other.

        @type self: Point
        @type other: Point
        @rtype: float

        >>> print(Point(3, 4).distance(Point(6, 8)))
        5.0
        """
        return ((self.x - other.x)**2 + (self.y - other.y)**2) ** (1 / 2)

    def add(self, other):
        """
        Return sum of Point self and Point p2.

        @type self: Point
        @type other: Point
        @rtype: Point

        >>> p1 = Point(3, 4)
        >>> p2 = Point(4, 3)
        >>> print(p1.add(p2))
        (7.0, 7.0)
        """
        return Point(self.x + other.x, self.y + other.y)

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    p1 = Point(3, 4)
    p2 = Point(12, 13)
    p1.distance_to_origin()
    p1.add(p2)
    p1 == p2
