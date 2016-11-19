class Point:
    def __init__(self, x, y):
        '''Create'''

        self.x, self.y = float(x), float(y)


    def __eq__(self, other):
        '''Return whether this point is equivalent to other.

        @type self: Point
        @type other: Point | Any
        @rtype: boo1

        >>> p1 = Point(3, 4)
        >>> p2 = Point(4, 3)
        >>> p3 = Point(3. 0)

        4.0)
        >>> p1 == p2
        False
        >>> p1 == p3
        True
        '''
        return (type(self) == type(other) and
                    self.x == other.x and
                    self.y == other.y)

    def __str__(self):
        '''

        Return a user friendly string representation of Point self

        @type self: Point
        @rtype: str
        >>> p = Point(3,4)
        >>> print(p)
        (3.0, 4.0)
        '''
        return "({}, {})".format(self.x, self.y)

    def distance_to_origin(self):
        '''

        Return distance form Point self to (0.0, 0.0)

        @type self: Point
        @rtype: float

        >>> p = Point(3, 4)
        >>> p.distance_to_origin()
        5.0
        '''

        return (self.x ** 2 + self.y ** 2) ** (1/2)
