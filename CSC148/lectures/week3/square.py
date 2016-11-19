from shape import Shape
from point import Point


class Square(Shape):
    """
    A square Shape.
    """

    def __init__(self, corners):
        """
        Create Square self with vertices corners.

        Assume all sides are equal and corners are square.

        Extended from Shape.

        @param Square self: this Square object
        @param list[Point] corners: corners that define this Square
        @rtype: None

        >>> s = Square([Point(0, 0), Point(1, 0), Point(1, 1), Point(0, 0)])
        """
        Shape.__init__(self, corners)

    def _set_area(self):
        """
        Set Square self's area.

        Overrides Shape._set_area

        @type self: Square
        @rtype: float

        >>> s = Square([Point(0,0), Point(10,0), Point(10,10), Point(0,10)])
        >>> s.area
        100.0
        """
        self._area = self.corners[-1].distance(self.corners[0])**2


if __name__ == '__main__':
    import doctest
    doctest.testmod()
    s = Square([Point(0, 0)])
    # Pycharm flags these,
    # as it should.
    # print(s.corners + "three")
    # print(s.area + "three")
