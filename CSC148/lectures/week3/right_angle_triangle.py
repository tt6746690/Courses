from shape import Shape
from point import Point


class RightAngleTriangle(Shape):
    """
    A right-angle-triangle Shape.
    """

    def __init__(self, corners):
        """
        Create RightAngleTriangle self with vertices corners.

        Overrides Shape.__init__

        Assume corners[0] is the 90 degree angle.

        @param RightAngleTriangle self: this RightAngleTriangle object
        @param list[Point] corners: corners that define this RightAngleTriangle
        @rtype: None

        >>> s = RightAngleTriangle([Point(0, 0), Point(1, 0), Point(0, 2)])
        """
        Shape.__init__(self, corners)

    def _set_area(self):
        """
        Set RightAngleTriangle self's area.

        Overrides Shape._set_area

        @type self: RightAngleTriangle
        @rtype: float

        >>> s = RightAngleTriangle([Point(0,0), Point(10,0), Point(0,20)])
        >>> s.area
        100.0
        """
        leg1 = self.corners[-1].distance(self.corners[0])
        leg2 = self.corners[0].distance(self.corners[1])
        self._area = (leg1 * leg2) / 2.0


if __name__ == '__main__':
    import doctest
    doctest.testmod()
