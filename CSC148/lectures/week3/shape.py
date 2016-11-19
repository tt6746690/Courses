from point import Point
from turtle import Turtle


class Shape:
    """
    A Shape shape that can draw itself, move, and
    report area and perimeter.

    === Attributes ===
    @param list[Point] corners: corners that define this Shape
    @param float area: area of this Shape
    @param float perimeter: perimeter of this Shape
    """
    def __init__(self, corners):
        """
        Create a new Shape self with defined by its corners.

        @param Shape self: this Shape object
        @param list[Point] corners: corners that define this Shape
        @rtype: None
        """
        # shallow copy of corners
        self.corners = corners[:]           # use [:] to make a copy, otherwise merely redirecting
        self._turtle = Turtle()                 # should be cautious since list is mutable
        self._set_perimeter()
        self._set_area()

    def __eq__(self, other):
        """
        Return whether Shape self is equivalent to other.

        @param Shape self: this Shape object
        @param Shape|Any other: object to be compared to
        @rtype: bool
        """
        return (type(self) == type(other) and
                self.corners == other.corners and
                self.area == other.area and
                self.perimeter == other.perimeter)

    def __str__(self):
        """
        Return an informative string describing self.

        @param Shape self: this Shape object
        @rtype: str

        >>> print(Shape([Point(0, 0), Point(1, 0), Point(0, 1)]))
        Shape([(0, 0), (1, 0), (0, 1)])
        """
        point_list = ", ".join([str(p) for p in self.corners])
        # classes have a __name__ string
        return type(self).__name__ + "([{}])".format(point_list)

    def _set_perimeter(self):
        # """
        # Set Shape self's perimeter to the sum of the distances
        # between corners.
        #
        # @type self: Shape
        # @rtype: None
        # """
        distance_list = []
        for i in range(len(self.corners)):
            distance_list.append(self.corners[i].distance(
                    self.corners[i - 1]))
        self._perimeter = sum(distance_list)

    def _get_perimeter(self):
        # """
        # Return the perimeter of this Shape
        #
        # @type self: Shape
        # @rtype: float
        # """
        return self._perimeter

    # perimeter is immutable --- no setter method in property
    perimeter = property(_get_perimeter)

    def _set_area(self):
        # """
        # Set the area of Shape self to the Shape of
        # its sides.
        #
        # @type self: Shape
        # @rtype: None
        # """
        # impossible area to satisfy PyCharm...
        self._area = -1.0
        raise NotImplementedError("Set area in subclass!!!")

    def _get_area(self):
        # """
        # Return the area of Shape self.
        #
        # @type self: Shape
        # @rtype: float
        #
        # >>> Shape([Point(1, 1), Point(2, 1), Point(2, 2), Point(1, 2)]).area
        # 1.0
        # """
        return self._area

    # area is immutable --- no setter method in property
    area = property(_get_area)

    def move_by(self, offset_point):
        """
        Move Shape self to a new position by adding
        Point offset_point to each corner.

        @type self: Shape
        @type offset_point: Point
        @rtype: None
        """
        # list comprehension
        # see course web page, week 2
        self.corners = [c.add(offset_point) for c in self.corners]
        # equivalent to...
        # new_corners = []
        #  for c in self.corners:
        #     new_corners.append(c.add(offset_point))

    def draw(self):
        """
        Draw Shape self.

        @type self: Shape
        @rtype: None
        """
        self._turtle.penup()
        self._turtle.goto(self.corners[-1].x, self.corners[-1].y)
        self._turtle.pendown()
        for i in range(len(self.corners)):
            self._turtle.goto(self.corners[i].x, self.corners[i].y)
        self._turtle.penup()
        self._turtle.goto(0, 0)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    s = Shape([Point(0, 0)])
    # Pycharm flags this,
    # as it should
    # print(s.area + "three")
