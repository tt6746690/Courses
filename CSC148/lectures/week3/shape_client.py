# demonstrate code that uses Shape
# without knowing whether it is Square or Triangle
from shape import Shape
from point import Point


def shape_place(mylist):
    """
    Place Shapes in mylist and draw them.

    @type mylist: list[Shape]
    @rtype: None
    """
    for s in mylist:
        for p in [Point(100, 0), Point(100, 200), Point(-200, -100)]:
            s.move_by(p)
            s.draw()

if __name__ == '__main__':
    # code that uses shape_place with specific Shapes
    from square import Square
    from right_angle_triangle import RightAngleTriangle
    L = [Square([Point(0, 0), Point(40, 0), Point(40, 40), Point(0, 40)]),
         RightAngleTriangle([Point(-100, 0), Point(0, 0), Point(-100, 50)])]
    shape_place(L)
    from time import sleep
    sleep(5)
