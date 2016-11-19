import turtle
# constant for choosing color by number
COLOR = ("red", "green", "blue")


def tree_burst(level, base, turtle_):
    """
    Draw a ternary tree of height level, edge length base, using turtle_.

    @param int level: how many levels of recursion to use
    @param int base: pixels to draw base shape
    @param Turtle turtle_: drawing turtle
    @rtype: None
    """
    if level == 0:
        pass
    else:
        turtle_list = []  # place to keep 3 turtles
        for h in range(3):
            # store new turtle
            turtle_list.append(turtle_.clone())
            # set colour, using weird spelling
            turtle_list[h].color(COLOR[h])
            # set direction
            turtle_list[h].setheading(120 * h)
            # draw a little
            turtle_list[h].forward(base)
            # 1/2 size version
            tree_burst(level - 1, base / 2, turtle_list[h])


if __name__ == "__main__":
    import time
    T = turtle.Turtle()
    T.color("red")
    T.speed("slow")
    # hide the tail...
    T.hideturtle()
    tree_burst(4, 128, T)
    time.sleep(5)
