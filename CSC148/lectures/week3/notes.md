
Object oriented programming

### OPP Features
Composition and Inheritance
+ a rectangle has some vertices -> vertices is a subclass of rectangle
+ a triangle has some vertices
+ a triangle is a shape -> shape is a super class of rectangle
+ a rectangle is a shape     

_has_a_ vs _is_a_ relationship    

A shape has a perimeter
  + a rectangle can inherit the perimeter from a shape
  + a triangle too    

A shape has an area
  + can be area of a rectangle or triangle abstracted to the shape level  

- **
### More specific example
Assume you are reading a project specification which is about defining, drwing, and animating some geometrical shapes...
Assume we are only concerned with only two shapes: __square__ and __right angled triangles__

 Object   
  |--living creatures   
  |--non living creates

#### Square
Squares have four vertices (__corners__), have a __perimeter__,and a __area__,can _move_ themselvesby adding an offset point to each corner, and can _draw_ themselves

####Right angled triangle
Right angled triangles have three vertices (__corners__), have a __perimeter__, an __area__, can _move_ themselves by adding an offset point to each corner, and can _draw_ themselves

- **
### Abstraction
we need to define two classes: Square and RightAngleTriangle
we also need to define subclasses: Shape

#### Shape class
develop the common features into an _abstract class_ __Shape__ to avoid redundancy
  + Points, perimeter, area

remember to follow the class design recipe
  + read project specification
  + define class with a short description and client code examples to show how to use it
  + develop API for all methods including the special ones, \__eq\__, \__str\__,...
    + follow the _function design recipe_, just dont implement it until your API is complete
    + then, implement it


```python
from point import Point
from turtle import Turtle

class Shape:
  '''
  A Shape shape that can draw itself, move, and report area and perimeter

  === Attributes ===
  @param list[Point] corners: corners that define this shape
  @param float area: area of this Shape
  @param float perimeter: perimeter of this Shape
  '''

  def __init__(self):
    '''
    Create a new Shape self with defined by its corners.

    @param Shape self: this Shape object
    @param list[Point] corners: corners that define this Shape
    @rtype: None
    '''
    # shallow copy of corners
    self.corners = corners[:]
    self._turtle = Turtle()
    self._set_perimeter()
    self._set_area()

  def __eq__(self, other):

  def __str__(self):

  # private methods user not supposed to use
  def _set_perimeter(self):

  def _get_perimeter(self):

  def _set_area(self):

  def _get_area(self, offset_point):

  def move_by(self, offset_point):

  def draw(self):
```

#### Shape implementation
So far we implemented the common features of Square

```python
  def _set_area(self):
    '''
    Set the area of Shape self to the Shape of its sides
    @type self: Shape
    @rtype: None
    '''
    self._area = -1.0
    raise NotImplementedError("Set area in subclass!!!")

  def _get_area(self):
    '''
    return the area of Shape self.

    @type self: Shape
    @rypte: float

    >>> Shape([Point(1, 1), Point(2, 1), Point(2, 2), Point(1, 2)]).area
    1.0
    '''
    return self._area

  # area is immutable --- no setter method in property
  area = property(_get_area)

```
- **
### Inheritance
Develop a super class that is abstract
  + it defines the common features of subclasses
  + but it's missing some features that must be defined in subclasses
  + exists for inheritance only

__Square__ and __RightAngleTriangle__ are two subclass examples of _Shape_ from which inheriting the identical features
```python
class Square(Shape):

class RightAngleTriangle(Shape):
```

Develop __Square__ and __RightAngleTriangle__ by following recipes

```python
from shape import Shape
  '''
  A square Shape
  '''
  def __init__(self):
    '''
    create Square self with vertices corners
    Assume all side are equal and corners are square

    Extend from shape

    @param Square self: this Square object
    @param list[Point] corners: corners that  define this Square
    @rtype: None

    >>> s = Square([Point(0,0), Point(1,0), Point(1,1), Point(0,1)])
    '''
    Shape.__init__(self, corners)

  def _set_area(self):
    '''
    Set Square self's area
    Overrides: Shape._set_area

    @type self: Square
    @rtype: None

    >>> s = Square([Point(1, 0), Point(10, 0), Point(10, 10), Point(0, 10)])
    >>> s.area
    100.0

    '''
    self._area = self.corners[-1].distance(self.corners[0])**2

  if __name__ = '__main__':
    import doctest
    doctest.testmod()
    s = Square(Point(0, 0))
```

- **
### Discussion
+ A __Shape__ is a composition of some __Points__
+ __Square__ and __RightAngleTriangle__ inherit from Shape
  + inherit the perimeter, area, move, and draw from __Shape__
  + they slightly extend the constructor of Shape
  + they override the \_set_area of Shape
+ The client code can use subclasses Square and RightAngleTriangle, to construct different objects (instances), get their perimeter and area, move them around, and draw them
+ Other subclasses can also inherit from that class

### Final note
+ Don't need to maintain documentation in two places  e.g. superclass and subclass
  - inherited methods, attributes --> no need to document again
  - extended methods   -->  document that they are extended and how
  - overriden methods, attributes  --> document that they are overriden and how


- **
### Stack
A __stack__ or __LIFO__ (last in first out) is an abstract data containing item of various sorts. New items are _added on to the top_ of the stack, items may only be _removed from the top_ of the stack with _push_ and _pop_ respectively. _push_ or _pop_ may only occur at one end of a stack. It's a mistake to try to remove an item from an empty stack, so we need to know _if it is empty_. We can tell _how big_ a stack is. The lower elements on the stack have been on the stack the longest.
__stack__ is used in computer operating systems and compliers

```python
class Stack:
    """
    Last-in, first-out (LIFO) stack.
    """

    def __init__(self):
        """
        Create a new, empty Stack self.

        @param Stack self: this stack
        @rtype: None
        """
        pass

    def add(self, obj):
        """
        Add object obj to top of Stack self.

        @param Stack self: this Stack
        @param Any obj: object to place on Stack
        @rtype: None
        """
        pass

    def remove(self):
        """
        Remove and return top element of Stack self.

        Assume Stack self is not empty.

        @param Stack self: this Stack
        @rtype: object

        >>> s = Stack()
        >>> s.add(5)
        >>> s.add(7)
        >>> s.remove()
        7
        """
        pass

    def is_empty(self):
        """
        Return whether Stack self is empty.

        @param Stack self: this Stack
        @rtype: bool
        """
        pass


if __name__ == "__main__":
    import doctest
    doctest.testmod()

```

### Sack
Sack contains items of various sorts. New items are added on to a random place in the sack, so the order items are removed from the sack is completely unpredictable. It's a mistake to try to remove an item from an empty sack, so we need to know if it is empty. We can tell how big a sack is.

```python
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
        >>> s.removes()
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


```

### Container
