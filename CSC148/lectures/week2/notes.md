LECTURE 2
=========

### Key terms
<br>
Class: (abstract/advanecd/compound) data type
  + It models a thing or concept (let's name it object), based on its common attributes and actions
  + i.e. it _bundles_ together attributes and methods that are relevant to each instance of those object  
  - __can you define a chair?__    

In object oriented world, _objects_ are often __active__ agents
  + i.e. concrete and discrete actions are invoked on the objects
  + an instance of a class  
  - __can you show me a chair?__

### Encapsulation
  + Designing classes by separating the public interface of the class from the implementation details
  + advantage
    - after our initial implementation, we can feel free to modify it (e.g., add new features or make it more efficient) without disturbing the public interface, and rest assured that this doesn't affect other code that might be using this class
    - the public interface should be as "small" as possible: we want to restrict precisely what others can do with our classes, to prevent others from abusing them.

### Scope
 + Attributes can be accessed outside of class via the dot notation
 + __public__ and __private__
 + In Python, all attributes are technically public, in that nothing in the language prevents you from ever accessing an attribute.
 + Naming convention
 > all private attributes, and methods, must start with an underscore.

 + Because Python won't enforce privateness, we must trust the authors of the client code not to access any such labelled attributes.

### Representation Invariant
  +  a boolean (true/false) statement we make about the attributes of a class which must always be true for every instance of that class.
    - ex. `@type age: int` means "the age attribute of the object must always be an int."
  + a generalization of a type annotation where the class designer can use English to explicitly state any restriction she wants.
    - ex. `age is always positive`
  + plays an important role in _restricting_ the behaviour of methods
    - At the beginning of the method body (i.e., right when the method is called), you can always assume that all of the representation invariants are satisfied. This is a special precondition on all methods except the constructor.
    - At the end of the method (i.e., right before the method returns), it is your responsibility to ensure that all of the representation invariants are satisfied. This is a special postcondition on all methods, including the constructor.

### An Example
```python
class Restaurant:
    """A restaurant in a recommendation system.

    A food truck could be a restaurant too!

    === Public Attributes ===
    @type name: str
        The name of the restaurant.
    @type location: str
        The location of the restaurant.
    """
    # === Private Attributes ===
    # @type _dishes: dict[str, list[str]]
    #     The dishes served at this restaurant, mapping names of dishes
    #     to corresponding dietary restrictions.
    #
    # === Representation Invariants ===
    # - There can be at most 100 dishes in the Restaurant.
    # - The location has to be in Toronto.
    def add_dish(self, dish, restrictions):
        """Add a new dish to the menu of this restaurant.

        Do nothing if there are 100 dishes already at this restaurant.

        @type self: Restaurant
        @type dish: str
        @type restrictions: list[str]
        @rtype: None
        >>> dennys = Restaurant('Dennys', '10 Dundas St.')
        >>> dennys.add_dish('salad', ['vegetarian'])
        >>> dennys.print_dishes()
        salad: vegetarian
        """
        pass

    def _add_dish_helper(self, blah, blah2):
        """
        """
        pass

if __name__ == '__main__':          #  $ python Restaurant.py will call this method -> a tester method
    dennys = Restaurant('Dennys', '10 Dundas St.')
    dennys.add_dish('burger', [])
    dennys.add_dish('salad', ['vegetarian'])
    dishes = dennys.get_all('vegetarian')
    print(dishes)

    print(dennys.name)
    print(dennys.location)
    dennys.name = 'Davids'
```

###Points
#### Dot notation misconception
__When using dot notation, make sure the name on the left refers to an object, not a class!__

#### Constructor again
Wrong to think that constructor acts as a one-to-one mapping between parameters and attributes.
  ```python
  class Restaurant:
    def __init__(self, name, location):
        self.name = name
        self.location = location
  ```
Rather, constructor is meant to _initialize all attributes for the object_
  ```python
  class Restaurant:                       # look at all public and private attributes
    def __init__(self, name, location):
        self.name =
        self.location =
        self._dishes =                    # PRIVATE
  ```
#### 'Private' in Python
the _underscore_ convention does not stop the python program from accessing those attributes/methods
  ```
  >>> dennys = Restaurant('Dennys', '123 fake St.')
  >>> dennys._dishes
  {}
  ```


#### Composition
A class is almost never defined and used in isolation; it is much more often the case that it belongs to a large collection of classes, which are all related to each other in various ways. One fundamental type of relationship between two classes is when one has an attribute which is an instance of the other:
```python
class A:
    def __init__(self, number):
        self.my_b = B(number)

class B:
    def __init__(self, x):
        self._num = x
```

+ here `my_b` has attributes `B`, we say ' A has B'

####Reinforce __encapsulation__
  + so that the private attributes of the class are made private to hide and protect from other code
  + Makeshift solutions that is not pythonic
    - declare setters in constructor without initializing attributes
    - assign getters/setters but not an intuitive interface
  + python __properties__
    - two functions with the same name but different parameters are allowed due to _decoration_
  ```python
  class P:

    def __init__(self,x):
        self.x = x      # here x.setter is called to check the limits
# othwise  self.__x = x to set __x without calling x.setter

    @property           # decorated with @property
    def x(self):
        return self.__x

    @x.setter           # decorated with @foo.setter
    def x(self, x):
        if x < 0:
            self.__x = 0
        elif x > 1000:
            self.__x = 1000
        else:
            self.__x = x

  ```


### Design Roadmap
<br>
0. Read project __specification__
  + frequent _nouns_ may be good candidate for classes
  + _properties_ of such nouns may be good candidates for _attributes_, or information associated with the _nouns_
  + _actions_, as verbs, of such nouns may be good candidates for _methods_
  + there may be __special__ methods that are relevant to many classes

> In two dimensions, a __point__ is two numbers (_coordinates_) that are treated collectively as a single object. __Points__ are often written in parentheses with a comma separating the coordinates. For example, _(0, 0)_ represents the origin, and _(x, y)_ represents the point _x_ units to the right and _y_ units up from the origin. Some of the typical operations that one associates with __points__ might be __calculating the distance__ of a point from the origin, or from another __point__, or __finding__ a midpoint of two __points__, or __asking__ if a point falls within a given rectangle or circle.

__points__ are the __nouns__  
_x, y, coordinates_ are the properties
__calculate, find, ask__ there are actions  

1. Design a class __API__:
  1. choose a class __name__ and write a brief __description__ in the class docstring
  > __docstring__ is a string literal specified in source code that is used, like a comment, to document a specific segment of code.

  2. write some examples of client code that uses your class
    - best if put in ` if __name__ == __main__: `, the main block
  3. decide what services your class should provide as public methods, for each method declare an API
    - examples
    - type contract
    - header
    - description
  4. decide which _public_ attributes your class should provide without calling a method, list them in the class docstring
    - Treat attribute as public until you have a good reason until you have a good reason not to
    - good reasons:
      1. An attribute with complex restrictions on its value. If client code were to assign a value to the attribute directly, it might inadvertently violate the restriction. If instead it is required to call a method to change the value, the method implementation can enforce any restriction.
      2. An attribute that represents data from the domain in a complex way. (We’ll learn some fairly complex data structures this term.) By expecting client code to access the information through a method call, we spare the client code from having to be aware of the complex details, and we also avoid the problem of client code accidentally messing up important properties of the data structure.
    5. Internal Private attributes
    Define the type, name, and description of each of the internal attributes. Put this in a class comment (using the hash symbol) below the class docstring.
      ```python
      # === Private Attributes ===
      # @type _scheme: dict[str, int]
      # The marking scheme for this course. Each key is an element of the
      # course, and its value is the weight of that element towards the
      # course grade.
      # @type _grades: dict[str, dict[str, int]]
      # The grades earned so far in this course. Each key is a student
      # ID and its value is a dict mapping a course element to the student’s
      # mark on that element. If a student did not submit that element,
      # it does not appear as a key in the student’s dict. If, however,
      # they earned a grade of zero, this is recorded as any other grade
      # would be.

      ```
    6. Representation invariants.
    Add a section to your internal class comment containing “invariants” that involve your attributes: things that must always be true
    ```python
    # === Representation Invariants ===
# - The sum of all weights in self._scheme must be 100.
# - Each key in every student’s dict of grades must be an element of the
# course grading scheme, i.e., must occur as a key in self._scheme.
    ```



2. Implement the class:
  + body of special methods,
    - \__init\__: called after instance has been created
    - \__del\__ : called when instance is about to be destroyed
    - \__eq\__  :
    - \__str\__ : converts object to string presentation
  + body of other methods
    - ex. distance
  + testing

```python
a = 5
b = 6  
a == b    # false

p1 = point(15, 60)
p2 = point(15, 61)
p1 == p2    # __eq__ defined to evaluate instances of the class -> returns false
            # both the x and y coordinates are compared
print(p1)   # __str__ defined to provide informal string representation of class
```

### Point Class

```python
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
        >>> p3 = Point(3.0, 4.0)
        >>> p1 == p2
        False
        >>> p1 == p3
        Trues
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
```

To test it

```python
p1 = Point(5,6)
p2 = Point(5,7)
p3 = Point(5,6)
p1 == p2    # false
p1 == p3    # true
p1 is p3    # false memory location for instances

print(p1)   # returns the memory location of the object if __str__ not specified
print(p1)   # (5, 6) depending on the __str__ function  

```

### Rational Class
  + In Python, fractions are converted to decimals and rounded, rather than kept an exact form
>Rational numbers are ration of two integers _p/q_, where _p_ is _numerator_ _q_ is _denominator_. The _denominator_ _q_ is non-zero. Operations are rationals include __addition__, __multiplication__, and __comparison__

+ defined the concept of rational class. PAINTING....
+ Python provides special methods:
  - \__ne\__
  - \__lt\__
  - \__le\__
  - \__gt\__
  - \__ge\__
  - \__eq\__
+ Other special methods: \__init\__, \__add\__, \__mul\__, \__str\__

#### API: class definition & constructor

```python
class Rational:
  ''' a rational number '''

  def __init__(self, num, denom=1):
    ''' create new Rational self with numerator num and denominator denom --- denom must not be 0.

    @type self: Rational
    @type num: int
    @type denom: int
    @rtype: None
    '''
    self.num, self.denom = int(num), int(denom) # int() convert value to plain integer. Coercion may be necessary
```
#### API: other methods
  + examples must be common and broad


```python
  def __eq__(self, other):
    '''
    Return whether Rational self is equivalent to other.
    @type self: Rational
    @type other: Rational | Any
    @rtype: bool

    >>> r1 = Rational(3, 5)
    >>> r2 = Rational(6, 10)
    >>> r3 = Rational(4, 7)
    >>> r1 == r2   
    True
    >>> r1.equals(3/5)
    True
    '''
    return (type(self) == type(others) and
          self.num * other.denom == self.denom * other.num)

  def __str___(self):
    '''
    Return a user-friendly string representation of Rational itself

    @type self:Rational
    @rtype: str

    >>> print(Rational(3, 5))
    3/5
    '''
    return "{} / {}".format(self.num, self.denom)

  def __lt__(self, other):
    return self.num

  def __add__(self, other):
    return

```

#### Design more class and implement them
#### What if the denominator is 0?
  + needs to manage attributes
    - sets pre- and post- conditions
    - raise exceptions
  + how to enforce it?

####Getters, setters, and properties

```python
def _get_num(self):
  '''
  Return numerator num,

  @type self: Rational
  @rtype: int

  >>> Rational(3,4)._get_num()
  3
  '''


def _set_num(self):
  '''
  set numerator of rational self to num

  @type self: Rational
  @rtype num: init
  @r
  num = property(_get_num, _set_num)
  '''
def _get_denom(self):
def _set_denom(self, denom):
  '''
  Set denominator of Rational self to denom

  @type self: Rational
  @type denom: int
  @rtype: None

  '''
  if denom == 0;
    raise Exception("Zero denominator!")
  else:
    self._denom = int(denom)    # self._denom is a convention for marking PRIVATE

denom = property(_get_denom, _set_denom)

```


### OOP Features
1. Composition and Inheritance
  + a rectangle has some vertices
  + a triangle has some vertices
  + a triangle is a shape
  + a rectangle is a shape  

__has_a__ vs __is_a__ relationship
2. A shape has a perimeter
  + a rectangle can inherit the perimeter from a shape
  + a triangle too
3. A shape has an area
  + Can be area of a rectangle or triangle abstracted to shape level?
