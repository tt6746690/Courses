
### A datum has 3 components:
	1. Id: a reference/alias to its address in memory
	2. Data type: determines what function can be performed on the value
	3. Value: the data
```python
5				# No stdout but still register in memory
x = 5   # x is an identifier for storage in memory
s = 'python is fun'   #s is also an identifier for storage in memory
```


### Immutable data type
    * Once stored in memory, it cannot change!
    * e.x. integers, strings, booleans, tuples
		* variables: refer to data but not data themselves
		* you can always change the reference in Python, but only sometime change the value, depending on data type
```python
x = 1			# x assignment 1 -> 1 stored in a specific memory location
id(x)			# fetch the memory location of x
y = x			# y is pointing to x and also pointing to 1
id(y)			# fetech the memory location of y, which is same as that of x
x = 2     # x assignemnt 2 -> 2 stored in a new memory location
id(x)     # returns a different memory location, BECAUSE number is IMMUTABLE
					# does not change the value stored for the data - since int is immutable -
						# but instead changes where x refers to
```

### Mutable data type
		* a data type that is not IMMUTABLE
		* e.x. lists, dictionaries, user-defined classes
``` python
x = [1,2,3]
x[0] = 4				# memory location does not change, SINCE list is mutable
										# - only MUTATES the value x refers to, but did not change where x refers to
x = [4,2,3]			# memory location change
```

### Aliasing
		* when two variables refer to the same data, they are aliases of each other
		* can be a source of bugs for mutable data types
``` python
x = [1, 2, 3]
y = x								# 'make y refer to the data that x is refering to'
x[0] = 1000000			# mutation of value of x --> also mutates the value of y, since they refer to the same data
```


### Two types of equality
		* ==	for equality of values in memory
		* is  for equality of addresses in memory, or, for comparing id

```python
x=5
z=5
y=x
x==y		# true
x==z 		# true
x is z
'''true: memory address of x and z is the same because python return
reference to any existing object with the same type and entry for immutable types'''
x is y	# true

#
x = [1,2,3]
z = [1,2,3]
x == z  # true 		since the list value is identical
x is z  # false 	Python allocates new memory for each mutable object written in memory
						# meaning that there are 2 instances of the list [1,2,3]
```

### Primitive data types
		* a basic type is a data type provided by the programming language as a basic building block
		* a built-in type is a data type for which the programming language provides build-in support
		* classic basic primitive data types
				* character 							`character`, `char`
				* Integer 								`integer`, `int`, `short`, `long`
				* FLoating-point number 	`float`, `double`
				* Fixed-point number
				* Boolean
				* Reference

### Data structure
		* a particular way of organizing data in a computer so that it can be used efficiently
		* examples
				* array or list
				* associative array (dictionary or map)
				* record (tuple)
				* set - a type of abstract data type
				* graph or tree - linked abstract data type composed of nodes
				* class - contains data field and methods, which operate on the contents of the record

### Abstract (advanced) data type
		* is a class of objects whose logical behavior is defined by a set of values and a set of operations
		* may be implemented by specific data types or data structures
		* integers are an ADT, defined as the values …, −2, −1, 0, 1, 2, …, and by the operations of addition, subtraction, multiplication, and division, together with greater than, less than, etc., which behave according to familiar mathematics
		* are often implemented as modules: the module's interface declares procedures that correspond to the ADT operations, sometimes with comments that describe the constraints. This information hiding strategy allows the implementation of the module to be changed without disturbing the client programs.

### Object
		* is a structured collections of data, bundled together with some functions that operate on this data

### Class
		* is a type of object, which allows the creation of many objects with same properties without redundant code
			* classes are _custom_ types
			* An object created from a class is an _instance_ of that class
		* each piece of data associated with a class is an __attribute__, which can be of different types: intergers, strings ...
		* the operatiosn associated with a class are __methods__ of a class
		* abstract (advanced) data type = data type
		* data type, abstract data type and advanced data type ARE Class
		* compared to primitive attributes
				* x = 5 has one attribute -- value

```python
x=5
type(x)    # <class 'int'>

class MyRectangle:
	'''
	a rectangle defined by its top-left coordinates width and height
	'''
```

### Constructor
	* a special method called `__init__` used to create instances of a class
	* the purpose is to initialize all attributes of the class
	* the `self` refers to the current object for which the methods is being called
			* every class method should start with `self` parameter
	* is called every time an instance is created
	* Instance variables should instead be initialized in `__init__` so that we can rely on all instances of a class having the same attributes.

```python
def __init__(self,x, y, width,height):			# def = definding a method   
	''' Initialize this MyRectangle.

	@type self: MyRectangle
	@type x: int
					the x coordinate of top-left corner of this rectangle
	@type y: int
					the y coordinate of top-left corner of this rectangle
	@type width: int
					the width of this rectangle
	@type heigt: int
					the height of this rectangle
	'''
	self.x = x
	self.y = y
	self.width = width
	self.height = height
```

		* use the name of the class as a function to call the class
		* the returned value is an instance of class

```python
r = MyRectangle(100, 200, 300, 400)				
	'''requires system to allocate location for 4 integers
				creates an instance of the class, or object.
				instances have different memory location
	'''
r.x		# 100
r.y		# 200
r.width		#300
r.height	#400
```

### Method to Move to Right
		* require the first parameter `self` to refer to the calling object.
		* Python uses the special "dot notation" syntax for calling methods to automatically bind the calling object to the `self` parameter

```python
def moveToRight(self, num):
	''' move the rectangle to the right by a number of pixels

	@type self: MyRectangle
	@type num: intergers
	@type: NoneType				change some property of class but does not return value
	'''
	this.x = self.x + num
```
```python
r.moveToRight(20)				# since class is mutable, memory addresses after change is still the same as previous
r.x			# 120
r.x = 100
r.x     # 100
```

		* we can access and even modify its attributes directly using "dot notation": the object, followed by a ., followed by the name of the attribute.

### Point Class
		* In two dimensions a point is two members (coordinates) that are treated collectively as a single objectPoints are often written in parentheses with a comma separating the coordinates. For example, (0,0) represents the origin, and (x,y) represents the point x units to the right and y units up from the origin. Some of the typical operations that one associates with points might be calculating the distance of point from the origin, or from another point, or adding a midpoint of two points, or asking if a point falls within a given rectangle or circle.

```python
class point:
	def __init__(self, x, y):
		self.x = x
		self.y = y

	def toOrigin(self):

	def midpoint(self):
	def within
```

### Methods vs functions
		* Methods are defined within a class definition. They are called on a particular instance of a class using the "dot notation".
		* the `str` methods must be called with a `str` argument using the dot notation
		* functions like len() and print() can be used on other types of object, therefore not a method of `str`

```python
s = 'hello'
s.upper()             # 'HELLO'
'bye'.find('y')       # 1

len('hello')					#5
print('hello')	  		# hello

```

> In a sense, methods are for behaviours which the creators of the class think are essential for anyone using the class, while functions are for behaviours which users of the class must implement themselves


### Python data model ([doc] (https://docs.python.org/3/reference/datamodel.html))

#### Standard type hierarchy

1. None
2. numbers.Number
	1. numbers.Integral
		+ integer (int)
		+ boolean (bool)
	2. numbers.Real (float)
	3. numbers.Complex (complex)
2. Sequences
	+ len() returns the number of items of a sequence
	+ supports slicing a[i:j] selects all items with index k such that i <= k < j
	+ distinguished according to their mutability
 	1. immutable
		1. Strings
		2. Tuples - arbitrary python objects, that are comma separated within `()`
		3. Bytes
	2. mutable
		+ can be changed after they are created
		1. Lists  - formed by placing a comma-separated list of expressions in `[]`
		2. Byte Arrays		- mutable counter part of Bytes
3. Set types
	+ represent unordered, finite sets of unique, immutable objects
	+ cannot be indexed by subscript, but can be iterated over, `len()` returns number of items in a set
	1. sets
	2. frozen sets
4. Mapping
	1. dictionaries - represent finite sets of objects indexed by nearly arbitrary values
5. Callable types
6. Modules
7. Custum classes
8. Class instances
