
Python
======

### Atom trick
__ALT  [0-9]__     - move to tabs in sequential order  
__ALT  B__         - move to the beginning of a word  
__CTRL [ or ]__    - outdent / indent selected line  
__CTRL UP/DOWN__   - move entire line up or down  
__CTRL \__         - toggle tree  
__CTRL L__         - select current line  
__CTRL N__         - new file  
__CTRL P__         - file finder  
__CTRL PgUp/PgDn__ - view previous/next item (tab)  
__CTRL SHIFT C__   - copy to clipboard current _path_  
__CTRL SHIFT F__   - find in project  
__CTRL SHIFT K__   - delete selected line  
__CTRL SHIFT M__   - show markdown  
__PgUp/PgDn__      - move up/down pages  

- **
### Python is dynamically typed
The type of variable is interpreted at runtime (instead of at compile time)

  ```python
  x = 1
  x = 'abc'    # reassign x to a string
  print(x)     # prints 'abc'
  ```
_One_ variable can change type during runtime
- **
### Blueprint of a python file
```python
from random import randint    # import name from other modules
from math import cos

def my_function(arg):         # define functions and classes
  ...
  return answer

class MyClass:


if __name__ == '__main__':    # main block
  my_variable = 21*2
```
- **
### Python runs interactively in the console

```python
>>> 42
42
>>> 1+2
3
```
- **
### Variable

Variable refers to an __obejct__ of some __type__.  
Basic data type:
  + Intergers: __int__
  ```python
  >>> integer = 23
  23  
  ```
  + Floating-point numbers: __float__
  ```python
  >>> pi =  3.14159             
  >>> pi*2.0
  12.56636             # compared to >>> 5//2 = 2
  ```
    + operators  
      \*,  /,  %,  +,  -,  \**,  //
    + 'shortcut' operators   
    ``` python
    x = x + 1 -> x += 1
    ```
  + Boolean values: __bool__
  ``` python
  >>> passed = False
  >>> not passed
  True
  >>> 5 > 4     # comparison returns bool
  True
  >>> 5 and 4   # this can bite you !!
  4
  ```
    + operators  
    _and_,   _or_,   _not_
  + None
  ```python
  >>> x = None
  >>> print(x)
  None
  >>> x         # no output
  ```
- **
### String
#### strings are a _list of characters_: __str__  
```python   
>>> s = 'hello, world!'  
>>> s[0]        # index start with 1
'h'
```
#### slices returns substrings
```python
>>> s[1:5]          # slice from 1 (inclusive) to 5 (exclusive)
'ello'
>>> s[:3]           # from start to 3 (exclusive)
'hel'
>>> s[9:]           # from 9 (inclusive) to the end
'rld!'
>>> s[:-2]          # from the start to the second last (exclusive)
'Hello, worl'
```
#### concatenation
```python
>>> a = 'wtf'
>>> b = ' is this'
>>> a + b             # evaluates to a new string
'wtf is this'
```
#### len
```python
>>> len(a)
3
```
#### Other useful methods
```python
>>> s = 'Orion'
>>> s.startswith('Or')
True
>>> s.endswith('ion')
True
>>> 'rio' in s
True
>>> s.lower()
'orion'
>>> s.index('i')      # use help(str.index) to see how it works
2         
```
#### string formatting (str.format)
```python
>>> n = 99
>>> where = 'on the wall'
>>> '{} bottles of beer {}'.format(n, where)
99 bottles of beer on the wall
```   
`{}` are replaced by arguments to format  

- **
### Standard Input/Output
```python
>>> name = input('Type you name here: ')    # read keyboard input
Type your name here: Orion
>>> name
'Orion'
>>> print('Hello ' + name)          # generate stdout
Hello Orion    
>>> print('Hello {}'.format(name))
Hello Orion
```
- **
#### Converting between types  
sanitize user input using _int()_, _str()_, _float()_, _bool()_
```python
>>> float('3.4')
3.4
>>> int(9/5)      # truncates
1
>>> str(3.4)
'3.4'
>>> '{:.4f}'.format(3.14159265358)      # .4f  specifies decimal places
'3.1416'
>>> int('fishy')
ValueError
>>> int('3.0')        # since string is a float
ValueError   
```
- **

### Sequences
> the [MIGHTY] list  
  the (humbl) tuple


#### List
1. List is an __mutable__ sequence of __any datastructure__
```python
>>> colors = ['red', 'blue', 'yellow']
>>> anyThing = [[], 2, 'wtf', colors]
>>> copyList = list(anyThing)
>>> colors[0]       # index
'red'
>>> colors[2:]      # slice
'yellow'
```        
2. We can change, add, remove element from a list

  ```python
  >>> marks = [90, None, 62, 54]
  >>> marks[1] = 75     # change the second element
  >>> marks.append(55)   # add 55 to the end
  >>> marks.remove(62)   # remove the first occurrence of 62
  >>> marks.sort()       # sort to ascending order
  [54, 55, 75, 90]
  >>> 54 in marks     # membership testing
  True
  >>> marks.pop(3)    # remove AND return value at index 1
  54
  >>> marks + [1, 2]  # concatenation to a new list
  [54, 55, 75, 1, 2]

  ```

3. Variable aliasing: multiple variable may be refering to the __same__ data structure
```python
>>> l = [1,2,3]
>>> not_a_copy = l
>>> not_a_copy.append(4)
>>> l
[1,2,3,4]
>>> actually_a_list = list(l)
>>> another_copy = l[:]
```

#### Tuples
Tuples are simple list that is __immutable__, but can create list from them
```python
>>> t = (1,2,3)
>>> t[2] = '4'
TypeError  

>>> L = list(t)     # Elements are mutable now
```

- **
### For Loops
1. For loops repeat some code for each element in a _sequence_   
  ```python
  >>> colors = ['red', 'yellow', 'blue']
  >>> for color in colors:
  ...      print(color )
  red
  yellow
  blue

  ```
2. Use `rang(n)` in a for loop to loop over a range to get index.      
  ```python    
  >>> for i in range(2):
  ...     print(i)

  0
  1
  >>> for i in range(4,6):            # to start at a value other than 4
  ...     print(i)

  4
  5
  >>> for i in range(len(colors)):    # to loop over the indices of a list
  ...     print('{}. {}'.format(i, colors[i]))

  0. red
  1. yellow
  2. blue
  ```    

3. To get both index and item in the for loop    

  ```python
  >>> n = len(colors)
  >>> for (i, color) in zip(range(n), colors):  # Use zip to return a list of the pair
  ...      print('{}. {}'.format(i, color))

  0. red
  1. yellow
  2. blue
  >>> for (i, color) in enumerate(colors):      # enumerate() does the same thing
  ...       print('{}. {}'.format(i, color))

  0. red
  1. yellow
  2. blue

  ```

- **
### Conditionals
__if__ statement allows you to execute code based on conditoins   
__elif__, and __else__ are optionals
```python
if a>b:
  print('a is greater b')
elif a==b:
  print('a is equal to b')
```
- **

### Functions    
functions allow one to put together a bunch of statement into a block that you call. They take in information (_arguments_) and give back information(_return value_). `None` will be returned if `return` is not specified
#### Docstrings
Each function should have a docstring, a multi-lined, triple-quoted string right after the function declaration. It describes __what__ the function does, not __how__ it does it. Describes argument and return types   
```python
def C_to_F(degrees):
  ''' convert degrees from C to F

  @type degrees: int | float
  @rtype: float
  '''
```

####  pass
`pass` is a null operation
```python
def is_reverse(s1, s2):
  pass  
def is_reverse(s1, s2):
  return s1[::-1] == s2
```
- **
### Dictionaries
> {'dictionary': 'awesome'}

Dictionaries, or  __dict__, are an _unordered_ association of __key__ with __values__. Often used to store associations
+ Keys must be __unique__ and __immutable__
  ```python
  >>> scores = {'Alice': 90, 'Bob': 76, 'Eve': 82}
  >>> scores['Alice']           # get
  90
  >>> scores['Charles'] = 64    # set
  >>> scores.pop('Bob')         # delete
  76
  >>> 'Eve' in  scores          # membership testing
  True
  >>> for names in scores:      # lops over keys
  ...      print('{}: {}'.format(name, scores[name]))

  Charlie: 64
  Alice: 88
  Eve: 82
  ```

- **

### Accessing file system
Naively, without error handling    
```python
f = open('my_text.txt')
for line in f
  # do something
f.close()
```
Alternatively,  use `with as` to open something for a while, but always close it
```python
with open('my_text.txt') as open_file:
  for line in open_file:
    # do something

with open('my_text.txt', 'w' ) as file:
  file.write(data)       # write to file
```
- **
### While loop    
While loops keep repeating a block of code while a condition is `True`
```python
var = 10
while var > 0:      # prints hello 10 times
  print('hello')
  val -= 1
```

`break` can be used to exists the loop early
```python
while True:       # infinity loop
  # stops when user type in q
  response = input('enter number or "quit": ')
  if response.lower().startswith('q'):
    break         # breaks out of the while loop

```  

- **
### Modules
> Why reinvent the wheels

import a variety of modules
```python
>>> from random import randint
>>> randint(1,6)

```
- **

### Function design recipe
+ example: calls of your function and the expected return value
+ type contract: describes the type of the parameters and return value
+ header: write the function header above the docstring
+ description: describe what the function does and mention each parameter by name
+ body: write the implementation of the function

- **
### Testing - Unittest
```python
import unittest
from even import is_even

class EvenTestCase(unittest.TestCase):

  def test_is_two_even(self):       # here test_* methods are recognized by the module
    self.assertTrue(is_even(2))

is __name__ == '__main__':
  unittest.main()
```

- **
### Testing - Doctest
The module scans all doctrings and search for pieces of text that looks like _interactive Python session_, then execute these sessions to verify that it works exactly as shown  

```python
def is_even(value):
  '''
  return True iff value is divisible by 2

  @type value: int
  @rtype: bool  

  >>> is_even(2)        # execute this 
  True
  >>> is_even(7)
  False
  '''
  return value%2 == 0

if __name__ == '__main__':  
  import doctest              
  doctest.testmod()
```
