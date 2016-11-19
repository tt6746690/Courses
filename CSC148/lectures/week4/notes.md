
STACK and SACK ADTs
==================
- **
### Definition
A __stack__ contains item of various sorts. New items are added on to the top of the stack, items may only be removed from the top of the stack. It's a LIFO structure. It is a mistake to try to remove an item from an empty stack, so we need to know if it is empty. We can tell how big a stack is.


A __sack__ contains items of various sorts. New items are added on to a random place in the sack, so the order items are removed from the sack is completely unpredictable. It is a mistake to try to remove an item from an empty sack, so we need to know if it is empty. We can tell how big a sack is.

A __queue__ is an abstract data type (ADT) that stores a sequence of values. A queue makes sure that the first item in is the first item out(FIFO). This models a lineup at a coffee shop. should support
  + add: add to the end of queue
  + remove: remove and return object at the beginning of the queue
  + is_empty: returns True if the queue is empty

### Design of container API
Abstract the commonalities in stack and sack to a higher level (super) class

```python
s.__init__()    # implement in subclasses
s.__str__()     # super
s.__eq__()      # subclasses   for stack: if type, location of objects, and object values equates
                #              for sack:  if type, and object values equates. location are random in sack
                # should not be implemented in container level
s.add           # subclasses
s.remove        # subclasses
s.is_empty      # subclasses
                # because Container can be super class for other ADTs, which may contain different
                # is_empty implementation. stack/sack happens to share, for example, list as
                # object holder. Other ADT's inheriting from Container may have different ones
                # Therefore better to implement it in subclass
```

### Testing
Drawbacks of testing on console:
+ not organized (not able to test large codes)
+ not documented (not conforming with basic principles)
+ not reused (not being able to do regression test)
+ tedious to conduct independent test

#### unittest   
Unittest is a framework to setup test cases. run them independently from one another, document them, and reuse them when needed,...

Extending `unittest.TestCase` is not necessarily different than extending any other class
```python
class myStackTestCase(unittest.TestCase)
```
Override some special methods:
```python
def setUp():           # called before every test method => to initialize objects
def tearDown():        # called after every test methods
```
follow some conventions
```python
def test_foo():      #everthing function starts with test_
  assert
```

- **
### Application of stack
for a newly developed data types
> parenthesization

In some situations it is important that opening and closing parentheses match. Many computer programs (interpreters, compliers) need to evaluate such expressions. Programs see character one at a time. How do program keep track of parentheses?
+ keep track of parentheses by a iterator: increment when left parentheses appears, decrement when right parentheses appears. But a drawback is that this is not scalable. Additional symbols increase complexity.
+ Scan the string from left to right. Discard any character other than parenthesis. Push a left parenthese to a stack and pop the left parentheses when encoutering a right parentheses. This approach is scalable since other symbols are also kept track of. `{ab(c)[]}`


- **
### Linked List
Regular Python lists are flexible and useful, but overkill in some situations. They allocate large blocks of contiguous memory, which becomes increasingly difficult as memory is in use. A change in the middle of the list will result in a shift of values in all subsequent values. A big list is less efficient.

Linked List ADT is designed to be scalable as values not need to be reallocated when one value in the middle is changed. Linked list nodes reserve just enough memory for the object value they want to refer to, a reference to it, and a reference to the next node in the list. For now, we implement a linked list as objects (nodes) with a value and a reference to other similar values.
```
12|ref-->99|ref-->37|ref-->end

```
