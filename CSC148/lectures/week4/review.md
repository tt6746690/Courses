
### Abstract data types
Abstract data types is a particular type of `class` defined purely by its interface rather than its implementation, making it _abstract_  
>An abstract data type, or ADT, specifies a set of operations (or methods) and the semantics of the operations (what they do), but it does not not specify the implementation of the operations. That’s what makes it abstract.


+ lists: store elements sequentially. Access elements by index, add new elements to the list, determine if the list is empty.
+ dictionaries: store key-value pairs. Look up a value based on key, add/remove key-value pairs.
+ files: objects that you can open for reading in data or for writing data. Start reading from the beginning of a file, starting writing at the beginning of a file, skip to some part of a file.    

- **

### [STACK](http://openbookproject.net/thinkcs/python/english3e/stacks.html)- a simpler ADT
Benefits of stack lies in its simplicity: a client knows exactly what operations can be performed on it:    
+ if it's empty
+ push an item
+ pop an item from it
A stack therefore can be used to store information

### Use `assert` to debug   
`assert` can be thought of as a __raise-if-not__, where an expression evaluated and exceptions raised if the result come up false

`assert Expression[, Arguments]`

An example
```python
def KelvinToFahrenheit(Temperature):
   assert (Temperature >= 0),"Colder than absolute zero!"
   return ((Temperature-273)*1.8)+32

print KelvinToFahrenheit(273)
print int(KelvinToFahrenheit(505.78))
print KelvinToFahrenheit(-5)

'''
32.0
451
Traceback (most recent call last):
  File "test.py", line 9, in <module>
    print KelvinToFahrenheit(-5)
  File "test.py", line 4, in KelvinToFahrenheit
    assert (Temperature >= 0),"Colder than absolute zero!"
AssertionError: Colder than absolute zero!
'''
```


### Exceptions is better
```python
class EmptyStackError(Exception):
    # Most of the time we'll leave the class body empty, using the
    # default method implementations provided in Exception.
    # However, there are more complex uses for Exception subclasses
    # which we might touch on briefly in this course.
    pass

...

    def pop(self):
        """Remove and return the element at the top of this stack.

        Raise an EmptyStackError if <self> is empty.

        @type self: Stack
        @rtype: object
        """
        if self.is_empty():
            raise EmptyStackError()
        else:
            return self._items.pop()

'''
>>> s = Stack()
>>> s.pop()
Traceback (most recent call last):
  File "<input>", line 1, in <module>
  File "C:\Users\David\Documents\csc148_f15\pycharm\csc148\week4\week4.py", line 60, in pop
    raise EmptyStackError()
EmptyStackError
'''
```

- **

### Efficiency
Quality of program
+ correctness
+ design - well documented, structured
+ run time efficiency

### Runtime   
Use a function of the amount of time an algorithm takes to run in terms of the size of the input. We can write something like T(n) to denote the runtime of a function of size n (but note that this isn't always necessarily n). Many confounding factors such as CPU, programs running concurrently... This is where __Big-Oh__ comes in: it allows an elegant way of roughly characterising the type of growth of the runtime of an algorithm, without actually worrying about things like how different CPUs implement different operations, whether a for loop is faster than a while loop, etc. Not that these things aren't important -- they are simply much too technical for this course. When we characterise the Big-Oh property of a function, we really are thinking about general terms like linear, quadratic, or logarithmic growth.

```python
for item in lst:
    # do something with item
```
Here, we know that the runtime is proportional to the length of the list. If the list gets twice as long, we'd expect this algorithm to take twice as long. The runtime grows linearly with the size of the list, and we write that the runtime is O(n).


Big-Oh notation allows us to analyse the running time of algorithms while ignoring two details:

1. The constants and lower-order terms involved in the step counting: 5n, n + 10, 19n − 10, 0.05n are all O(n) (they all grow linearly).
2. The algorithm's running time on small inputs. The key idea here is that an algorithm's behaviour as the input size gets very large is much more important than how quickly it runs on small inputs. (This is what's meant when we say that Big-Oh deals with that asymptotic runtime. )
