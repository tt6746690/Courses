
### Links

[tuple unpacking] (http://python.net/~goodger/projects/pycon/2007/idiomatic/handout.html#swap-values)

tuple  
comma: `,` is a tuple constructor. A tuple is created on the right(packing) and assigned to target on the left (tuple unpacking)

```python
>>> l =['David', 'Pythonista', '+1-514-555-1234']
>>> name, title, phone = l
>>> name
'David'
>>> title
'Pythonista'
>>> phone
'+1-514-555-1234'

# loops
>>> people = [l, ['Guido', 'BDFL', 'unlisted']]
>>> for (name, title, phone) in people:
...     print name, phone
...
David +1-514-555-1234
Guido unlisted
```

### Manipulating lists

zip  
`zip([iterable, ...])` returns a list of tuples, where the i-th tuple contains the i-th element from each of the argument sequences or iterables. The returned list is truncated in length to the length of the shortest argument sequence.

```python
>>> x = [1, 2, 3]
>>> y = [4, 5, 6]
>>> zipped = zip(x, y)
>>> zipped
[(1, 4), (2, 5), (3, 6)]
>>> x2, y2 = zip(*zipped)
>>> x == list(x2) and y == list(y2)
True
```


map  
`map(func, seq)`

filter
`filter(function, iterable)` Construct an iterator from those elements of iterable for which function returns true. iterable may be either a sequence, a container which supports iteration, or an iterator. Note that `filter(function, iterable)` is equivalent to the generator expression `(item for item in iterable if function(item))` if function is not `None` and `(item for item in iterable if item)` if function is `None`.


### test case design

[Choosing test cases] (http://www.cdf.toronto.edu/~csc148h/winter/Labs/lab05/ChoosingTestCases.html)
[testing function that mutate values] (http://www.cdf.toronto.edu/~csc148h/winter/Labs/lab05/TestingFunctionsThatMutateValues.html)
