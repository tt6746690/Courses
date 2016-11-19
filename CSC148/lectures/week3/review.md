Inheritance
==========

### Super()
1. call parent init/methods
2. avoid referring to the base class explicitly, adding another level of indirection !!!

```python
class Child(Base):
    def __init__(self, items=()):
        super().__init__(items)
        super().sort()              # access overriden methods in Base class

    def sort(self):
      pass

```

- **

### Inheritance
The goal of inheritance is to use one class as template to define another - _shared public interfaces_
+ A class that has at least one unimplemented class is a __abstract class__, which should never be instantiated directly, but should instead serve as a _base_ for other classes

```python
class Animal:
    """An animal in the zoo.

    This is the public interface of animals only,
    and is not meant to be instantiated.
    """

    def feed(self):
        """Feed the animal.

        @type self: Animal
        @rtype: None

        >>> a = Animal()
        >>> a.feed()
        NotImplementedError
        """
        raise NotImplementedError()

    def greet(self, other):
        """Return a message from <self> to <other>.

        @type self: Animal
        @type other: Animal
        @rtype: str
        """
        raise NotImplementedError()

    def sleep(self):
        """Print out a message indicating the animal is asleep.

        @type self: Animal
        @rtype: None
        """
        print('I\'m sleeping now. Zzzzzzz...')

```

```python
class Lion(Animal):
    pass
```

+ Here we call `Lion` a subclass of `Animal`. `Animal` is the _base_ or _super_ class of `Lion`
> All instances of `Lion` is also an instance of `Animal`, therefore the access to all methods of `Animal`

+ This enables codes to be shared between classes


- **
### Method overrides
When a subclass implements a method with the same name as a superclass method, we say that the subclass method overrides the superclass method. Python searches for the method in the child class first, and if not found, search the superclass, superclass of superclass and so on...
```python
class Lion(Animal):
    ...

    def sleep(self):
        """Print out a message indicating the lion is asleep.

        @type self: Lion
        @rtype: None
        """
        print('In the jungle...')
```

To call the `Animal` version of the method on a `Lion` instance, use `super`! Super takes a class and an instance of that class, and returns a _new version_ of that class whose methods will be looked up in the original object's superclass
```python
>>> simba = Lion()
>>> super(Lion, simba).sleep()        # super(class, instance) returns the superclass
I'm sleeping now. Zzzzzzz...
```

- **
It makes sense to raise NotImplementedError in the constructor of base class since abstract class are not meant to be instantiated. But we may want to put code for initializing these attributes in the class constructor, rather than duplicating the code inside the constructor of each subclass. In the process of doing so, we must call the parent constructor explicitly to initialize all attributes inside the constructor
> the parent's constructor is not called automatically in the subclass constructor

```python
class Animal:
    def __init__(self, name):
        """Create a new animal with the given name.

        Note: this constructor is meant for internal use only; Animal is an
        abstract class and should not be instantiated.
        """
        self.name = name
        self.hunger = 0

class Lion(Animal):
    def __init__(self, name):
        # Note that to call the parent class' constructor, we need to use the
        # full name '__init__'. This is the *only* time we'd need to do so.
        super(Lion, self).__init__(name)
        self._kills = 0

class Giraffe(Animal):
    def __init__(self, name, neck_length):
        super(Giraffe, self).__init__(name)
        self._neck_length = neck_length
```

- **

### Composition
Inheritance is not perfect in every situation. We can use composition, which signifies a _has_a_ relationship rather than _is_a_ relationship. The benefit of composition is that there is no possibilities of namespace clash, as every attributes/methods are readily available

- **

### Python Magic class
> Every python class is a descendent of base `object` class, which provides universal methods named with double underscore

| Methods | Special Syntax    |
| :------------- | :------------- |
| `a.__str__()`       | `str(a) or print(a)`   |
| `a.__eq__(b)` | `a == b` |
| `a.__len__()` | `len(a)`|
|`a.__contain__(b)`| `b in a`|
| `a.__getitem__(key)`| `a[key]` |
|`a.__add__(b)` | `a+b`|
