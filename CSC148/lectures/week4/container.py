class Container:
    """
    Container with add, remove, and is_empty methods.

    This is an abstract class that is not meant to be instantiated itself,
    but rather subclasses are to be instantiated.
    """
    def __init__(self):
        """
        Create a new Container self.

        @param Container self: this Container
        @rtype: None
        """
        self._contents = None
        raise NotImplementedError("Subclass this!")

    def __str__(self):
        """
        Return a string representation of Container self.

        @param Container self: this container
        @rtype: str
        """
        return str(self._contents)

    def __eq__(self, other):
        """
        Return whether Container self is equivalent to other.

        @param Container self: this Container
        @param Container|object other: object to compare to self.
        @rtype: bool
        """
        return (type(self) is type(other) and
                self._contents == other._contents)

    def add(self, obj):
        """
        Add obj to Container self.

        @param Container self: this Container
        @param object obj: object to add to self
        @rtype: None
        """
        raise NotImplementedError("Subclass this!")

    def remove(self):
        """
        Remove and return an object from Container self.

        Assume that Container self is empty.

        @param Container self: this Container.
        @rtype: object
        """
        raise NotImplementedError("Subclass this!")

    def is_empty(self):
        """
        Return whether Container self is empty.

        @param Container self: this Container
        @rtype: bool
        """
        raise NotImplementedError("Subclass this!")
