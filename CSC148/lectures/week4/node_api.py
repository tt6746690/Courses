class LinkedListNode:
    """
    Node to be used in linked list

    === Attributes ===
    @param LinkedListNode next_: successor to this LinkedListNode
    @param object value: data this LinkedListNode represents
    """
    def __init__(self, value, next_=None):
        """
        Create LinkedListNode self with data value and successor next_.

        @param LinkedListNode self: this LinkedListNode
        @param object value: data of this linked list node
        @param LinkedListNode|None next_: successor to this LinkedListNode.
        @rtype: None
        """
        self.value, self.nxt = value, next

    def __str__(self):
        """
        Return a user-friendly representation of this LinkedListNode.

        @param LinkedListNode self: this LinkedListNode
        @rtype: str

        >>> n = LinkedListNode(5, LinkedListNode(7))
        >>> print(n)
        5 -> 7 ->|
        """
        pass

    # def __eq__(self, other):
    #     """
    #     Return whether LinkedListNode self is equivalent to other.
    #
    #     @param LinkedListNode self: this LinkedListNode
    #     @param LinkedListNode|object other: object to compare to self.
    #     @rtype: bool
    #
    #     >>> LinkedListNode(5).__eq__(5)
    #     False
    #     >>> n1 = LinkedListNode(5, LinkedListNode(7))
    #     >>> n2 = LinkedListNode(5, LinkedListNode(7, None))
    #     >>> n1.__eq__(n2)
    #     True
    #     """
    #     pass


class LinkedList:
    """
    Collection of LinkedListNodes

    === Attributes ==
    @param: LinkedListNode front: first node of this LinkedList
    @param LinkedListNode back: last node of this LinkedList
    @param int size: number of nodes in this LinkedList
                        a non-negative integer
    """
    def __init__(self):
        """
        Create an empty linked list.

        @param LinkedList self: this LinkedList
        @rtype: None
        """
        pass

    def __str__(self):
        """
        Return a human-friendly string representation of
        LinkedList self.

        @param LinkedList self: this LinkedList

        >>> lnk = LinkedList()
        >>> lnk.prepend(5)
        >>> print(lnk)
        5 ->|
        """
        pass

    # def __eq__(self, other):
    #     """
    #     Return whether LinkedList self is equivalent to
    #     other.
    #
    #     @param LinkedList self: this LinkedList
    #     @param LinkedList|object other: object to compare to self
    #     @rtype: bool
    #
    #     >>> LinkedList().__eq__(None)
    #     False
    #     >>> lnk = LinkedList()
    #     >>> lnk.prepend(5)
    #     >>> lnk2 = LinkedList()
    #     >>> lnk2.prepend(5)
    #     >>> lnk.__eq__(lnk2)
    #     True
    #     """
    #     pass
    #
    # def append(self, value):
    #     """
    #     Insert a new LinkedListNode with value after self.back.
    #
    #     @param LinkedList self: this LinkedList.
    #     @param object value: value of new LinkedListNode
    #     @rtype: None
    #
    #     >>> lnk = LinkedList()
    #     >>> lnk.append(5)
    #     >>> lnk.size
    #     1
    #     >>> print(lnk.front)
    #     5 ->|
    #     >>> lnk.append(6)
    #     >>> lnk.size
    #     2
    #     >>> print(lnk.front)
    #     5 -> 6 ->|
    #     """
    #     pass
    #
    def prepend(self, value):
        """
        Insert value before LinkedList self.front.

        @param LinkedList self: this LinkedList
        @param object value: value for new LinkedList.front
        @rtype: None

        >>> lnk = LinkedList()
        >>> lnk.prepend(0)
        >>> lnk.prepend(1)
        >>> lnk.prepend(2)
        >>> str(lnk.front)
        '2 -> 1 -> 0 ->|'
        >>> lnk.size
        3
        """
        pass

    # def delete_front(self):
    #     """
    #     Delete LinkedListNode self.front from self.
    #
    #     Assume self.front is not None
    #
    #     @param LinkedList self: this LinkedList
    #     @rtype: None
    #
    #     >>> lnk = LinkedList()
    #     >>> lnk.prepend(0)
    #     >>> lnk.prepend(1)
    #     >>> lnk.prepend(2)
    #     >>> lnk.delete_front()
    #     >>> str(lnk.front)
    #     '1 -> 0 ->|'
    #     >>> lnk.size
    #     2
    #     >>> lnk.delete_front()
    #     >>> lnk.delete_front()
    #     >>> str(lnk.front)
    #     'None'
    #     """
    #     pass
    #
    # def __getitem__(self, index):
    #     """
    #     Return the value at LinkedList self's position index.
    #
    #     @param LinkedList self: this LinkedList
    #     @param int index: position to retrieve value from
    #     @rtype: object
    #
    #     >>> lnk = LinkedList()
    #     >>> lnk.prepend(1)
    #     >>> lnk.prepend(0)
    #     >>> lnk.__getitem__(1)
    #     1
    #     >>> lnk[-1]
    #     1
    #     """
    #     pass
    #
    def __contains__(self, value):
        """
        Return whether LinkedList self contains value.

        @param LinkedList self: this LinkedList.
        @param object value: value to search for in self
        @rtype: bool

        >>> lnk = LinkedList()
        >>> lnk.prepend(0)
        >>> lnk.prepend(1)
        >>> lnk.prepend(2)
        >>> lnk.__contains__(1)
        True
        >>> lnk.__contains__(3)
        False
        """
        pass


if __name__ == '__main__':
    import doctest
    doctest.testmod()
