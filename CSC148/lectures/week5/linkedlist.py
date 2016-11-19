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
        self.value, self.next_ = value, next_

    def __str__(self):
        """
        Return a user-friendly representation of this LinkedListNode.

        @param LinkedListNode self: this LinkedListNode
        @rtype: str

        >>> n = LinkedListNode(5, LinkedListNode(7))
        >>> print(n)
        5 -> 7 ->|
        """
        s = "{} ->".format(self.value)
        cur_node = self
        while cur_node is not None:
            if cur_node.next_ is None:
                s += "|"
            else:
                s += " {} ->".format(cur_node.next_.value)
            cur_node = cur_node.next_
        return s

    def __eq__(self, other):
        """
        Return whether LinkedListNode self is equivalent to other.

        @param LinkedListNode self: this LinkedListNode
        @param LinkedListNode|object other: object to compare to self.
        @rtype: bool

        >>> LinkedListNode(5).__eq__(5)
        False
        >>> n1 = LinkedListNode(5, LinkedListNode(7))
        >>> n2 = LinkedListNode(5, LinkedListNode(7, None))
        >>> n1.__eq__(n2)
        True
        """
        self_node, other_node = self, other

        while (self_node is not None and
               type(self_node) is type(other_node) and
               self_node.value == other_node.value):
            self_node, other_node = self_node.next_, other_node.next_
        return self_node is None and other_node is None


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
        self.front, self.back = None, None
        self.size = 0

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
        return str(self.front)

    def __eq__(self, other):
        """
        Return whether LinkedList self is equivalent to
        other.

        @param LinkedList self: this LinkedList
        @param LinkedList|object other: object to compare to self
        @rtype: bool

        >>> LinkedList().__eq__(None)
        False
        >>> lnk = LinkedList()
        >>> lnk.prepend(5)
        >>> lnk2 = LinkedList()
        >>> lnk2.prepend(5)
        >>> lnk.__eq__(lnk2)
        True
        """
        return (type(self) is type(other) and
                (self.size, self.front, self.back) ==
                (other.size, other.front, other.back))

    def append(self, value):
        """
        Insert a new LinkedListNode with value after self.back.

        @param LinkedList self: this LinkedList.
        @param object value: value of new LinkedListNode
        @rtype: None

        >>> lnk = LinkedList()
        >>> lnk.append(5)
        >>> lnk.size
        1
        >>> print(lnk.front)
        5 ->|
        >>> lnk.append(6)
        >>> lnk.size
        2
        >>> print(lnk.front)
        5 -> 6 ->|
        """
        new_node = LinkedListNode(value)
        if self.front is None:
            # append to an empty LinkedList
            self.front = self.back = new_node
        else:
            # self.back better not be None
            assert self.back, 'Unexpected None node'
            self.back.next_ = new_node
            self.back = new_node
        self.size += 1

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
        self.front = LinkedListNode(value, self.front)
        if self.back is None:
            self.back = self.front
        self.size += 1

    def delete_front(self):
        """
        Delete LinkedListNode self.front from self.

        Assume self.front is not None

        @param LinkedList self: this LinkedList
        @rtype: None

        >>> lnk = LinkedList()
        >>> lnk.prepend(0)
        >>> lnk.prepend(1)
        >>> lnk.prepend(2)
        >>> lnk.delete_front()
        >>> str(lnk.front)
        '1 -> 0 ->|'
        >>> lnk.size
        2
        >>> lnk.delete_front()
        >>> lnk.delete_front()
        >>> str(lnk.front)
        'None'
        """
        assert self.front is not None, "Delete from empty list!"
        if self.back == self.front:
            self.back = None
        self.front = self.front.next_
        self.size -= 1

    def __getitem__(self, index):
        """
        Return the value at LinkedList self's position index.

        @param LinkedList self: this LinkedList
        @param int index: position to retrieve value from
        @rtype: object

        >>> lnk = LinkedList()
        >>> lnk.prepend(1)
        >>> lnk.prepend(0)
        >>> lnk.__getitem__(1)
        1
        >>> lnk[-1]
        1
        """
        while index < 0:
            # deal with negative index
            index += self.size
        if index >= self.size or self.size == 0:
            return IndexError("too big!!!")
        else:
            cur_node = self.front
            for step in range(0, index):
                cur_node = cur_node.next_
                assert cur_node, \
                    'step: {}, out of range'.format(step)
            return cur_node.value

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
        >>> 1 in lnk
        True
        >>> lnk.__contains__(3)
        False
        """
        cur_node = self.front
        while cur_node:
            if cur_node.value == value:
                return True
            cur_node = cur_node.next_
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
