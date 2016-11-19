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
        # start with a string s to represent current node.
        s = "{} ->".format(self.value)
        # create a reference to "walk" along the list
        current_node = self.next_
        # for each subsequent node in the list, build s
        while current_node is not None:
            s += " {} ->".format(current_node.value)
            current_node = current_node.next_
        # add "|" at the end of the list
        assert current_node is None, "unexpected non_None!!!"
        s += "|"
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
        while(self is not None and
            type(self) == type(other) and
            self.value == other.value):
            self, other = self.next_, other.next_
        return (self is None and other is None)


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
        self.front, self.back, self.size = None, None, 0

    def __str__(self):
        """
        Return a human-friendly string representation of
        LinkedList self.

        @param LinkedList self: this LinkedList

        >>> lnk = LinkedList()
        >>> print(lnk)
        I'm so empty... experiencing existential angst!!!
        """
        # deal with the case where this list is empty
        if self.front is None:
            assert self.back is None and self.size is 0, "ooooops!"
            return "I'm so empty... experiencing existential angst!!!"
        else:
            # use front.__str__() if this list isn't empty
            return str(self.front)              # prints the entire list starting from front

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
        return (isinstance(other, LinkedList) and
            self.front == other.front and
            self.back == other.back and
            self.size == other.size)


    def delete_after(self, value):
        """
        Remove the node following the first occurrence of value, if
        possible, otherwise leave self unchanged.

        @param LinkedList self: this LinkedList
        @param object value: value just before the deletion
        @rtype: None

        >>> lnk = LinkedList()
        >>> lnk.append(1)
        >>> lnk.append(4)
        >>> lnk.append(9)
        >>> lnk.delete_after(4)
        >>> print(lnk)
        1 -> 4 ->|
        """
        cur_node = self.front
        while cur_node is not None:
            if cur_node.value == value:
                cur_node.next_ = cur_node.next_.next_
                return None
            cur_node = cur_node.next_
        assert cur_node is None, 'didnt reach end of list when delete after'
        return None

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
        # create the new node
        new_node = LinkedListNode(value)
        # if the list is empty, the new node is front and back
        if self.size == 0:
            assert self.back is None and self.front is None, "ooops"
            self.front = self.back = new_node
        # if the list isn't empty, front stays the same
        else:
            # change *old* self.back.next_ first!!!
            self.back.next_ = new_node
            self.back = new_node
        # remember to increase the size
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
        # Create new node with next_ referring to front
        new_node = LinkedListNode(value, self.front)
        # change front
        self.front = new_node
        # if the list was empty, change back
        if self.size == 0:
            self.back = new_node
        # update size
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
        assert self.front is not None, "unexpected None!"
        # if back == front, set it to None
        if self.front == self.back:
            self.back = None
        # set front to its successor
        self.front = self.front.next_
        # decrease size
        self.size -= 1

    def __setitem__(self, index, value):
        """
        Set the value of list at position index to value. Raise IndexError
        if index >= self.size

        @param LinkedList self: this LinkedList
        @param index int: position of list to change
        @param object value: new value for linked list
        @rtype: None

        >>> lnk = LinkedList()
        >>> lnk.prepend(5)
        >>> print(lnk)
        5 ->|
        >>> lnk[0] = 7
        >>> print(lnk)
        7 ->|
        """
        while index < 0:
            index += self.size

        if index >= self.size:
            raise IndexError('index out of range')
        else:
            cur_node = self.front
            for i in range(index):
                cur_node = cur_node.next_
                assert cur_node is not None, 'unexpected None'
            cur_node.value = value

    def __add__(self, other):
        """
        Return a new list by concatenating self to other.  Leave
        both self and other unchanged.

        @param LinkedList self: this LinkedList
        @param LinkedList other: Linked list to concatenate to self
        @rtype: LinkedList

        >>> lnk1 = LinkedList()
        >>> lnk1.prepend(5)
        >>> lnk2 = LinkedList()
        >>> lnk2.prepend(7)
        >>> print(lnk1 + lnk2)
        5 -> 7 ->|
        """
        if len(self) == 0:
            return other.copy()
        elif len(other) == 0:
            return self.copy()
        else:
            list1 = self.copy()
            list2 = other.copy()
            list1.back.next_ = list2.front
            list1.back = list2.back
            return list1
        # newlist = LinkedList()
        # cur_node = self.front
        #
        # # keep in mind that append() appends just a value not a linkedlistnode
        # while cur_node is not None:
        #     newlist.append(cur_node.value)
        #     newlist.back.next_ = cur_node.next_
        #     cur_node = cur_node.next_
        # assert (newlist.size == self.size), 'list1 not added properly'
        #
        # newlist.back.next_ = other.front
        # cur_node = other.front
        # while cur_node is not None:
        #     newlist.append(cur_node.value)
        #     newlist.next_ = cur_node.next_
        #     cur_node = cur_node.next_
        # assert (newlist.size == self.size + other.size), 'list2 not added properly'
        #
        # return newlist

    def insert_before(self, value1, value2):
        """
        Insert value1 into LinkedList self before the first occurrence
        of value2, if it exists.  Otherwise leave self unchanged.

        @param LinkedList self: this LinkedList
        @param object value1: value to insert, if possible
        @param object value2: value to insert value1 ahead of
        @rtype: None

        >>> lnk1 = LinkedList()
        >>> lnk1.append(5)
        >>> lnk1.append(7)
        >>> lnk1.append(9)
        >>> lnk1.insert_before(2, 7)
        >>> print(lnk1)
        5 -> 2 -> 7 -> 9 ->|
        """
         # keep track of two node, for linking later on
        previous_node, current_node = None, self.front
        # keep going until value2 or end of list
        # the order of the two conditions matters!!!
        while current_node is not None and current_node.value != value2:
            previous_node = current_node
            current_node = current_node.next_
        # only proceed if current_node.value == value2
        if current_node is not None:
            new_node = LinkedListNode(value1, current_node)
            if previous_node is not None:
                previous_node.next_ = new_node
            else:
                self.front = new_node
            self.size += 1
        # nothing to change here
        else:
            pass
        # cur_node = self.front
        # if cur_node.value == value2:
        #     self.prepend(value1)
        #     return None
        #
        # nxt_node = cur_node.next_
        # while nxt_node is not None:
        #     if nxt_node.value == value2:
        #         n_node = LinkedListNode(value1, nxt_node)
        #         if cur_node is not None:
        #             cur_node.next_ = n_node
        #         else:
        #             self.front = n_node
        #         return None
        #     cur_node = cur_node.next_
        #     nxt_node = nxt_node.next_


    def copy(self):
        """
        Return a copy of LinkedList self.  The copy should have
        different nodes, but equivalent values, from self.

        @param LinkedList self: this LinkedList
        @rtype: LinkedList

        >>> lnk = LinkedList()
        >>> lnk.prepend(5)
        >>> lnk.prepend(7)
        >>> print(lnk.copy())
        7 -> 5 ->|
        """
        copy_list = LinkedList()
        original_node = self.front
        while original_node is not None:
            copy_list.append(original_node.value)
            original_node = original_node.next_
        return copy_list
        # l = LinkedList()
        # if self.size == 0:
        #     return l
        # cur_node = self.front
        # while cur_node is not None:
        #     n_node = LinkedListNode(cur_node.value, cur_node.next_) # create new node of different memory address
        #     l.append(n_node.value)
        #     l.back.next_ = n_node.next_
        #     cur_node = cur_node.next_
        # return l


    def __len__(self):
        """
        Return the number of nodes in LinkedList self.

        @param LinkedList self: this LinkedList
        @rtype: int
        """
        return self.size

    def __getitem__(self, index):
        """
        Return the value at LinkedList self's position index.

        @param LinkedList self: this LinkedList
        @param int index: position to retrieve value from
        @rtype: object

        >>> lnk = LinkedList()
        >>> lnk.append(1)
        >>> lnk.append(0)
        >>> lnk.__getitem__(1)
        0
        >>> lnk[-1]
        0
        """
        # deal with a negative index by adding self.size
        while index < 0:
            index += self.size
        if index >= self.size or self.size == 0:
            raise IndexError("index too big!!!")
        # an empty list, or an index that is too large is an error
        else:
            current_node = self.front
            # walk index steps along from 0 to retrieve element
            for step in range(index):
                current_node = current_node.next_
                assert current_node is not None, "unexpected None!!!!!"
            # return the value at position index
            return current_node.value

    def __contains__(self, value):
        """
        Return whether LinkedList self contains value.

        @param LinkedList self: this LinkedList.
        @param object value: value to search for in self
        @rtype: bool

        >>> lnk = LinkedList()
        >>> lnk.append(0)
        >>> lnk.append(1)
        >>> lnk.append(2)
        >>> 2 in lnk
        True
        >>> lnk.__contains__(3)
        False
        """
        current_node = self.front
        # "walk" the linked list
        while current_node is not None:
            # if any node has a value == value, return True
            if current_node.value == value:
                return True
            current_node = current_node.next_
        # if you get to the end without finding value,
        # return False
        return False


if __name__ == '__main__':
    import doctest
    doctest.testmod()
