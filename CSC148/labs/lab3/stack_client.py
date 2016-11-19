from stack import Stack


def list_stack(l, s):
    """
    Adds each element of the list to the stack.
    Remove non-list elements and print them and store list elements to stack
    continue until the stack is empty


    @param list l: a list
    @param Stack s: stack object
    @rtype: None
    >>> ls = list_stack([1, 3, 5], Stack())
    5
    3
    1
    >>> ls = list_stack([1, [3, 5], 7], Stack())
    7
    5
    3
    1
    >>> ls = list_stack([1, [3, [5, 7], 9], 11], Stack())
    11
    9
    7
    5
    3
    1
    """
    for i in l:
        s.add(i)
    while not s.is_empty():
        el = s.remove()
        if isinstance(el, list):
            for j in el:
                s.add(j)
        else:
            print(el)


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    s = Stack()
    imp = input('type a string: ')
    while not imp == 'end':
        s.add(imp)
        inp = input('type a string: ')

    while not s.is_empty():
        print(s.remove())
    list_stack([1, [3, [5, 7], 9], 11], Stack())
