from csc148_queue import Queue



def list_queue(l, q):
    """
    Adds each element of the list to the queue
    Remove non-list elements and print them and store list elements to queue
    continue until the stack is empty

    q assumed to be empty

    @param list l: a list
    @param Queue q: queue object
    @rtype: None
    >>> ls = list_queue([1, 3, 5], Queue())
    1
    3
    5
    >>> ls = list_queue([1, [3, 5], 7], Queue())
    1
    7
    3
    5
    >>> ls = list_queue([1, [3, [5, 7], 9], 11], Queue())
    1
    11
    3
    9
    5
    7
    """
    while(True):
        if l:
            for e in l:
                q.add(e)
        if not q.is_empty():
            firste = q.remove()
            if type(firste) == list:
                l = firste
                continue
            else:
                print(firste)
                l = None
                continue
        else:
            break

    ''' alternatively
    for i in l:
        q.add(i)
    while not q.is_empty():
        el = q.remove()
        if isinstance(el, list):
            for j in el:
                q.add(j)
        else:
            print(el)'''


if __name__ == '__main__':
    import doctest
    doctest.testmod()

    q = Queue()

    while(True):
        inp = int(input('enter an integer: '))
        if inp == 148:
            break
        q.add(inp)

    mul = 1
    while(not q.is_empty()):
        mul = q.remove() * mul
    print(mul)
