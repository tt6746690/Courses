# use a stack to check whether parentheses are balanced
import stack
# STACK CLIENT

def balanced_delimiters(s):
    """
    Return whether the delimiters in string s
    are balanced.

    Assume: Only delimiters are brackets, parentheses, braces

    @param str s: string to check for balanced delimiters.
    @rtype: bool

    >>> balanced_delimiters("[({])}")
    False
    >>> balanced_delimiters("[({})]]")
    False
    >>> balanced_delimiters("[[]")
    False
    >>> balanced_delimiters("[(){}]")
    True
    """
    st = stack.Stack()
    for c in s:
        if c not in '(){}[]':
            pass                    # pass does NOTHING - different from continue, which jumps to next ieration
        elif c in '({[':
            st.add(c)
        # now must have c in ')}]'
        elif not st.is_empty():
            c2 = st.remove()
            if ((c == ')' and not c2 == '(') or
                    (c == '}' and not c2 == '{') or
                    (c == ']' and not c2 == '[')):
                return False                        # return false when detecting imbalance
        else:  # prematurely empty stack!
            return False
    return st.is_empty()  # better be empty at the end


if __name__ == '__main__':
    import doctest

    doctest.testmod()
