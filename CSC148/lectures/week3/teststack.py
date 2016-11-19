import unittest
from stack import Stack


class StackEmptyTestCase(unittest.TestCase):
    """Test behaviour of an empty Stack."""

    def setUp(self):
        """Set up an empty stack."""
        self.stack = Stack()

    def tearDown(self):
        """Clean up."""
        self.stack = None

    def testIsEmpty(self):
        """Test is_empty() on empty Stack."""
        # it's hard to avoid \ continuation here.
        assert self.stack.is_empty(), \
            'is_empty returned False on an empty Stack!'

    def testadd(self):
        """Test add to empty Stack."""

        self.stack.add("foo")
        assert self.stack.remove() == "foo", \
            'Wrong item on top of the Stack! Expected "foo" here.'


class StackAllTestCase(unittest.TestCase):
    """Comprehensive tests of (non-empty) Stack."""

    def setUp(self):
        """Set up an empty stack."""
        self.stack = Stack()

    def tearDown(self):
        """Clean up."""
        self.stack = None

    def testAll(self):
        """Test adding and removeping multiple elements."""

        for item in range(20):
            self.stack.add(item)
            assert not self.stack.is_empty(), \
                'is_empty() returned True on a non-empty Stack!'

        expect = 19
        while not self.stack.is_empty():
            assert self.stack.remove() == expect, \
                ('Something wrong on top of the Stack! Expected ' +
                 str(expect) + '.')
            expect -= 1


if __name__ == '__main__':
    unittest.main(exit=False)
