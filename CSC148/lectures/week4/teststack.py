import unittest
from stack import Stack


class StackEmptyTestCaseByProf(unittest.TestCase):
    """Test behaviour of an empty Stack."""

    def setUp(self):
        """Set up an empty stack."""
        self.s1 = Stack()
        self.s2 = Stack()

    def tearDown(self):
        """Clean up."""
        self.s1 = None
        self.s2 = None

    def test_IsEmpty(self):
        """Test is_empty() on empty Stack."""
        # it's hard to avoid \ continuation here.
        self.assertTrue(self.s1.is_empty())

    def test_add(self):
        """Test add to empty Stack."""

        self.s1.add("foo")
        self.assertTrue(self.s1.remove() == "foo")


    def test_equality(self):
        """test if two non-empty stack are equal"""
        self.s1.add("foo")
        self.s1.add("jijiji")
        self.s2.add("foo")
        self.s2.add("jijiji")

        self.assertTrue(self.s1 == self.s2)

    def test_not_equality(self):
        """test if two non-empty stack are equal"""
        self.s1.add("foo")
        self.s1.add("Joo")
        self.s2.add("Joo")
        self.s2.add("foo")
        self.assertFalse(self.s1 == self.s2)

# alternatively --
class StackEmptyTestCase(unittest.TestCase):
    """Test behaviour of an empty Stack."""

    def setUp(self):    # executed for every function
        """Set up an empty stack."""
        self.stack = Stack()

    def tearDown(self): # executed for every function
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
