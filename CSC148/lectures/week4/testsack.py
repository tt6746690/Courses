import unittest
from sack import Sack


class SackEmptyTestCase(unittest.TestCase):
    """Test behaviour of an empty Sack."""

    def setUp(self):
        """Set up an empty Sack."""
        self.Sack = Sack()

    def tearDown(self):
        """Clean up."""
        self.Sack = None

    def testIsEmpty(self):
        """Test is_empty() on empty Sack."""
        # it's hard to avoid \ continuation here.
        assert self.Sack.is_empty(), \
            'is_empty returned False on an empty Sack!'

    def testadd(self):
        """Test add to empty Sack."""

        self.Sack.add("foo")
        assert self.Sack.remove() == "foo", \
            'Wrong item on top of the Sack! Expected "foo" here.'


class SackAllTestCase(unittest.TestCase):
    """Comprehensive tests of (non-empty) Sack."""

    def setUp(self):
        """Set up an empty Sack."""
        self.Sack = Sack()

    def tearDown(self):
        """Clean up."""
        self.Sack = None

    def testAll(self):
        """Test adding and removeping multiple elements."""
        number_set = set(range(20))

        for item in number_set:
            self.Sack.add(item)
            assert not self.Sack.is_empty(), \
                'is_empty() returned True on a non-empty Sack!'

        while not self.Sack.is_empty():
            element = self.Sack.remove()
            assert element in number_set, \
                'Something wrong on top of the Sack!'
            number_set -= {element}


if __name__ == '__main__':
    unittest.main(exit=False)
