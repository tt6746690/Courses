import unittest
from stack import Stack

class EmptyTestCase(unittest.TestCase):
    '''test behavior of an empty stack
    '''
    def setUp(self):
        self.s = Stack()
    def tearDown(self):
        self.s = None
    def testIsEmpty(self):
        '''test is_empty() on empty Queue'''
        self.assertTrue(self.s.is_empty(), 'is_empty returned False on an empty stack')

class SingletonTestCase(unittest.TestCase):
    '''check whether adding a single item makes it appear in the top'''
    def setUp(self):
        self.s = Stack()
        self.s.add('a')
    def tearDown(self):
        self.s = None
    def testIsEmpty(self):
        self.assertFalse(self.s.is_empty(), 'is_empty returned true on non-empty stack')
    def testRemove(self):
        top = self.s.remove()
        self.assertEqual(top, 'a', 'The item at the top should have been "a" but was ' +
        top + '.')
        self.assertTrue(self.s.is_empty, ' stack with one element not empty after remove()')

class TypicalTestCase(unittest.TestCase):
    def setUp(self):
        self.s = Stack()
    def tearDown(self):
        self.s = None
    def testAll(self):
        for item in range(20):
            self.s.add(item)
            self.assertFalse(self.s.is_empty(), 'stack should not be empty after adding item ' + str(item))

        item = 19
        while not self.s.is_empty():
            top = self.s.remove()
            self.assertEqual(top, item, 'wrong item at the top of the stack. Found' + str(top) + ' but expecting ' + str(item))
            item -=1


if __name__ == '__main__':
    unittest.main(exit=False)
