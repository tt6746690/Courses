# test loop and comprehension implementations

import loop
from comprehension import dot_prod, matrix_vector_prod
from comprehension import  pythagorean_triples#, any_pythagorean_triples
# from loopy import dot_prod, matrix_vector_prod, pythagorean_triples
import unittest

class DotProductTester(unittest.TestCase):

    def setUp(self: unittest.TestCase) -> None:
        self._zero_vector = [0.0, 0.0, 0.0]
        self._v1 = [1.0, 2.0, 3.0]
        self._v2 = [3.0, 2.0, 1.0]
        self._result = 10.0

    def tearDown(self: unittest.TestCase) -> None:
        self._zero_vector = None
        self._v1, self._v2, self._result = None, None, None

    def test_zero_case(self: unittest.TestCase) -> bool:
        """Dot product of zero vectors is zero?"""
        self.assertEqual(dot_prod(self._zero_vector, self._zero_vector), 0.0)

    def test_zero_case_loop(self: unittest.TestCase) -> bool:
        """Dot product of zero vectors is zero?"""
        self.assertEqual(loop.dot_prod(self._zero_vector, self._zero_vector), 0.0)

    def test_one_case_loop(self:unittest.TestCase) -> bool:
        """ dot product of provided vectors is expected"""
        self.assertEqual(loop.dot_prod(self._v1, self._v2), self._result)



class MatrixVectorProductTester(unittest.TestCase):

    def setUp(self: unittest.TestCase) -> None:
        self._identity_matrix = [[1, 0], [0, 2]]
        self._zero_vector = [0.0, 0.0]
        self._v1 = [1.0, 2.0]
        self._result = [1.0, 4.0]

    def tearDown(self: unittest.TestCase) -> None:
        self._identity_matrix = None
        self._zero_vector = None

    def test_zero_case(self: unittest.TestCase) -> bool:
        """Identity times zero matrix is zero"""
        self.assertEqual(matrix_vector_prod(self._identity_matrix,
                                            self._zero_vector),
                         self._zero_vector)

    def test_zero_case_loop(self: unittest.TestCase) -> bool:
        """Identity times zero matrix is zero"""
        self.assertEqual(loop.matrix_vector_prod(self._identity_matrix,
                                            self._zero_vector),
                         self._zero_vector)

    def test_one_case_loop(self: unittest.TestCase) -> bool:
        """maxtric vector product of provided matrix and vector is expected"""
        self.assertEqual(loop.matrix_vector_prod(self._identity_matrix, self._v1), self._result)



class PythagoreanTripleTester(unittest.TestCase):

    def test_10(self: unittest.TestCase) -> None:
        """triples up to 10"""
        self.assertEqual(set(pythagorean_triples(10)), {(3, 4, 5), (6, 8, 10)})
        # create a set in which triples are

    def test_10_loop(self: unittest.TestCase) -> None:
        """triples up to 10"""
        self.assertEqual(set(loop.pythagorean_triples(10)), {(3, 4, 5), (6, 8, 10)})

class AnyPythagoreanTripleTester(unittest.TestCase):

    def test_100_110(self: 'AnyPythagoreanTripleTester') -> bool:
        """triples in [100, 110]?"""
        self.assertFalse(any_pythagorean_triples(100, 110))

if __name__ == '__main__':
    unittest.main(exit=False)
