import unittest
import numpy as np

def change_of_basis(A, Q):
    return np.dot(np.dot(np.linalg.inv(Q), A), Q)

class TestJCF(unittest.TestCase):

    # 7.2.a
    def test_1(self):
        Q1 = np.array([[1, 0], [1, 1]])
        Q2 = np.array([[1, -1], [1, 0]])
        A = np.array([[1, 1], [-1, 3]])
        result = np.array([[2, 1], [0, 2]])
        np.testing.assert_array_equal(change_of_basis(A, Q1), result)
        np.testing.assert_array_equal(change_of_basis(A, Q2), result)
        # idea is the choice of end vector in this case does not matter

    # 7.2.b
    def test_2(self):
        Q = np.array([[2, 1], [3, -1]])
        A = np.array([[1, 2], [3, 2]])
        result = np.array([[4, 0], [0, -1]])
        self.assertTrue(np.allclose(change_of_basis(A, Q), result))

    # 7.3.a
    def test_3(self):
        Q = np.array([[1,0,0], [0,-1,0], [0,0,0.5]])
        A = np.array([[2,-1,0],[0,2,-2],[0,0,2]])
        result = np.array([
                [2,1,0],
                [0,2,1],
                [0,0,2]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q), result)

    # 7.3.b
    def test_4(self):
        A = np.array([[0,1,0,0,0],[0,0,2,0,0],[0,0,0,0,0], [0,0,0,1,1], [0,0,0,0,1]])
        Q = np.array([[1,0,0,0,0], [0,1,0,0,0], [0,0,0.5,0,0], [0,0,0,1,0], [0,0,0,0,1]])
        result = np.array([
                [0,1,0,0,0],
                [0,0,1,0,0],
                [0,0,0,0,0],
                [0,0,0,1,1],
                [0,0,0,0,1]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q), result)

    # 7.3.d
    def test_5(self):
        A = np.array([
                [3, 0, 0, 0],
                [0, 2, 1, 0],
                [0, 1, 2, 0],
                [0, 0, 0, 3]
            ])
        Q1 = np.array([
                [1, 0, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, -1],
                [0, 0, 1, 0]
            ])
        Q2 = np.array([
                [0,1,0,0],
                [0,0,1,1],
                [0,0,1,-1],
                [1,0,0,0]
            ])
        result = np.array([
                [3, 0, 0, 0],
                [0, 3, 0, 0],
                [0, 0, 3, 0],
                [0, 0, 0, 1]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q2), result)

    # example in class
    def test_6(self):
        A = np.array([
                [4,-2,2,0,2],
                [1,0,1,0,3],
                [-1,2,5,0,-1],
                [0,2,0,4,-2],
                [1,-4,1,0,7]
            ])
        # lambda = 4 with m = 5
        U = A - 4 * np.eye(5)  # dim U = 2
        U2 = np.dot(U, U)      # dim U^2 = 4

        # dot diagram as follows
        #   . .
        #   . . 
        #   .

        v1 = np.array([1, 0, 0, 0, 0])  # not in N(U) and N(U^2) is end vector v1
        Uv1 = np.dot(U, v1)
        UUv1 = np.dot(U, Uv1)

        #  print("\n{}".format(U))
        #  print("\n{}".format(U2))
        #  print("\n{}".format(Uv1))
        #  print("\n{}".format(UUv1))
    
    # 7.2 4.a
    def test_7(self):
        A = np.array([
                [-3,3,-2],
                [-7,6,-3],
                [1,-1,2]
            ])
        Q = np.array([
                [-1,-1,1],
                [-1,-2,2],
                [1,0,1]
            ])
        result = np.array([
                [2,1,0],
                [0,2,0],
                [0,0,1]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q), result)
        
    # 7.2 4.b
    def test_8(self):
        A = np.array([
                [0,1,-1],
                [-4,4,-2],
                [-2,1,1]
            ])
        Q = np.array([
                [1,1,-1],
                [2,2,0],
                [1,0,2]
            ])
        result = np.array([
                [1,0,0],
                [0,2,0],
                [0,0,2]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q), result)

    # 7.2 4.c
    def test_9(self):
        A = np.array([
                [0,-1,-1],
                [-3,-1,-2],
                [7,5,6]
            ])
        Q = np.array([
                [0,-1,2/3.],
                [-1,-1,-1/3.],
                [1,3,0]
            ], dtype=float)
        result = np.array([
                [1,0,0],
                [0,2,1],
                [0,0,2]
            ], dtype=float)
        self.assertTrue(np.allclose(change_of_basis(A, Q), result))

    # 7.2 4.d
    def test_9(self):
        A = np.array([
                [0,-3,1,2],
                [-2,1,-1,2],
                [-2,1,-1,2],
                [-2,-3,1,4]
            ])
        Q = np.array([
                [-1,1,1,0],
                [1,0,1,-1],
                [1,0,1,-2],
                [0,1,1,0]
            ])
        # idea: cycle has to be in order of [initial_vector, ..., end_vector]
        #  Q = np.array([
        #          [-1,1,0,1],
        #          [1,0,-1,1],
        #          [1,0,-2,1],
        #          [0,1,0,1]
        #      ])
        # 
        result = np.array([
                [2,0,0,0],
                [0,2,0,0],
                [0,0,0,1],
                [0,0,0,0]
            ])
        np.testing.assert_array_equal(change_of_basis(A, Q), result)

if __name__ == '__main__':
    unittest.main()







