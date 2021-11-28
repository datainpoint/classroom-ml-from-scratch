import unittest
import ipynb.fs.full.exercises as ex
import numpy as np

class TestExercisesOne(unittest.TestCase):
    def test_01_add_intercepts(self):
        A = np.array([5, 5, 6, 6]).reshape(-1, 1)
        np.testing.assert_array_equal(ex.add_intercepts(A),
        np.array([[1, 5],
                  [1, 5],
                  [1, 6],
                  [1, 6]]))
        B = np.zeros((5, 2), dtype=int)
        np.testing.assert_array_equal(ex.add_intercepts(B),
        np.array([[1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0],
                  [1, 0, 0]]))
        C = np.array([5, 6, 7, 8]).reshape(-1, 1)
        np.testing.assert_array_equal(ex.add_intercepts(C),
        np.array([[1, 5],
                  [1, 6],
                  [1, 7],
                  [1, 8]]))
    def test_02_split_train_test(self):
        A = np.ones((10, 2))
        A_train, A_test = ex.split_train_test(A, test_size=0.3)
        self.assertEqual(A_train.shape, (7, 2))
        self.assertEqual(A_test.shape, (3, 2))
        B = np.ones((20, 3))
        B_train, B_test = ex.split_train_test(B, test_size=0.4)
        self.assertEqual(B_train.shape, (12, 3))
        self.assertEqual(B_test.shape, (8, 3))
        C = np.ones((100, 4))
        C_train, C_test = ex.split_train_test(C, test_size=0.2)
        self.assertEqual(C_train.shape, (80, 4))
        self.assertEqual(C_test.shape, (20, 4))
    def test_03_is_invertible(self):
        A = np.array([1, 2, 2, 4]).reshape(2, 2)
        self.assertFalse(ex.is_invertible(A))
        B = np.array([5, 5, 6, 6]).reshape(2, 2)
        self.assertFalse(ex.is_invertible(B))
        C = np.array([5, 6, 7, 8]).reshape(2, 2)
        self.assertTrue(ex.is_invertible(C))
        D = np.array([7, 8, 9, 10]).reshape(2, 2)
        self.assertTrue(ex.is_invertible(D))
        E = np.array([5, 10, 10, 20]).reshape(2, 2)
        self.assertFalse(ex.is_invertible(E))
    def test_04_create_diagonal_split_matrix(self):
        np.testing.assert_array_equal(ex.create_diagonal_split_matrix(2, 5566),
        np.array([[   0, 5566],
                  [5566,    0]]))
        np.testing.assert_array_equal(ex.create_diagonal_split_matrix(3, 55),
        np.array([[ 0, 55, 55],
                  [55,  0, 55],
                  [55, 55,  0]]))
        np.testing.assert_array_equal(ex.create_diagonal_split_matrix(4, 66),
        np.array([[ 0, 66, 66, 66],
                  [66,  0, 66, 66],
                  [66, 66,  0, 66],
                  [66, 66, 66,  0]]))
    def test_05_create_square_matrix(self):
        np.testing.assert_array_equal(ex.create_square_matrix(3),
        np.array([[1, 2, 3],
                  [2, 4, 6],
                  [3, 6, 9]]))
        np.testing.assert_array_equal(ex.create_square_matrix(4),
        np.array([[ 1,  2,  3,  4],
                  [ 2,  4,  6,  8],
                  [ 3,  6,  9, 12],
                  [ 4,  8, 12, 16]]))
        np.testing.assert_array_equal(ex.create_square_matrix(5),
        np.array([[ 1,  2,  3,  4,  5],
                  [ 2,  4,  6,  8, 10],
                  [ 3,  6,  9, 12, 15],
                  [ 4,  8, 12, 16, 20],
                  [ 5, 10, 15, 20, 25]]))

suite = unittest.TestLoader().loadTestsFromTestCase(TestExercisesOne)
runner = unittest.TextTestRunner(verbosity=2)
if __name__ == '__main__':
    test_results = runner.run(suite)
number_of_failures = len(test_results.failures)
number_of_errors = len(test_results.errors)
number_of_test_runs = test_results.testsRun
number_of_successes = number_of_test_runs - (number_of_failures + number_of_errors)
print("You've got {} successes among {} questions.".format(number_of_successes, number_of_test_runs))
