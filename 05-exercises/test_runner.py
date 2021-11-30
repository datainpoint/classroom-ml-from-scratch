import unittest
import ipynb.fs.full.exercises as ex
import numpy as np

class TestExercisesFive(unittest.TestCase):
    def test_01_MeanError(self):
        y = np.array([5, 5, 6, 6])
        y_hat = np.array([5, 5, 6, 6])
        me = ex.MeanError(y, y_hat)
        self.assertAlmostEqual(me.get_mse(), 0.0)
        self.assertAlmostEqual(me.get_mae(), 0.0)
        y_hat = np.array([5, 6, 7, 8])
        me = ex.MeanError(y, y_hat)
        self.assertAlmostEqual(me.get_mse(), 1.5)
        self.assertAlmostEqual(me.get_mae(), 1.0)
        y_hat = np.array([-5, -5, -6, -6])
        me = ex.MeanError(y, y_hat)
        self.assertAlmostEqual(me.get_mse(), 122.0)
        self.assertAlmostEqual(me.get_mae(), 11.0)
    def test_02_get_confusion_matrix(self):
        np.random.seed(0)
        y = np.random.randint(0, 2, size=100)
        np.random.seed(1)
        y_hat = np.random.randint(0, 2, size=100)
        np.testing.assert_array_equal(ex.get_confusion_matrix(y, y_hat),
        np.array([[21, 23],
                  [24, 32]]))
        np.random.seed(2)
        y = np.random.randint(0, 2, size=100)
        np.random.seed(3)
        y_hat = np.random.randint(0, 2, size=100)
        np.testing.assert_array_equal(ex.get_confusion_matrix(y, y_hat),
        np.array([[27, 28],
                  [23, 22]]))

suite = unittest.TestLoader().loadTestsFromTestCase(TestExercisesFive)
runner = unittest.TextTestRunner(verbosity=2)
if __name__ == '__main__':
    test_results = runner.run(suite)
number_of_failures = len(test_results.failures)
number_of_errors = len(test_results.errors)
number_of_test_runs = test_results.testsRun
number_of_successes = number_of_test_runs - (number_of_failures + number_of_errors)
print("You've got {} successes among {} questions.".format(number_of_successes, number_of_test_runs))
