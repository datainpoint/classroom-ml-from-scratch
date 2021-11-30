import unittest
import ipynb.fs.full.exercises as ex
import numpy as np
import pandas as pd

class TestExercisesFour(unittest.TestCase):
    def test_01_predict_survived_chimp_model(self):
        survived_chimp_model = ex.predict_survived_chimp_model()
        self.assertIsInstance(survived_chimp_model, np.ndarray)
        self.assertEqual(survived_chimp_model.shape, (418,))
    def test_02_predict_survived_expert_model(self):
        survived_expert_model = ex.predict_survived_expert_model()
        self.assertIsInstance(survived_expert_model, np.ndarray)
        self.assertEqual(survived_expert_model.shape, (418,))
    def test_03_predict_survived_sklearn_model(self):
        survived_sklearn_model = ex.predict_survived_sklearn_model()
        self.assertIsInstance(survived_sklearn_model, np.ndarray)
        np.testing.assert_almost_equal(survived_sklearn_model,
                                      np.array([ 4.73195566, -1.16846277, -2.61196369, -0.03342722]))
    def test_04_predict_survived_gradient_descent_model(self):
        survived_gradient_descent_model = ex.predict_survived_gradient_descent_model()
        self.assertIsInstance(survived_gradient_descent_model, np.ndarray)
        self.assertEqual(survived_gradient_descent_model.size, (4,))
        self.assertTrue(survived_gradient_descent_model[1] < 0)
        self.assertTrue(survived_gradient_descent_model[2] < 0)
        self.assertTrue(survived_gradient_descent_model[3] < 0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestExercisesFour)
runner = unittest.TextTestRunner(verbosity=2)
if __name__ == '__main__':
    test_results = runner.run(suite)
number_of_failures = len(test_results.failures)
number_of_errors = len(test_results.errors)
number_of_test_runs = test_results.testsRun
number_of_successes = number_of_test_runs - (number_of_failures + number_of_errors)
print("You've got {} successes among {} questions.".format(number_of_successes, number_of_test_runs))
