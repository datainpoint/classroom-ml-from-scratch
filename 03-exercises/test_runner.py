import unittest
import ipynb.fs.full.exercises as ex
import numpy as np
import pandas as pd

class TestExercisesThree(unittest.TestCase):
    def test_01_predict_sale_price_chimp_model(self):
        sale_price_chimp_model = ex.predict_sale_price_chimp_model()
        self.assertIsInstance(sale_price_chimp_model, np.ndarray)
        self.assertEqual(sale_price_chimp_model.shape, (1459,))
    def test_02_predict_sale_price_expert_model(self):
        sale_price_expert_model = ex.predict_sale_price_expert_model()
        self.assertIsInstance(sale_price_expert_model, np.ndarray)
        self.assertEqual(sale_price_expert_model.shape, (1459,))
    def test_03_predict_sale_price_sklearn_model(self):
        sale_price_sklearn_model = ex.predict_sale_price_sklearn_model()
        self.assertIsInstance(sale_price_sklearn_model, np.ndarray)
        self.assertEqual(sale_price_sklearn_model.size, 2)
        self.assertTrue(sale_price_sklearn_model[0] > 0)
        self.assertTrue(sale_price_sklearn_model[1] > 0)
    def test_04_predict_sale_price_normal_equation_model(self):
        sale_price_normal_equation_model = ex.predict_sale_price_normal_equation_model()
        self.assertIsInstance(sale_price_normal_equation_model, np.ndarray)
        self.assertEqual(sale_price_normal_equation_model.size, 2)
        self.assertTrue(sale_price_normal_equation_model[0] > 0)
        self.assertTrue(sale_price_normal_equation_model[1] > 0)
    def test_05_predict_height_meters_sklearn_model(self):
        height_meters_sklearn_model = ex.predict_height_meters_sklearn_model()
        self.assertIsInstance(height_meters_sklearn_model, np.ndarray)
        self.assertEqual(height_meters_sklearn_model.size, 3)
        self.assertTrue(height_meters_sklearn_model[0] > 0)
        self.assertTrue(height_meters_sklearn_model[1] < 0)
        self.assertTrue(height_meters_sklearn_model[2] > 0)
    def test_06_predict_height_meters_gradient_descent_model(self):
        height_meters_gradient_descent_model = ex.predict_height_meters_gradient_descent_model()
        self.assertIsInstance(height_meters_sklearn_model, np.ndarray)
        self.assertEqual(height_meters_sklearn_model.size, 3)
        self.assertTrue(height_meters_sklearn_model[0] > 0)
        self.assertTrue(height_meters_sklearn_model[1] < 0)
        self.assertTrue(height_meters_sklearn_model[2] > 0)

suite = unittest.TestLoader().loadTestsFromTestCase(TestExercisesThree)
runner = unittest.TextTestRunner(verbosity=2)
if __name__ == '__main__':
    test_results = runner.run(suite)
number_of_failures = len(test_results.failures)
number_of_errors = len(test_results.errors)
number_of_test_runs = test_results.testsRun
number_of_successes = number_of_test_runs - (number_of_failures + number_of_errors)
print("You've got {} successes among {} questions.".format(number_of_successes, number_of_test_runs))
