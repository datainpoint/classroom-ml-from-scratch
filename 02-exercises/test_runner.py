import unittest
import ipynb.fs.full.exercises as ex
import numpy as np
import pandas as pd

class TestExercisesTwo(unittest.TestCase):
    def test_01_extract_titanic_X_y(self):
        titanic_X, titanic_y = ex.extract_titanic_X_y()
        self.assertIsInstance(titanic_X, np.ndarray)
        self.assertIsInstance(titanic_y, np.ndarray)
        self.assertEqual(titanic_X.shape, (891, 11))
        self.assertEqual(titanic_y.shape, (891,))
    def test_02_extract_house_prices_X_y(self):
        house_prices_X, house_prices_y = ex.extract_house_prices_X_y()
        self.assertIsInstance(house_prices_X, np.ndarray)
        self.assertIsInstance(house_prices_y, np.ndarray)
        self.assertEqual(house_prices_X.shape, (1460, 80))
        self.assertEqual(house_prices_y.shape, (1460,))
    def test_03_get_standard_scaled_fare(self):
        standard_scaled_fare = ex.get_standard_scaled_fare()
        self.assertIsInstance(standard_scaled_fare, np.ndarray)
        self.assertEqual(standard_scaled_fare.shape, (891, 1))
        self.assertAlmostEqual(standard_scaled_fare.std(), 1.0)
        self.assertAlmostEqual(standard_scaled_fare.max(), 9.667166525013505)
        self.assertAlmostEqual(standard_scaled_fare.min(), -0.6484216535389205)
    def test_04_get_min_max_scaled_gr_liv_area(self):
        min_max_scaled_gr_liv_area = ex.get_min_max_scaled_gr_liv_area()
        self.assertIsInstance(min_max_scaled_gr_liv_area, np.ndarray)
        self.assertAlmostEqual(min_max_scaled_gr_liv_area.max(), 1.0)
        self.assertAlmostEqual(min_max_scaled_gr_liv_area.min(), 0.0)
    def test_05_get_polynomial_gr_liv_area(self):
        polynomial_gr_liv_area = ex.get_polynomial_gr_liv_area()
        self.assertIsInstance(polynomial_gr_liv_area, np.ndarray)
        self.assertEqual(polynomial_gr_liv_area.shape, (1460, 4))
        self.assertAlmostEqual(polynomial_gr_liv_area[:, 0].mean(), 1.0)
        self.assertAlmostEqual(polynomial_gr_liv_area[:, 1].mean(), 1515.463698630137)
        self.assertAlmostEqual(polynomial_gr_liv_area[:, 2].mean(), 2572570.7253424656)
        self.assertAlmostEqual(polynomial_gr_liv_area[:, 3].mean(), 4932874793.497945)

suite = unittest.TestLoader().loadTestsFromTestCase(TestExercisesTwo)
runner = unittest.TextTestRunner(verbosity=2)
if __name__ == '__main__':
    test_results = runner.run(suite)
number_of_failures = len(test_results.failures)
number_of_errors = len(test_results.errors)
number_of_test_runs = test_results.testsRun
number_of_successes = number_of_test_runs - (number_of_failures + number_of_errors)
print("You've got {} successes among {} questions.".format(number_of_successes, number_of_test_runs))
