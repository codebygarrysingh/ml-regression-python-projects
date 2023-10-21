# -*- coding: utf-8 -*-
"""
Regression Toolkit Test
Author: Garry Singh

This is a unit test file for the Regression Toolkit library.
It covers standard linear regression functions.

"""

# Importing necessary libraries
import numpy as np
import unittest

# Import functions to be tested from the Regression Toolkit
from regression_toolkit import compute_dep_var, compute_cost_fn, compute_deriv_fn, compute_gradient_descent, compute_feature_scaling

# Set the desired display precision for NumPy arrays
np.set_printoptions(precision=2, suppress=True)

class TestRegressionToolkit(unittest.TestCase):

    # Test for computing the dependent variable
    def test_compute_dep_var(self):

        # Sample training data (features)
        x_train = np.array([
            [1500, 3, 2, 10],   # House 1
            [2000, 4, 3, 15],   # House 2
            [1200, 2, 1, 5],    # House 3
            [1800, 3, 2, 12]    # House 4
        ])

        # Sample weight (coefficients) and bias (intercept) values (for demonstration)
        w_train = np.array([10, 200, 150, 80])  # For each feature
        b_train = 500

        # Call the function to compute the dependent variable
        actual_pred_val = compute_dep_var(x_train, w_train, b_train)

        # Calculate the expected output
        expected_pred_val = np.dot(x_train, w_train) + b_train

        # Assert that the result matches the expected output
        self.assertTrue(np.array_equal(actual_pred_val, expected_pred_val))

    # Test for computing the cost function
    def test_compute_cost_fn(self):

        # Define sample x training data
        x_train = np.array([[952,2,1,65]])
        y_train = np.array([200])
        w_train = np.array([0.0005,1,1,1])
        b_train = 500

        # Call the function to compute the cost function
        actual_cost_fn_val = compute_cost_fn(x_train, y_train, w_train, b_train)

        # Calculate the expected cost function value
        expected_cost_fn_val = np.mean(np.square((np.dot(x_train,w_train) + b_train) - y_train))/2

        # Assert that the result matches the expected output
        self.assertEqual(actual_cost_fn_val, expected_cost_fn_val)

    # Test for computing the derivative of w and b
    def test_compute_deriv_fn(self):

        # Define sample x training data
        x_train = np.array([[952,2,1,65],[1200,1,5,70]])
        y_train = np.array([200,400])
        w_train = np.array([0.0005,1,1,1])
        b_train = 500

        # Call the function to compute the derivatives
        actual_deriv_w, actual_deriv_b = compute_deriv_fn(x_train, y_train, w_train, b_train)

        # Define the expected derivatives
        expected_deriv_w = np.array([281354.576, 456.776, 625.738, 18156.47])
        expected_deriv_b = 272.538

        # Assert that the results match the expected derivatives
        self.assertTrue(np.array_equal(actual_deriv_w, expected_deriv_w))
        self.assertEqual(actual_deriv_b, expected_deriv_b)

    # Test for computing gradient descent
    def test_gradient_descent_fn(self):

        # Define sample x training data
        x_train = np.array([[952,2,1,65],[1244,3,2,64],[1947,3,2,17]])
        y_train = np.array([271.5,232,509.8])

        # Sample weight (coefficients) and bias (intercept) values (for demonstration)
        a = 1.0e-6
        w_train = np.array([5.0e-01,9.1e-04,4.7e-04,1.1e-02])
        b_train = 3.3e-04
        n = 1000

        # Call the function to compute gradient descent
        actual_w, actual_b, actual_j_hist = compute_gradient_descent(x_train, y_train, n, a, w_train, b_train)

        # Define the expected values
        expected_w = np.array([12828617112934059875215437266419712.00,
                            23580630276533951393997833371648.00,
                            15067775271686064618375265386496.00,
                            358891474068332064858982792036352.00], dtype=float)
        expected_b = 8.512855004847887e+30
        expected_j_hist = 1.4621506518459783e+74

        # Assert that the results match the expected values
        self.assertTrue(np.allclose(actual_w, expected_w, rtol=1e-5, atol=1e-8))
        self.assertEqual(actual_b, expected_b)
        self.assertEqual(actual_j_hist[n-1], expected_j_hist)

    # Test for feature scaling
    def test_compute_feature_scaling(self):
        # Define sample x training data
        x_train  = np.array([[1250,140,500],[1500,200,700]])

        # Call the function to compute feature scaling
        actual_scaled_x  = compute_feature_scaling(x_train)

        # Define the expected scaled values
        expected_scaled_x  = np.array([[-1,-1,-1],[1,1,1]], dtype=float)

        # Assert that the results match the expected values
        self.assertTrue(np.array_equal(actual_scaled_x, expected_scaled_x))

# Create a test suite and run the tests
test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressionToolkit)
unittest.TextTestRunner(verbosity=2).run(test_suite)
