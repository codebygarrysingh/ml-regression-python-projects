# -*- coding: utf-8 -*-

# -- Sheet --

# **This is a toolkit lib file covering standard linear regression functions**


# let's import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math, copy
import unittest
from regression_toolkit import compute_dep_var, compute_cost_fn, compute_deriv_fn, compute_gradient_descent
np.set_printoptions(precision=None, suppress=True)

class TestRegressionToolkit(unittest.TestCase):


    # callable test for computing dependent variable 
    def test_compute_dep_var(self):

        # Training data (features)
        x_train = np.array([
            [1500, 3, 2, 10],   # House 1
            [2000, 4, 3, 15],   # House 2
            [1200, 2, 1, 5],    # House 3
            [1800, 3, 2, 12]    # House 4
        ])

        # Weight (coefficients) and bias (intercept) values (for demonstration)
        w_train = np.array([10, 200, 150, 80])  # For each feature
        b_train = 500

        actual_pred_val = compute_dep_var(x_train, w_train, b_train)

        expected_pred_val = np.dot(x_train, w_train) + b_train

        self.assertTrue(np.array_equal(actual_pred_val, expected_pred_val))

    # callable test for computing cost function
    def test_compute_cost_fn(self):

        # define sample x training data
        x_train = np.array([[952,2,1,65]])
        y_train = np.array([200])
        w_train = np.array([0.0005,1,1,1])
        b_train = 500

        actual_cost_fn_val = compute_cost_fn(x_train, y_train, w_train, b_train)

        expected_cost_fn_val = np.mean(np.square((np.dot(x_train,w_train) + b_train) - y_train))/2

        self.assertEqual(actual_cost_fn_val, expected_cost_fn_val)

    # callable test for computing derivative of w and b
    def test_compute_deriv_fn(self):

        # define sample x training data
        x_train = np.array([[952,2,1,65],[1200,1,5,70]])
        y_train = np.array([200,400])
        w_train = np.array([0.0005,1,1,1])
        b_train = 500

        actual_deriv_w, actual_deriv_b = compute_deriv_fn(x_train, y_train, w_train, b_train)

        expected_deriv_w = np.array([281354.576, 456.776, 625.738, 18156.47])
        expected_deriv_b = 272.538

        self.assertTrue(np.array_equal(actual_deriv_w, expected_deriv_w))
        self.assertEqual(actual_deriv_b, expected_deriv_b)

     # callable test for computing gradient descent 
    def test_gradient_descent_fn(self):

        # define sample x training data
        x_train = np.array([[952,2,1,65],[1244,3,2,64],[1947,3,2,17]])
        y_train = np.array([271.5,232,509.8])

        # Weight (coefficients) and bias (intercept) values (for demonstration)
        a = 1.0e-6
        w_train = np.array([5.0e-01,9.1e-04,4.7e-04,1.1e-02])
        b_train = 3.3e-04
        n = 1000

        actual_w, actual_b, actual_j_hist = compute_gradient_descent(x_train, y_train, n, a, w_train, b_train)

        np.set_printoptions(suppress=True, formatter={'all': lambda x: f'{x:.15f}'})

        print(f"Actual w is: {actual_w}")

        expected_w = np.array([12828617112934059875215437266419712.00,
                            23580630276533951393997833371648.00,
                            15067775271686064618375265386496.00,
                            358891474068332064858982792036352.00])
        expected_b = 8.512855004847887e+30
        expected_j_hist = 1.4621506518459783e+74

        self.assertTrue(np.allclose(actual_w, expected_w, rtol=1e-5, atol=1e-8))
        self.assertEqual(actual_b, expected_b)
        self.assertEqual(actual_j_hist[n-1], expected_j_hist)

test_suite = unittest.TestLoader().loadTestsFromTestCase(TestRegressionToolkit)
unittest.TextTestRunner(verbosity=2).run(test_suite)





