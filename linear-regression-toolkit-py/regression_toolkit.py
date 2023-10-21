# -*- coding: utf-8 -*-
"""
Regression Toolkit Functions
Author: Garry Singh

This is a toolkit library for standard linear regression functions.
It includes functions for predicting dependent variable values, calculating cost functions, derivatives of weights and bias, and performing gradient descent.

"""

# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math, copy
import unittest

# Set the desired display precision for NumPy arrays
np.set_printoptions(precision=None, suppress=True)

# Function for predicting dependent variable value
def compute_dep_var(x, w, b):
    """
    Predicts the dependent variable values.

    Args:
        x (ndarray): Independent feature training data.
        w (ndarray): Weight parameter.
        b (float): Bias parameter.

    Returns:
        ndarray: Predicted dependent variable values.
    """
    x_size = x.shape[0]
    x_features = x.shape[1]
    dot_prod = np.zeros(x_size)
    for i in range(x_size):
        dot_sum = 0
        for j in range(x_features):
            dot_sum += x[i, j] * w[j]
        dot_prod[i] = dot_sum + b
    return dot_prod

# Function for calculating the cost function
def compute_cost_fn(x, y, w, b):
    """
    Calculates the cost function.

    Args:
        x (ndarray): Independent feature training data.
        y (ndarray): Computed/dependent feature training data.
        w (ndarray): Weight parameter.
        b (float): Bias parameter.

    Returns:
        float: Cost function value.
    """
    x_size = x.shape[0]
    error = 0
    for i in range(x_size):
        pred_y_val = np.dot(x[i], w) + b
        error += np.square(pred_y_val - y[i])

    cost_fn_val = error / (2 * x_size)
    return cost_fn_val

# Function for computing derivatives of weight and bias
def compute_deriv_fn(x, y, w, b):
    """
    Computes the derivatives of weight and bias.

    Args:
        x (ndarray): Independent feature training data.
        y (ndarray): Computed/dependent feature training data.
        w (ndarray): Weight parameter.
        b (float): Bias parameter.

    Returns:
        ndarray: Derivative of weights.
        float: Derivative of bias.
    """
    size_x = x.shape[0]
    features_x = x.shape[1]
    deriv_w = np.zeros(features_x)
    deriv_b = 0

    for i in range(size_x):
        pred_dep_val = np.dot(x[i], w) + b
        error = pred_dep_val - y[i]
        for j in range(features_x):
            deriv_w[j] += error * x[i, j]
        deriv_b += error

    return deriv_w / size_x, deriv_b / size_x

# Function for computing gradient descent
def compute_gradient_descent(x, y, n, a, w, b):
    """
    Computes gradient descent based on a specific number of iterations with a history of the cost function.

    Args:
        x (ndarray): Independent feature training data.
        y (ndarray): Computed/dependent feature training data.
        n (int): Number of iterations.
        a (float): Learning rate.
        w (ndarray): Weight parameter.
        b (float): Bias parameter.

    Returns:
        ndarray: Updated weight parameter.
        float: Updated bias parameter.
        ndarray: History of cost function values for each iteration.
    """
    size_x = x.shape[0]
    features_x = x.shape[1]
    w_final = copy.deepcopy(w)
    b_final = b
    j_history = np.zeros(n)

    for iter in range(n):
        deriv_w = np.zeros(features_x)
        deriv_b = 0
        cost = 0
        for i in range(size_x):
            pred_dep_val = np.dot(x[i], w_final) + b_final
            error = pred_dep_val - y[i]
            for j in range(features_x):
                deriv_w[j] += error * x[i, j]
            deriv_b += error
            cost += np.square(error)
        j_history[iter] = cost / (2 * size_x)
        deriv_w = deriv_w / size_x
        deriv_b = deriv_b / size_x
        w_final = w_final - a * deriv_w
        b_final = b_final - a * deriv_b

    return w_final, b_final, j_history

# Function for feature scaling using Z score normalization
def compute_feature_scaling(x):
    """
    Performs feature scaling using Z score normalization.

    Args:
        x (ndarray): Input data.

    Returns:
        ndarray: Scaled input data.
    """
    mu = np.mean(x, axis=0)
    sigma = np.std(x, axis=0)
    scaled_x = (x - mu) / sigma
    return scaled_x