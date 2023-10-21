# -*- coding: utf-8 -*-

# -- Sheet --

# **This is a toolkit lib file covering standard linear regression functions**


# let's import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math, copy
import unittest
np.set_printoptions(precision=None, suppress=True)


# let's define the first function for predicting dependant variable value

def compute_dep_var(x, w, b):

    # Args:
        #   x => independent feature training data
        #   w => weight parameter
        #   b => bias parameter

    # dot  product is: 

    # an easier implementation would be to use a dot product np.dot(x, w) + b
    x_size = x.shape[0]
    x_features = x.shape[1]
    dot_prod = np.zeros(x_size)
    for i in range(x_size): # loop through all rows
        dot_sum = 0
        for j in range(x_features):
            dot_sum += x[i,j] * w[j]
        dot_prod[i] = dot_sum + b
        
    return dot_prod

# now let's define the cost function for calculating cost of error
# formula for calculating cost function is sum of (square of each training example (pred_dep_var - act_dep_var))/2 * count_training_data

def compute_cost_fn(x, y, w, b):

    # Args:
        #   x => independent feature training data
        #   y => computed/dependent feature training data
        #   w => weight parameter
        #   b => bias parameter

    x_size = x.shape[0]
    error = 0
    for i in range(x_size):
        pred_y_val = np.dot(x[i], w) + b
        error += np.square(pred_y_val - y[i])

    cost_fn_val = error/(2*x_size)

    return cost_fn_val

# now let's define the derivative function for calculating derivative of weight and bias
# derivative of weight is defined as (pred_dep_var - act_dep_var) * train_feat_val / total_train_count
def compute_deriv_fn(x, y, w, b):

    # Args:
        #   x => independent feature training data
        #   y => computed/dependent feature training data
        #   w => weight parameter
        #   b => bias parameter

    size_x = x.shape[0]
    features_x = x.shape[1]
    deriv_w = np.zeros(features_x)
    deriv_b = 0

    for i in range(size_x):
        pred_dep_val = np.dot(x[i], w) + b
        error = pred_dep_val - y[i]
        for j in range(features_x):
            deriv_w[j] += error * x[i,j]
        deriv_b += error

    return deriv_w/size_x, deriv_b/size_x

# now let's define function for computing gradient descent based on specific iterations with history of cost function

def compute_gradient_descent(x, y, n, a, w, b):
    # Args:
        #   x => independent feature training data
        #   y => computed/dependent feature training data
        #   a => learning rate
        #   n => number of iterations
        #   w => weight parameter
        #   b => bias parameter

    size_x = x.shape[0]
    features_x = x.shape[1]
    w_final = copy.deepcopy(w)
    b_final = b
    j_history = np.zeros(n)

    for iter in range(n):
        deriv_w = np.zeros(features_x)
        deriv_b = 0
        cost = 0
    
        #for i in range(size_x):
        #    pred_dep_val = np.dot(x[i], w_final) + b_final
        #    error = pred_dep_val - y[i]
        #    for j in range(features_x):
        #        deriv_w[j] += error * x[i,j]
        #    deriv_b += error
        #    cost += np.square(error)

        #j_history[iter] = cost/(2*size_x)
        j_history[iter] = compute_cost_fn(x,y,w_final,b_final)

        #deriv_w = deriv_w/size_x
        #deriv_b = deriv_b/size_x
        deriv_w, deriv_b = compute_deriv_fn(x,y,w_final,b_final)

        #print(f"when w is {w_final} and b is {b_final} and deriv_w is: {deriv_w} and deriv_b is: {deriv_b} then cost is: {j_history[iter]}")

        w_final = w_final - (a * deriv_w)
        b_final = b_final - (a * deriv_b)


    return w_final, b_final, j_history