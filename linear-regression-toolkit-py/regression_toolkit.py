# -*- coding: utf-8 -*-

# -- Sheet --

# **This is a toolkit lib file covering standard linear regression functions**


# let's import all necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import math, copy

# let's define the first function for calculating y predict value

def compute_dep_var(x, w, b):

    # an easier implementation would be to use a dot product np.dot(x, w) + b
    dot_prod = 0
    for i in range(x.shape[0]): # loop through all rows
        dot_prod += x[i] * w[i]
    
    dep_var = dot_prod + b

    return dep_var

def test_compute_dep_var():

    # define sample x training data
    x_train = np.array([952,2,1,65])
    w_train = np.array([0.0005,1,1,1])
    b_train = 500

    pred_val = compute_dep_var(x_train, w_train, b_train)

    print(f"Predicted value is: {pred_val}")


test_compute_dep_var()



