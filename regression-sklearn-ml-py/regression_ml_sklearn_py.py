# -*- coding: utf-8 -*-
"""
Regression ML SkLearn Python script
Author: Garry Singh

This script covers regression computations in python using Scikit Learn library

"""

# Importing necessary libraries
import numpy as np
from sklearn.linear_model import SGDRegressor # lib for computing gradient descent
from sklearn.preprocessing import StandardScaler # lib for computing z score normalized features (feature scaling)
import matplotlib.pyplot as plt

# let's prepare sample input data

# Create a NumPy array with feature names as strings
x_features = np.array(["Square Feet", "Bedrooms", "Bathrooms", "Neighborhood", "Garage"])

# Sample input data as a 2D NumPy array
x_train = np.array([
    [1500, 3, 2, 1, 1],
    [1800, 4, 2.5, 2, 0],
    [1200, 2, 1, 0, 0],
    [2100, 4, 3, 1, 1],
    [1600, 3, 2.5, 2, 1],
    [1400, 2, 1.5, 1, 0],
    [2500, 4, 3.5, 2, 1]
])

# Corresponding prices as a 1D NumPy array
y_train = np.array([250000, 320000, 180000, 400000, 300000, 220000, 450000])

# let's normalize the given input training data
scalar = StandardScaler()
x_norm = scalar.fit_transform(x_train)

#print(f"Normalized housing feature data is: \n {x_norm}")

# let's try to fit the normalized feature data with given price training values
sgdr = SGDRegressor(max_iter=100000)
sgdr.fit(x_norm, y_train)

n_iter_ran = sgdr.n_iter_
n_weight_updates = sgdr.t_
b_norm = sgdr.intercept_ 
w_norm = sgdr.coef_

#print(f"Number of iterations used: {n_iter_ran}")
#print(f"Number of updates completed to weight: {n_weight_updates}")

#print(f"Normalized bias is: {b_norm} ")
#print(f"Normalized weight is: {w_norm} ")

# now let's try to predict price for given denormalized data

y_pred_sgd = sgdr.predict(x_norm)
y_pred_dot = np.dot(x_norm, w_norm) + b_norm

for i in range(x_train.shape[1]):
    plt.scatter(x_train[:,i], y_train,marker='o', color='g')
    plt.ylabel("House price in $")
    plt.xlabel(x_features[i])
    plt.show()