# Linear Regression Toolkit

**Description:**

This toolkit library provides a set of functions for implementing simple linear regression. Linear regression is a widely used method for modeling the relationship between a dependent variable (target) and one or more independent variables (features). This library is focused on simple linear regression, which deals with a single independent variable and a single dependent variable. It includes functions for making predictions, computing the cost function, and performing gradient descent to optimize model parameters.

**Table of Contents:**

- [Usage](#usage)
- [Functions](#functions)
- [Example Usage](#example-usage)
- [Notes](#notes)

## Usage

This library contains the following functions:

1. `compute_dep_var(x, w, b)`

   This function computes the dependent variable's predicted values based on the input independent feature data `x`, weight parameter `w`, and bias parameter `b`. It returns an array of predicted values.

2. `compute_cost_fn(x, y, w, b)`

   This function calculates the cost function, which represents the error in the model's predictions. It uses the sum of squared errors between predicted and actual dependent variable values.

3. `compute_deriv_fn(x, y, w, b)`

   This function computes the derivatives of the weight and bias parameters, which are used in gradient descent to update the model parameters.

4. `compute_gradient_descent(x, y, n, a, w, b)`

   This function performs gradient descent to optimize the weight and bias parameters for the linear regression model. It returns the final optimized weight and bias, as well as a history of cost function values over iterations.

## Example Usage

Here's an example of how you can use these functions to perform linear regression:

```python
import numpy as np

# Sample data
x = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

# Initial weight and bias
w_initial = 1
b_initial = 0

# Number of iterations and learning rate
iterations = 100
learning_rate = 0.01

# Perform gradient descent
final_w, final_b, cost_history = compute_gradient_descent(x, y, iterations, learning_rate, w_initial, b_initial)

print("Optimized weight:", final_w)
print("Optimized bias:", final_b)
