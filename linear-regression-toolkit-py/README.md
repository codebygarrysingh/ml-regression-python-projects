# Regression Toolkit Functions

Author: Garry Singh

This is a toolkit library for standard linear regression functions. It includes functions for predicting dependent variable values, calculating cost functions, derivatives of weights and bias, and performing gradient descent.

## Functions

### `compute_dep_var(x, w, b)`

Predicts the dependent variable values.

- Args:
  - `x` (ndarray): Independent feature training data.
  - `w` (ndarray): Weight parameter.
  - `b` (float): Bias parameter.

- Returns:
  - `ndarray`: Predicted dependent variable values.

### `compute_cost_fn(x, y, w, b)`

Calculates the cost function.

- Args:
  - `x` (ndarray): Independent feature training data.
  - `y` (ndarray): Computed/dependent feature training data.
  - `w` (ndarray): Weight parameter.
  - `b` (float): Bias parameter.

- Returns:
  - `float`: Cost function value.

### `compute_deriv_fn(x, y, w, b)`

Computes the derivatives of weight and bias.

- Args:
  - `x` (ndarray): Independent feature training data.
  - `y` (ndarray): Computed/dependent feature training data.
  - `w` (ndarray): Weight parameter.
  - `b` (float): Bias parameter.

- Returns:
  - `ndarray`: Derivative of weights.
  - `float`: Derivative of bias.

### `compute_gradient_descent(x, y, n, a, w, b)`

Computes gradient descent based on a specific number of iterations with a history of the cost function.

- Args:
  - `x` (ndarray): Independent feature training data.
  - `y` (ndarray): Computed/dependent feature training data.
  - `n` (int): Number of iterations.
  - `a` (float): Learning rate.
  - `w` (ndarray): Weight parameter.
  - `b` (float): Bias parameter.

- Returns:
  - `ndarray`: Updated weight parameter.
  - `float`: Updated bias parameter.
  - `ndarray`: History of cost function values for each iteration.

### `compute_feature_scaling(x)`

Performs feature scaling using Z score normalization.

- Args:
  - `x` (ndarray): Input data.

- Returns:
  - `ndarray`: Scaled input data.

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
