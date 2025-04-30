import numpy as np

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def cubic_function(x, a, b, c, d):
    return a * x**3 + b * x**2 + c * x + d
