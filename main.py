import numpy as np
from utils import relu, relu_derivative, cubic_function

# Define constants of the equation
a, b, c, d = 1, -6, 11, -6  # (example: roots are 1, 2, 3)

# Hyperparameters
lr = 0.01
epochs = 10000

# Initialize x randomly
x = np.random.randn()
print(f"Initial guess: x = {x:.4f}")

for epoch in range(epochs):
    # Forward pass
    y_pred = relu(cubic_function(x, a, b, c, d))

    # Loss: we want y_pred = 0
    loss = y_pred**2

    # Backward pass
    dy = 2 * y_pred * relu_derivative(cubic_function(x, a, b, c, d))
    dx = dy * (3*a*x**2 + 2*b*x + c)  # derivative of the cubic

    # Gradient Descent update
    x -= lr * dx

    if epoch % 500 == 0:
        print(f"Epoch {epoch}: Loss = {loss:.6f}, x = {x:.4f}")

print(f"\nApproximate root: x = {x:.4f}")
