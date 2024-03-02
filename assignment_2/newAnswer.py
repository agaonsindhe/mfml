import numpy as np

def power_method(A, x0, max_iter=10, tol=1e-6):
  """
  Implements the Power method to approximate the largest eigenvalue and eigenvector.

  Args:
      A: The square matrix.
      x0: The initial guess vector.
      max_iter: The maximum number of iterations (default: 10).
      tol: The tolerance for convergence (default: 1e-6).

  Returns:
      lambda_1: The approximate largest eigenvalue.
      x_1: The approximate largest eigenvector.
  """
  x = x0.copy()
  for _ in range(max_iter):
    y = A @ x
    lambda_1 = np.linalg.norm(y)
    x = y / lambda_1
    if np.linalg.norm(x - x_1) < tol:
      break
    x_1 = x.copy()
  return lambda_1, x

# Example usage
A1 = np.array( [[ 15,0, 2],
 [  0, 127,  32],
 [  2,  32, 103]])
x0 = np.array([1, 0,0])  # Initial guess vector

# Perform the Power method
lambda_1, x_1 = power_method(A1, x0)

# Calculate the normalized largest eigenvector
x_hat_1 = x_1 / np.linalg.norm(x_1)

# Print the first 10 iterates of the eigenvalue
print("First 10 iterates of the eigenvalue:")
for i in range(10):
  lambda_i, _ = power_method(A1, x0)
  print(f"Iterate {i+1}: {lambda_i}")

# Print the final results
print("\nFinal largest eigenvalue (lambda_1):", lambda_1)
print("Final largest eigenvector (x_1):", x_1)
print("Normalized largest eigenvector (x_hat_1):", x_hat_1)

# Comparison with ||x1||^2 (assuming you meant x_1):
norm_2_x_1 = np.linalg.norm(x_1) ** 2
print("\nComparison with ||x1||^2:")
print(f"||x1||^2: {norm_2_x_1}")

# Explanation:
print("\nExplanation:")
print("The Power method iteratively multiplies the matrix A with an initial guess vector.")
print("The resulting vector is normalized at each step. As the iterations progress,")
print("the vector converges towards the direction of the largest eigenvector.")
print("The Rayleigh quotient (lambda_1) is an approximation of the largest eigenvalue.")
print("The normalized eigenvector (x_hat_1) has a norm of 1, representing its unit length.")
print("The comparison with ||x1||^2 shows the magnitude of the eigenvector.")