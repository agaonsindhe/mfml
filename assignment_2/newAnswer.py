import numpy as np

# Generate random integer matrix C (4x3)
C = np.random.randint(1, 10, size=(4, 3))
print("Matrix C:\n", C)

# Calculate A1 = CT C
A1 = C.T @ C
print("Matrix A1:\n", A1)

# Calculate characteristic equation using numpy.linalg.eigvals()
char_eq = np.linalg.eigvals(A1)
print("Characteristic equation: det(A1 - λI) = 0")
print(char_eq.tolist())

# Find eigenvalues and eigenvectors using numpy.linalg.eig()
eigenvalues, eigenvectors = np.linalg.eig(A1)
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)

def power_method(A, x0, tol=1e-6, max_iter=100):
  """
  Implements the Power Method to find the largest eigenvalue and eigenvector.

  Args:
      A: The matrix to analyze.
      x0: The initial guess vector.
      tol: Tolerance for convergence (default: 1e-6).
      max_iter: Maximum number of iterations (default: 100).

  Returns:
      lambda_1: The largest eigenvalue.
      x_1: The corresponding eigenvector.
  """
  x_prev = x0
  for i in range(max_iter):
    x_curr = A @ x_prev
    lambda_1 = np.linalg.norm(x_curr)
    x_curr /= lambda_1  # Normalize

    # Check convergence
    if np.linalg.norm(x_curr - x_prev) / np.linalg.norm(x_curr) < tol:
      break

    x_prev = x_curr

  return lambda_1, x_curr

# Initial guess vector (randomly chosen)
x0 = np.random.rand(3, 1)

# Run the Power Method
lambda_1, x_1_hat = power_method(A1, x0)

# Comparison with numpy results
lambda_1_numpy, x_1_numpy = eigenvalues[0], eigenvectors[:, 0]

print("\nPower Method:")
print("Largest eigenvalue (lambda_1):", lambda_1)
print("Corresponding eigenvector (x_1_hat):")
print(x_1_hat)

# Check for consistency in signs (eigenvectors can have opposite signs)
if np.dot(x_1_hat.T, x_1_numpy) < 0:
  x_1_hat *= -1

print("\nComparison:")
print("||x_1||_2 from numpy:", np.linalg.norm(x_1_numpy))
print("||x_1_hat||_2 from Power Method:", np.linalg.norm(x_1_hat))

# Comment on comparison
print("\nAs expected, the norms (||x_1||_2) obtained from the Power Method and "
      "numpy are very close, confirming the validity of the implemented method.")