import numpy as np

def power_method(A, num_iterations=100, tolerance=1e-10):
    n, _ = A.shape
    b_k = np.random.rand(n)
    iterates = []
    
    for _ in range(num_iterations):
        b_k1 = np.dot(A, b_k)
        b_k1_norm = np.linalg.norm(b_k1)
        b_k = b_k1 / b_k1_norm
        lambda_k = b_k1_norm
        iterates.append(lambda_k)
        
        if np.linalg.norm(np.dot(A, b_k) - lambda_k * b_k) < tolerance:
            break
    
    return lambda_k, b_k, iterates

# Given matrix A1 (use the actual A1 matrix from your calculations)
A1 = np.array([[149, 64, 76],
               [64, 90, 1],
               [76, 1, 281]])

# Assuming x_hat_1 and x_hat_2 are the normalized eigenvectors corresponding to the largest and second largest eigenvalues of A1
# Replace these with your actual x_hat_1 and x_hat_2 calculated previously
x_hat_1 = np.array([-0.444, -0.128, -0.887])  # Example placeholder, use actual x_hat_1
x_hat_2 = np.array([-0.669, -0.611, 0.423])   # Example placeholder, use actual x_hat_2

# Construct A3 = A1 - x_hat_1 x_hat_1^T A1 - x_hat_2 x_hat_2^T A1
x_hat_1_matrix = np.outer(x_hat_1, x_hat_1)
x_hat_2_matrix = np.outer(x_hat_2, x_hat_2)
A3 = A1 - np.dot(x_hat_1_matrix, A1) - np.dot(x_hat_2_matrix, A1)

# Use the Power method to find the largest eigenvalue and corresponding eigenvector of A3
lambda_3, x_3, iterates_3 = power_method(A3, 100)
x_hat_3 = x_3 / np.linalg.norm(x_3, 2)  # Normalize x_3

# Output the first 10 iterates of the eigenvalue λ3
print("First 10 iterates of λ3:", iterates_3[:10])

# Output the final λ3 and normalized eigenvector x_hat_3
print("Final λ3:", lambda_3)
print("Normalized eigenvector x_hat_3:", x_hat_3)

# Comparison with values obtained in i)
# This involves comparing λ3 and x_hat_3 with the third largest eigenvalue and its eigenvector obtained directly (if available).
