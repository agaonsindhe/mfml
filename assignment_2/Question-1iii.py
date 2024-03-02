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

# Assuming x_hat_1 is the normalized eigenvector corresponding to the largest eigenvalue of A1
# Replace this with your actual x_hat_1 calculated in the previous step
x_hat_1 = np.array([-0.444, -0.128, -0.887])  # Example placeholder, use actual x_hat_1

# Construct A2 = A1 - x_hat_1 x_hat_1^T A1
x_hat_1_matrix = np.outer(x_hat_1, x_hat_1)
A2 = A1 - np.dot(x_hat_1_matrix, A1)

# Use the Power method to find the largest eigenvalue and corresponding eigenvector of A2
lambda_2, x_2, iterates_2 = power_method(A2, 100)
x_hat_2 = x_2 / np.linalg.norm(x_2, 2)  # Normalize x_2

# Output the first 10 iterates of the eigenvalue λ2
print("First 10 iterates of λ2:", iterates_2[:10])

# Output the final λ2 and normalized eigenvector x_hat_2
print("Final λ2:", lambda_2)
print("Normalized eigenvector x_hat_2:", x_hat_2)

# Comparison with values obtained in i)
# This would involve comparing λ2 and x_hat_2 with the second largest eigenvalue and its eigenvector obtained directly (if available).
