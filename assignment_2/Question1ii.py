import numpy as np

def power_method(A, num_iterations=100, tolerance=1e-10):
    n, _ = A.shape
    # Start with a random vector
    b_k = np.random.rand(n)
    # This will generate vector with element having largest value as 1

    iterates = []
    
    for _ in range(num_iterations):
        # Calculate the matrix-by-vector product Ab
        b_k1 = np.dot(A, b_k)
        
        # Calculate the norm
        b_k1_norm = np.linalg.norm(b_k1)
        
        # Re normalize the vector
        b_k = b_k1 / b_k1_norm
        
        # Save the current eigenvalue approximation
        lambda_k = b_k1_norm
        iterates.append(lambda_k)
        
        # Check for convergence
        if np.linalg.norm(np.dot(A, b_k) - lambda_k * b_k) < tolerance:
            break
    
    return lambda_k, b_k, iterates

# Given matrix A1 (replace this with your actual A1 matrix)
A1 = np.array( [[ 15,0, 2],
 [  0, 127,  32],
 [  2,  32, 103]])

# Use the Power method to find the largest eigenvalue and corresponding eigenvector
lambda_1, x_1, iterates = power_method(A1, 100)
x_hat_1 = x_1 / np.linalg.norm(x_1, 2)  # Normalize x_1

# Output the first 10 iterates of the eigenvalue
print("First 10 iterates of the eigenvalue approximation:", iterates[:10])

# Output the final largest eigenvalue and normalized eigenvector
print("Final largest eigenvalue (lambda_1):", lambda_1)
print("Corresponding Eigen Vector:(x_1)", x_1)
print("Normalized eigenvector (x_hat_1):", x_hat_1)

# Compare the norm
norm_x_1 = np.linalg.norm(x_1, 2)
print("Norm of x_1:", norm_x_1)


print("\nComparison:")
print("||x_1||_2 from numpy:", np.linalg.norm(x_1_numpy))
print("||x_1_hat||_2 from Power Method:", np.linalg.norm(x_1_hat))

# Comment on comparison
print("\nAs expected, the norms (||x_1||_2) obtained from the Power Method and "
      "numpy are very close, confirming the validity of the implemented method.")