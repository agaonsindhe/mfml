import numpy as np
import sympy
from sympy import symbols, Matrix,MutableDenseMatrix

from numpy import poly


# Generate a random integer matrix C of size 4x3
C = np.random.randint(low=-10, high=10, size=(4, 3))

# Calculate A1 = C.T * C (Matrix multiplication of C transpose and C)
A1 = np.dot(C.T, C)

sym_array = MutableDenseMatrix(A1)

# Compute the characteristic polynomial coefficients from the eigenvalues
# char_poly_coefficients = poly(eigenvalues)

# Define the symbolic variable
lambda_ = symbols('\u03BB')
char_poly = sym_array.charpoly(lambda_).as_expr()

# The characteristic equation is given by setting the polynomial to zero
char_eq = char_poly.as_expr()  # Convert to expression

# Compute the eigenvalues and eigenvectors of A1
eigenvalues, eigenvectors = np.linalg.eig(A1)

# Print the matrices, characteristic polynomial coefficients, eigenvalues, and eigenvectors
print("Matrix C:\n", C)
print("Matrix A1:\n", A1)
print(f"Characteristic Polynomial: {char_poly}")
print(f"Characteristic Equation: {char_eq} = 0")
print("Eigenvalues:\n", eigenvalues)
print("Eigenvectors:\n", eigenvectors)


# Vishal
# Matrix C:
#  [[ 3 -7 -1]
#  [ 1  2 -7]
#  [-2 -7 -7]
#  [ 1  5 -2]]
# Matrix A1:
#  [[ 15   0   2]
#  [  0 127  32]
#  [  2  32 103]]
# Characteristic Polynomial: λ**3 - 245*λ**2 + 15503*λ - 180347
# Characteristic Equation: λ**3 - 245*λ**2 + 15503*λ - 180347 = 0
# Eigenvalues:
#  [ 14.9493106   80.86500024 149.18568915]
# Eigenvectors:
#  [[-0.99965281  0.02494293 -0.00849185]
#  [-0.00723555 -0.56975946 -0.82177965]
#  [ 0.0253359   0.8214329  -0.56974212]]


#Anshuman
# Matrix C:
#  [[ 7 -5 -1]
#  [-3 -6 -7]
#  [-2 -5  5]
#  [ 1 -8  6]]
# Matrix A1:
#  [[ 63 -15  10]
#  [-15 150 -26]
#  [ 10 -26 111]]
# Characteristic Polynomial: λ**3 - 324*λ**2 + 32092*λ - 974187
# Characteristic Equation: λ**3 - 324*λ**2 + 32092*λ - 974187 = 0
# Eigenvalues:
#  [166.10581902  59.76936     98.12482098]
# Eigenvectors:
#  [[ 0.17105876 -0.98368352  0.05572829]
#  [-0.8787181  -0.12673357  0.46020985]
#  [ 0.4456382   0.12769238  0.88605939]]

#Amit
# Matrix C:
#  [[-1 -8 -2]
#  [-7  0  0]
#  [ 0 -3  7]
#  [-4  4  9]]
# Matrix A1:
#  [[ 66  -8 -34]
#  [ -8  89  31]
#  [-34  31 134]]
# Characteristic Polynomial: λ**3 - 289*λ**2 + 24463*λ - 629094
# Characteristic Equation: λ**3 - 289*λ**2 + 24463*λ - 629094 = 0
# Eigenvalues:
#  [161.90314393  51.18787437  75.9089817 ]
# Eigenvectors:
#  [[-0.33574344  0.8904764  -0.30712884]
#  [ 0.39956132 -0.16063391 -0.90252285]
#  [ 0.85301061  0.42573293  0.30186814]]

