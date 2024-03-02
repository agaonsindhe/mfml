from sympy import symbols, Matrix

# Define the symbolic variable
lambda_ = symbols('\u03BB')

# Define your matrix (as an example, a 2x2 matrix)
A = Matrix([[2, 3], [4, 5]])

print(type(A))

# Compute the characteristic polynomial of A
char_poly = A.charpoly(lambda_).as_expr()

# The characteristic equation is given by setting the polynomial to zero
char_eq = char_poly.as_expr()  # Convert to expression


print(f"Characteristic Polynomial: {char_poly}")
print(f"Characteristic Equation: {char_eq} = 0")
