import numpy as np

def matrix_multiply(A,B):
    # Check if matrices can be multiplied (cols of A == rows of B)
    if len(A[0]) != len(B):
        raise ValueError("Matrices cannot be multiplied: Number of columns in A must equal number of rows in B.")

    # Dimensions
    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    # Initialize result matrix C with zeros (rows_A x cols_B)
    C = [[0 for _ in range(cols_B)] for _ in range(rows_A)]

    # Nested loops for multiplication
    for i in range(rows_A): # Loop over rows of A
        for j in range(cols_B): # Loop over columns of B
            for k in range(cols_A): # Loop over columns of A / rows of B
                C[i][j] += A[i][k] * B[k][j] # Accumulate the sum

    return C

# Same matrices for testing (A is 2x3, B is 3x2, result should be 2x2)
A = [[1, 2, 3], [4, 5, 6]]
B = [[7, 8], [9, 10], [11, 12]]

# Compute with custom function
C_custom = matrix_multiply(A,B)

# Verify with NumPy
A_np = np.array(A)
B_np = np.array(B)
C_np = np.dot(A_np, B_np)

# Convert custom result to NumPy array for easy comparison

C_custom_np = np.array(C_custom)

# Output results
print("Custom Matrix Multiplication Result:")
print(C_custom)
print(C_custom_np)
print("\nNumpy np.dot() Result:")
print(C_np)
print("\nAre they equal? ", np.array_equal(C_custom_np, C_np))