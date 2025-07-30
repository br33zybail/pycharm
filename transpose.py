import numpy as np

def transpose_matrix(A):
    """Compute the transpose of Matrix A"""
    rows_A, cols_A = len(A), len(A[0])
    # Initialize result with swapped dimensions
    result = [[0 for _ in range(rows_A)] for _ in range(cols_A)]
    # Assign A[j][i] to result [i][j]
    for i in range(rows_A):
        for j in range(cols_A):
            result[j][i] = A[i][j]
    return result

def matrix_multiply(A,B):
    """Matrix multiplication from prior implementation"""
    if len(A[0]) != len(B):
        raise ValueError("Matrices cannot be multiplied: Number of columns in A must equal number of rows in B.")
    rows_A, cols_A = len(A), len(A[0])
    cols_B = len(B[0])
    result = [[0 for _ in range(rows_A)] for _ in range(cols_B)]
    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]
    return result

# Test Matrix
A = [[1, 2, 3], [4, 5, 6]]

# Compute A^T
A_T = transpose_matrix(A)

# Compute A^T * A
ATA = matrix_multiply(A_T, A)

# Convert to NumPy for verification
A_np = np.array(A)
A_T_np = A_np.T
ATA_np = np.dot(A_T_np, A_np)

# Print results
print("Custom Transpose (A^T):")
print(np.array(A_T))
print("\nNumPy Transpose (A^T):")
print(A_T_np)
print("\nCustom A^T × A:")
print(np.array(ATA))
print("\nNumPy A^T × A:")
print(ATA_np)
print("\nAre transposes equal? ", np.array_equal(np.array(A_T), A_T_np))
print("Are A^T × A results equal? ", np.array_equal(np.array(ATA), ATA_np))