import numpy as np

def manual_broadcast_add(A,B):
    """
    Manually broadcast and add two arrays (matrix A and vector B)
    Assumes B is a vector to be broadcast across rows of A
    """
    # Get shapes
    rows_A, cols_A = len(A), len(A[0])
    rows_B, cols_B = 1, len(B) # Treating B as a 1xN vector

    # Check compatibility
    if cols_A != cols_B:
        raise ValueError(f"Incompatible shapes: A is {rows_A}x{cols_A}, B is {rows_B}x{cols_B}")

    # Initialize result matrix with same shape as A

    result = [[0 for _ in range(cols_A)] for _ in range(rows_A)]

    # Perform element-wise addition (broadcast B across A's rows)

    for i in range(rows_A):
        for j in range(cols_A):
            result[i][j] = A[i][j] + B[j]

    return result

# Test Case 1: From the problem
W = [[0.5, 0.3, 0.2], [0.1, 0.8, 0.4]]
b = [0.1, 0.05, -0.1]

# Compute manually

result_manual = manual_broadcast_add(W, b)

# Convert to NumPy arrays for verification

W_np = np.array(W)
b_np = np.array(b)
result_numpy = W_np + b_np # Numpy's automatic broadcasting

# Print results
print("Test Case 1: Manual Broadcasting Result:")
print(np.array(result_manual)) # Convert to NumPy for pretty printing
print("\nTest Case 1: NumPy Broadcasting Result:")
print(result_numpy)
print("\n Test Case 1: Are they equal? ", np.array_equal(np.array(result_manual),result_numpy))