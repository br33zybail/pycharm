import numpy as np

def determinant_2x2(A):
    return A[0][0] * A[1][1] - A[0][1] * A[1][0]

def inverse_2x2(A):
    det = determinant_2x2(A)
    if det == 0:
        raise ValueError("Matrix is singular, and determinant is 0.")
    return [[A[1][1] / det, -A[0][1] / det], [-A[1][0] / det, A[0][0] / det]]

A = [[2, 1], [3, 2]]
det_A = determinant_2x2(A)
A_inv = inverse_2x2(A)

print(f"Determinant: {det_A}")
print("Inverse:\n", np.array(A_inv))

A_np = np.array(A)
det_np = np.linalg.det(A_np)
A_inv_np = np.linalg.inv(A_np)
print(f"NumPy Determinant: {det_np:.0f}")
print("NumPy Inverse:\n", A_inv_np)
