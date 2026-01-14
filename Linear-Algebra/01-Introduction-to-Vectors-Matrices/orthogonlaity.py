# Math4AI: Linear Algebra - Programming Assignment 5
# Starter Code Template

import numpy as np
from scipy.linalg import null_space, orth

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_vectors(name, vecs):
    """Prints a list of basis vectors from a list of arrays or a 2D array."""
    print(f"{name}:")
    if vecs is None or len(vecs) == 0:
        print("[] (or not implemented)")
    elif isinstance(vecs, list) and all(isinstance(v, np.ndarray) for v in vecs):
        for i, v in enumerate(vecs):
            print(f"  Basis Vector {i+1}:\n{v.reshape(-1, 1)}")
    elif isinstance(vecs, np.ndarray) and vecs.ndim == 2:
         for i in range(vecs.shape[1]):
             print(f"  Basis Vector {i+1}:\n{vecs[:, i].reshape(-1, 1)}")
    else:
        print("Unsupported format for printing vectors.")
    print("-" * 40)



A1 = np.array([
    [1., 2., 3., 5.],
    [2., 4., 8., 12.],
    [3., 6., 7., 13.]
])
print_matrix("Matrix A for Part 1", A1)


# --- Reusable Helper Function (Students must implement) ---
def to_rref(M):
    """
    Converts a matrix M to its Reduced Row Echelon Form (RREF).
    This is the core function needed to find the bases.
    """
    # --- YOUR CODE HERE ---
    # Implement the Gauss-Jordan elimination algorithm.
    # You can reuse this from a previous assignment.
    print("RREF function is not implemented.")
    return M.copy()


# --- 5.1: Bases for the Four Subspaces ---

def find_column_space_basis(A):
    """Basis for C(A) from pivot columns of original A."""
    # --- YOUR CODE HERE ---
    # 1. Get RREF of A.
    # 2. Identify pivot column indices.
    # 3. Return the corresponding columns from the *original* matrix A.
    return None

def find_null_space_basis(A):
    """Basis for N(A) from special solutions."""
    # --- YOUR CODE HERE ---
    # 1. Get RREF of A.
    # 2. Identify pivot and free column indices.
    # 3. For each free variable, create a special solution.
    # 4. Return the list of special solution vectors.
    return None

def find_row_space_basis(A):
    """Basis for C(A^T) from non-zero rows of RREF of A."""
    # --- YOUR CODE HERE ---
    # 1. Get RREF of A.
    # 2. The non-zero rows of the RREF form the basis for the row space.
    # 3. Return these rows as a list of vectors.
    return None

def find_left_null_space_basis(A):
    """Basis for N(A^T) is the nullspace of A^T."""
    # Hint: You can just use your existing nullspace function!
    print("Finding Left Nullspace by finding Nullspace of A.T")
    return find_null_space_basis(A.T)


print("\n--- 1.1: Finding the Bases ---")
C_A_basis = find_column_space_basis(A1.copy())
N_A_basis = find_null_space_basis(A1.copy())
C_AT_basis = find_row_space_basis(A1.copy())
N_AT_basis = find_left_null_space_basis(A1.copy())

print_vectors("Column Space Basis C(A)", C_A_basis)
print_vectors("Nullspace Basis N(A)", N_A_basis)
print_vectors("Row Space Basis C(A^T)", C_AT_basis)
print_vectors("Left Nullspace Basis N(A^T)", N_AT_basis)


# --- 5.2: Verification & The Fundamental Theorem ---
print("\n--- 1.2: Verification & The Fundamental Theorem ---")

# 1. Verify Dimensions
print("--- Verifying Dimensions ---")
if all(b is not None for b in [C_A_basis, N_A_basis, C_AT_basis, N_AT_basis]):
    rank_A = C_A_basis.shape[1]
    dim_N_A = len(N_A_basis)
    dim_C_AT = len(C_AT_basis)
    dim_N_AT = len(N_AT_basis)
    m, n = A1.shape

    print(f"Dimensions of A: {m}x{n}")
    print(f"dim(C(A)) = {rank_A} (Rank)")
    print(f"dim(C(A^T)) = {dim_C_AT} (Rank)")
    print(f"Are ranks equal? {rank_A == dim_C_AT}\n")

    print("Rank-Nullity Theorem for A:")
    print(f"dim(C(A)) + dim(N(A)) = {rank_A} + {dim_N_A} = {rank_A + dim_N_A}")
    print(f"Number of columns (n) = {n}")
    print(f"Is theorem satisfied? {rank_A + dim_N_A == n}\n")
    
    print("Rank-Nullity Theorem for A^T:")
    print(f"dim(C(A^T)) + dim(N(A^T)) = {dim_C_AT} + {dim_N_AT} = {dim_C_AT + dim_N_AT}")
    print(f"Number of rows (m) = {m}")
    print(f"Is theorem satisfied? {dim_C_AT + dim_N_AT == m}\n")
else:
    print("Bases not implemented, cannot verify dimensions.")

# 2. Verify Orthogonality
print("--- Verifying Orthogonality ---")
if all(b is not None for b in [C_A_basis, N_A_basis, C_AT_basis, N_AT_basis]):
    # Row Space _|_ Nullspace
    row_vec_1 = C_AT_basis[0]
    null_vec_1 = N_A_basis[0]
    dot_product_1 = np.dot(row_vec_1, null_vec_1)
    print(f"Dot product of a row space vector and a nullspace vector:")
    print(f"Result = {dot_product_1:.4f} (should be 0)")
    
    # Column Space _|_ Left Nullspace
    col_vec_1 = C_A_basis[:, 0]
    left_null_vec_1 = N_AT_basis[0]
    dot_product_2 = np.dot(col_vec_1, left_null_vec_1)
    print(f"\nDot product of a column space vector and a left nullspace vector:")
    print(f"Result = {dot_product_2:.4f} (should be 0)")
else:
    print("Bases not implemented, cannot verify orthogonality.")
