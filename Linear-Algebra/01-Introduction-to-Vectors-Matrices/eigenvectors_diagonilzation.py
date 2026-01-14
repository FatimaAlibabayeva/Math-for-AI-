
# Math4AI: Linear Algebra - Programming Assignment 7
# Starter Code Template

import numpy as np

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix or vector with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_eigenpairs(name, eigenvalues, eigenvectors):
    """Prints a set of eigenvalues and their corresponding eigenvectors."""
    print(f"{name}:")
    if eigenvalues is None or eigenvectors is None:
        print("None (or not implemented)")
        print("-" * 40)
        return

    for i, val in enumerate(eigenvalues):
        # eigenvectors are columns in the matrix
        vec = eigenvectors[:, i] if isinstance(eigenvectors, np.ndarray) else eigenvectors[i]
        print(f"  Eigenvalue λ_{i+1} = {val:.4f}")
        print(f"  Eigenvector v_{i+1} =\n{vec.reshape(-1, 1)}")
    print("-" * 40)

# ====================================================================
# Problem Setup for All Parts
# ====================================================================

# Matrix for Part 1 and 3
A = np.array([
    [4., -2.],
    [1., 1.]
])

# Matrix for Part 2 (Diagonalizable)
B = np.array([
    [1., 0., 1.],
    [0., 1., 0.],
    [1., 0., 1.]
])

# Matrix for Part 2 (Not Diagonalizable / Defective)
C = np.array([
    [1., 1.],
    [0., 1.]
])

# ====================================================================
# Prerequisite: Nullspace Function (Students must implement or reuse)
# ====================================================================
#### burda mene nullspace basis tapmaqcin rref lazim olacaq deye onunda functionini yazram
#mende your code burdan basliyir
def to_rref(M):
    """
    Converts a matrix M to its Reduced Row Echelon Form (RREF).
    This is the core function needed to find the bases.
    """
    # --- YOUR CODE HERE ---
    M = M.copy()
    rows, cols = M.shape
    pivot_row=0 #pivot setirler 0dan baslayir deye 0 yaziram
    epsilon = 1e-10 #yoxlamaqcindi

    for pivot_col in range(cols): #forward elimination burdan basdiyir
      if pivot_row>=rows:
        break
      max_val = -1 #pivotu tapim deye elementi en boyuk olan setri tapram
      max_row = -1
      for r in range(pivot_row, rows):
        current_abs = abs(M[r, pivot_col])
        if current_abs > max_val:
          max_val = current_abs
          max_row = r
      if max_val < epsilon:
        continue # birden en boyuk eleement bele 0a yaxin birsey cixsa demeli bu sutunu atlamaq lazimdi
      if max_row!=pivot_row:
        M[[pivot_row, max_row]] = M[[max_row, pivot_row]] #burda uje row swap edirem
      pivot_value=M[pivot_row, pivot_col]
      M[pivot_row,:] /= pivot_value #  butun setirleri ele pivota bolurem

      for i in range(pivot_row+1,rows): #pivotun asaqsindaki elementleri 0a cevirmek
        factor = M[i, pivot_col]
        M[i, :] -= factor * M[pivot_row, :] # butun setirlerde pivota yeni 1e vurulmus hemin ededi ededin ozunden cixiramki 0 alim
      pivot_row+=1 #soramki setre kecmeycin

    #backward elimination
    for i in range(rows - 1, -1, -1):
      pivot_col=-1 #axirdan evvele teref gedir deye

      for j in range(cols):
        if abs(M[i, j]-1.0) < epsilon: #pivot 1e yaxin olmalidir
          pivot_col = j
          break
      if pivot_col==-1:
        continue

      for k in range(i): #pivot setrinden yuxardaki setirler,pivotdan yuxardaki elementleri 0a cevirmekcin
        factor = M[k, pivot_col]
        if abs(factor) > epsilon:
          M[k, :]-= factor * M[i, :]  #yenede forwardda etdiyimiz kimi factora vurulmusunu ozunden cixiriqki 0 alinsin
    # Implement the Gauss-Jordan elimination algorithm.
    # You can reuse this from a previous assignment.
    return M.copy()
def find_nullspace_basis(M):
    """
    Finds the basis for the nullspace of matrix M.
    This is required for find_eigenvectors.
    """
    # --- YOUR CODE HERE ---
    rref_M = to_rref(M) #kodu evvelde yazdiqima gore hecne qeyd etmedim
    m, n = rref_M.shape
    pivot_cols = []
    free_cols = []
    current_row = 0
    for j in range(n):
        if current_row < m and np.isclose(rref_M[current_row, j], 1.0):
            pivot_cols.append(j)
            current_row += 1
        else:
            free_cols.append(j)

    num_free = len(free_cols)
    if num_free == 0:
      return []
    special_solutions = []

    for k in range(num_free):
      x = np.zeros(n)
      free_col_index = free_cols[k]

      x[free_col_index] = 1.0
      for i in range(len(pivot_cols)):
        p_col = pivot_cols[i]
        x[p_col] = -rref_M[i, free_col_index]

      special_solutions.append(x)

    return special_solutions
    # You should have this function from a previous assignment.
    # It takes a matrix M, computes its RREF, and returns a list
    # of basis vectors for the nullspace.

    # Placeholder:
    print("WARNING: `find_nullspace_basis` is not implemented. `find_eigenvectors` will not work.")
    return special_solutions


# ====================================================================
# PART 1: FINDING EIGENVALUES AND EIGENVECTORS
# ====================================================================
print("="*60)
print("PART 1: FINDING EIGENVALUES AND EIGENVECTORS")
print("="*60)
print_matrix("Matrix A for Part 1", A)

# --- 1.1: Eigenvalues from the Characteristic Equation ---
def find_eigenvalues_2x2(A):
    """
    Computes the eigenvalues of a 2x2 matrix using the characteristic equation.
    λ^2 - tr(A)λ + det(A) = 0
    """
    eigenvalues = []
    # --- YOUR CODE HERE ---
    tr_A = A[0,0] + A[1,1]
    det_A = A[0,0]*A[1,1] - A[0,1]*A[1,0] #yalnizca 2*2 e matrixcin olduquna gore bele yazmaq okaydi
    discriminant = tr_A**2 - 4*det_A
    from numpy import sqrt
    eigenvalue1 = (tr_A + sqrt(discriminant)) / 2
    eigenvalue2 = (tr_A - sqrt(discriminant)) / 2
    # 1. Calculate the trace of A: tr(A) = A[0,0] + A[1,1]
    # 2. Calculate the determinant of A: det(A) = A[0,0]*A[1,1] - A[0,1]*A[1,0]
    # 3. Use the quadratic formula to solve for the two eigenvalues λ.
    #    λ = [-b ± sqrt(b^2 - 4ac)] / 2a
    #    Here, a=1, b=-tr(A), c=det(A)
    eigenvalues.append(eigenvalue1)
    eigenvalues.append(eigenvalue2)
    # Placeholder:
    return eigenvalues

# --- 1.2: Eigenvectors from the Nullspace ---
def find_eigenvectors(A, eigenvalues):
    """
    Finds the eigenvectors for a matrix given its eigenvalues.
    """
    eigenvectors = []
    # --- YOUR CODE HERE ---
    identity_matrix=np.identity(len(A)) #mene identity matrix lazim olacaq deye yaradiram
    B1=A-eigenvalues[0]*identity_matrix
    B2=A-eigenvalues[1]*identity_matrix
    # Note: find_nullspace_basis returns a list of basis vectors.
    # For a 2x2 matrix and distinct eigenvalues, each nullspace should ideally have one basis vector.
    nullspace_basis_B1 = find_nullspace_basis(B1)
    nullspace_basis_B2 = find_nullspace_basis(B2)

    # Assuming each nullspace basis is a list containing one eigenvector for this case
    if nullspace_basis_B1:
        eigenvectors.append(nullspace_basis_B1[0])
    if nullspace_basis_B2:
        eigenvectors.append(nullspace_basis_B2[0])

    # For each eigenvalue λ in the `eigenvalues` list:
    # 1. Construct the matrix B = A - λ*I, where I is the identity matrix.
    # 2. Find the nullspace of B using your `find_nullspace_basis` function.
    # 3. The basis vector(s) of the nullspace are the eigenvector(s) for λ.
    # 4. Append the found eigenvector(s) to the `eigenvectors` list.

    # Placeholder:
    return eigenvectors


print("--- 1.1 & 1.2: Finding Eigenpairs from Scratch ---")
eigvals_scratch = find_eigenvalues_2x2(A)
eigvecs_scratch = find_eigenvectors(A, eigvals_scratch)
# For consistent printing, let's combine the eigenvectors into a matrix
eigvecs_matrix_scratch = np.column_stack(eigvecs_scratch) if eigvecs_scratch else None
print_eigenpairs("Eigenpairs (from scratch)", eigvals_scratch, eigvecs_matrix_scratch)


# --- 1.3: Verification ---
print("--- 1.3: Verification ---")
print("Verifying A*v = λ*v:")
if eigvals_scratch and eigvecs_scratch:
    for i in range(len(eigvals_scratch)):
        l, v = eigvals_scratch[i], eigvecs_scratch[i]
        Av = A @ v
        lv = l * v
        print(f"For λ = {l:.4f}:")
        print_matrix("  A @ v", Av.reshape(-1, 1))
        print_matrix("  λ * v", lv.reshape(-1, 1))
        print(f"  Are they close? {np.allclose(Av, lv)}\n")
else:
    print("Cannot verify, scratch implementation is missing.\n")

print("--- Comparing with NumPy ---")
eigvals_np, eigvecs_np = np.linalg.eig(A)
print_eigenpairs("Eigenpairs (from NumPy)", eigvals_np, eigvecs_np)


# ====================================================================
# PART 2: DIAGONALIZATION
# ====================================================================
print("\n" + "="*60)
print("PART 2: DIAGONALIZATION")
print("="*60)

def diagonalize(A):
    """
    Performs diagonalization of matrix A, if possible.
    Returns S, Lambda, S_inv if diagonalizable, otherwise returns (None, None, None).
    """
    # --- YOUR CODE HERE ---
    eigenvalue_A, eigen_vector = np.linalg.eig(A)
    # Corrected check for diagonalizability and function return values
    if np.linalg.det(eigen_vector) != 0:#bu 2 serte emele elese demeliki diagnolizabledi ve diger itarationlari implement eliye bilerem
       S=eigen_vector # S is the matrix of eigenvectors (columns)
       # Create a diagonal matrix with eigenvalues
       Lambda = np.diag(eigenvalue_A)
       S_inv=np.linalg.inv(S)
       return S, Lambda, S_inv
    else:
        return None, None, None # sertde eslinde none qaytarmaqi deyib sadece olaraq men bele yazdim



    # 1. Find the eigenvalues and eigenvectors of A. For this general function,
    #    it's okay to use np.linalg.eig() as a tool.
    # 2. Check for diagonalizability: An n x n matrix is diagonalizable if it has
    #    n linearly independent eigenvectors. A simple check is if the eigenvector
    #    matrix is invertible (i.e., its determinant is non-zero).
    # 3. If diagonalizable:
    #    a. Construct S (matrix of eigenvectors).
    #    b. Construct Lambda (diagonal matrix of eigenvalues).
    #    c. Compute S_inv using np.linalg.inv(S).
    #    d. Return S, Lambda, S_inv
    # 4. If not diagonalizable, return None, None, None.

    # Placeholder:

# --- 2.2: Verification and A Defective Case ---
print("--- 2.2: Verification ---")
print_matrix("Diagonalizable Matrix B", B)
S, L, S_inv = diagonalize(B)
if S is not None:
    print_matrix("Eigenvector Matrix S", S)
    print_matrix("Eigenvalue Matrix Lambda", L)
    print_matrix("Inverse Eigenvector Matrix S^-1", S_inv)

    # Verify by checking if S @ L @ S_inv == B
    B_reconstructed = S @ L @ S_inv
    print_matrix("Reconstructed B = S @ Lambda @ S^-1", B_reconstructed)
    print(f"Is reconstructed B close to original B? {np.allclose(B, B_reconstructed)}")
else:
    print("Diagonalization of B failed or is not implemented.")

print("\n--- Testing a Defective (Non-Diagonalizable) Case ---")
print_matrix("Non-Diagonalizable Matrix C", C)
S_C, L_C, S_inv_C = diagonalize(C)
if S_C is None:
    print("Function correctly identified that C is not diagonalizable.")
else:
    print("Function incorrectly diagonalized C.")

# ====================================================================
# PART 3: THE POWER METHOD
# ====================================================================
print("\n" + "="*60)
print("PART 3: THE POWER METHOD")
print("="*60)

# --- 3.1: Implement the Power Method ---
def power_iteration(A, num_iterations: int):
    """
    Estimates the dominant eigenvector of a matrix A.
    """
    # --- YOUR CODE HERE ---
    # Added check for zero vector initialization
    b_k = np.random.rand(A.shape[0])
    if np.linalg.norm(b_k) < 1e-10: # Avoid starting with a zero vector
        b_k = np.ones(A.shape[0])


    for i in range(num_iterations): # o vaxta kimi eliyirikki aldqiqimiz vector ele eigenvectorun ozune beraber olur(max olanina)
       b_k1=A@b_k
       if np.linalg.norm(b_k1) < 1e-10: # Avoid division by zero
           return np.zeros(A.shape[0])
       b_k=b_k1/np.linalg.norm(b_k1) #nromallasdirmaq ucun vectorun elementlerini uzunluquna bolurem
    return b_k
    # 1. Create a random initial vector `b_k` with the same number of rows as A.
    # 2. Loop `num_iterations` times:
    #    a. Calculate the matrix-vector product `A @ b_k`.
    #    b. Normalize the result to get the next `b_k`.
    # 3. Return the final `b_k`.

    # Placeholder:
    return b_k

# --- 3.2: Estimating the Eigenvalue with the Rayleigh Quotient ---
print("--- 3.1 & 3.2: Finding Dominant Eigenpair ---")
dominant_eigenvector = power_iteration(A, 100)

if dominant_eigenvector is not None:
    # Rayleigh Quotient: λ = (v.T @ A @ v) / (v.T @ v)
    v = dominant_eigenvector
    # Ensure v is a column vector for matrix multiplication
    v = v.reshape(-1, 1)
    numerator = v.T @ A @ v
    denominator = v.T @ v
    # Extract scalar values from the 1x1 matrices
    dominant_eigenvalue = numerator[0, 0] / denominator[0, 0]
    print(f"Estimated Dominant Eigenvalue (from Rayleigh Quotient): {dominant_eigenvalue:.4f}")
    print_matrix("Estimated Dominant Eigenvector (from Power Iteration)", v)
else:
    print("Power iteration not implemented.")

# --- 3.3: Verification and Convergence ---
print("\n--- 3.3: Verification and Convergence ---")
if eigvals_scratch is not None:
    # Get the exact dominant eigenvalue (largest in absolute value) from Part 1
    # Use NumPy's eigvals to ensure robustness
    eigvals_np_verify = np.linalg.eigvals(A)
    exact_dom_val = max(eigvals_np_verify, key=abs)
    print(f"Exact Dominant Eigenvalue: {exact_dom_val:.4f}\n")

    for iterations in [5, 10, 20, 50]:
        v_est = power_iteration(A, iterations)
        if v_est is not None:
            # Ensure v_est is a column vector for Rayleigh Quotient
            v_est = v_est.reshape(-1, 1)
            numerator = v_est.T @ A @ v_est
            denominator = v_est.T @ v_est
            # Extract scalar values
            l_est = numerator[0, 0] / denominator[0, 0]
            print(f"After {iterations} iterations:")
            print(f"  Estimated λ = {l_est:.4f}")
        else:
            print("Power iteration not implemented.")
            break
else:
    print("Cannot verify convergence, Part 1 results missing.")
