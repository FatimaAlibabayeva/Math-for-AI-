# Math4AI: Linear Algebra - Programming Assignment 8
# Starter Code Template

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image # Used for creating a sample image

# --- Helper Functions for Pretty Printing ---
def print_matrix(name, m):
    """Prints a matrix or vector with its name."""
    if m is None:
        print(f"{name}:\nNone (or not implemented)")
    else:
        np.set_printoptions(precision=4, suppress=True)
        print(f"{name}:\n{m}")
    print("-" * 40)

def print_vector(name, v):
    """Prints a vector with its name."""
    print_matrix(name, v.reshape(-1, 1) if v is not None else None)

# ====================================================================
# PART 1: DEFINING AND VERIFYING LINEAR TRANSFORMATIONS
# ====================================================================
print("="*60)
print("PART 1: DEFINING AND VERIFYING LINEAR TRANSFORMATIONS")
print("="*60)

# The transformation T from R^2 to R^3
def transform(v):
    """
    Applies the transformation T((x1, x2)) = (x1 + x2, x1 - 2*x2, 3*x1).
    Args:
        v (np.ndarray): A 2D input vector [x1, x2].
    Returns:
        np.ndarray: A 3D output vector.
    """
    # --- YOUR CODE HERE ---
    # Implement the transformation based on its definition.
    # Make sure your input `v` has two components, x1 and x2.
    x0=v[0]
    x1=v[1] #bunlari girisde olan vektoru complimentlere ayirmaqcin yaziriq
    result=[ #ele verildiyi kimi transformationi apply eliyrem
        x0+x1,
        x0-2*x1,
        3*x0
    ]

    return np.array(result) # Replace with your result

def verify_linearity(T_func):
    """
    Verifies the additivity and homogeneity properties of a transformation.
    """
    print("--- Verifying Linearithy Properties ---")

    # --- YOUR CODE HERE ---
    u = np.random.rand(2)
    v = np.random.rand(2)
    c = np.random.rand()
    # 1. Create two random 2D vectors, u and v, and a random scalar c.
    #    u = np.random.rand(2)
    #    v = np.random.rand(2)
    #    c = np.random.rand()
    lhs=T_func(u + v) #additivity yoxlanmasi
    rhs=T_func(u) + T_func(v)
    # 2. Check Additivity: T(u + v) == T(u) + T(v)
    #    a. Calculate LHS: T_func(u + v)
    #    b. Calculate RHS: T_func(u) + T_func(v)
    #    c. Check if they are close using np.allclose(LHS, RHS) and print the result.
    additive = np.allclose(lhs, rhs) #burda allclosenan close olub olmaliqlarini yoxlayiriq
    print(f"T(u+v) = {lhs}")
    print(f"T(u) + T(v) = {rhs}")
    print(f"Additivity holds: {additive}")
    # 3. Check Homogeneity: T(c*v) == c*T(v)
    #    a. Calculate LHS: T_func(c * v)
    #    b. Calculate RHS: c * T_func(v)
    #    c. Check if they are close and print the result.
    lhs_homogen=T_func(c*v) #homogenliyini yoxlayiriq
    rhs_homogen=c*T_func(v)
    homogen=np.allclose(lhs_homogen,rhs_homogen) #yaxin olub olmaliqlarini yoxlayiriq
    print(f"T(c*v) = {lhs_homogen}")
    print(f"c*T(v) = {rhs_homogen}")
    print(f"Homogeneity holds: {homogen}")


# --- Calling the functions for Part 1 ---
verify_linearity(transform)


# ====================================================================
# PART 2: THE MATRIX OF A LINEAR TRANSFORMATION
# ====================================================================
print("\n" + "="*60)
print("PART 2: THE MATRIX OF A LINEAR TRANSFORMATION")
print("="*60)

# --- 2.1: Finding the Standard Matrix ---
def find_standard_matrix(T_func, n, m):
    """
    Finds the standard matrix representation of a linear transformation.
    """
    standard_matrix = np.zeros((m, n))

    # --- YOUR CODE HERE ---
    identity_matrix=np.identity(n) #identity matrixi lazimdi deye hazir funksiyanan onu yaradiram
    basis=[]
    for i in range(n):
      e=identity_matrix[:,i] #identity matrixden(I) istiafede elyib standart basisi qururam cunki identitynin columnlari onsuzda basis vektorlardi
      T_ei=T_func(e) #transformationi apply edirem
      standard_matrix[:, i] = T_ei 

    # 1. Create the standard basis vectors for the input space R^n.
    #    An identity matrix of size n x n can be helpful here, as its
    #    columns are the basis vectors.
    # 2. For each basis vector `e_i`:
    #    a. Apply the transformation: `T(e_i)`.
    #    b. The result is the i-th column of your standard matrix.

    return standard_matrix # Replace with your result

# --- 2.2: Verification ---
print("--- 2.1 & 2.2: Finding and Verifying the Standard Matrix ---")
A_standard = find_standard_matrix(transform, n=2, m=3)
print_matrix("Standard Matrix A (from scratch)", A_standard)

if A_standard is not None:
    # Create a random test vector
    v_test = np.random.rand(2)
    print_vector("Random test vector v", v_test)

    # Apply the original function
    Tv = transform(v_test)
    print_vector("Result from T(v)", Tv)

    # Apply the matrix multiplication
    Av = A_standard @ v_test
    print_vector("Result from A @ v", Av)

    print(f"Are the results close? {np.allclose(Tv, Av)}")


# --- 2.3: Application: Common Geometric Transformations ---
print("\n--- 2.3: Composing Geometric Transformations ---")
p = np.array([2., 1.])
print_vector("Original point p", p)

# Step 1: Rotate the point by 90 degrees
theta = np.pi / 2
# --- YOUR CODE HERE ---
# Create the 2x2 rotation matrix R
R=np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]]) #2*2 e rotation matrixcin
# Calculate the rotated point p_prime
p_prime=R@p if p is not None else None #ele ozu dediyi kimi rotated pointi hesablayiram
print_vector("p' (rotated)", p_prime)

# Step 2: Translate the rotated point by t = [-3, 4]
t = np.array([-3., 4.])
p_double_prime = p_prime + t if p_prime is not None else None
print_vector("p'' (rotated then translated)", p_double_prime)

# Step 3: Compose and Verify using Homogeneous Coordinates
print("\n--- Verifying with a single composite matrix ---")
# --- YOUR CODE HERE ---
# Create the 3x3 augmented rotation matrix R_aug
# (It's the 2x2 R with a new row [0,0,1] and column [0,0,1]^T)
R_aug=np.array([[np.cos(theta),-np.sin(theta),0],[np.sin(theta),np.cos(theta),0],[0,0,1]])
# Create the 3x3 translation matrix T_mat
T_mat=np.array([[1,0,t[0]],[0,1,t[1]],[0,0,1]])
# Create the composite matrix M = T_mat @ R_aug
M=T_mat@R_aug
# Convert p to homogeneous coordinates p_h = [2, 1, 1]
p_h = np.append(p, 1)

print_matrix("Composite Matrix M = T @ R_aug", M)
# Apply the composite matrix
p_double_prime_h = M @ p_h if M is not None else None
print_vector("Transformed homogeneous vector M @ p_h", p_double_prime_h)

# Convert back to 2D
p_from_composite = p_double_prime_h[:2] if p_double_prime_h is not None else None
print_vector("Final point from composite matrix (in 2D)", p_from_composite)

if p_double_prime is not None and p_from_composite is not None:
    print(f"Is the result the same as step-by-step? {np.allclose(p_double_prime, p_from_composite)}")

# ====================================================================
# PART 3: SINGULAR VALUE DECOMPOSITION (SVD)
# ====================================================================
print("\n" + "="*60)
print("PART 3: SINGULAR VALUE DECOMPOSITION (SVD)")
print("="*60)

# --- 3.1: Computing SVD from Scratch ---
def compute_svd(A):
    """
    Computes the SVD of a matrix A.
    Returns U, Sigma (as a vector of singular values), and V^T.
    """
    # --- YOUR CODE HERE ---
    m,n=A.shape # setirnen sutunnari almaqcindi
    V_matrix = A.T @ A #burda Anin transposunu Aa vurram cunki mene onun eigenvaule ve vectorlari lazimdi
    eigen_val,eigen_vect = np.linalg.eig(V_matrix) #eigenvaluelarinan vectorlaeini tapir

    sorted_indices = np.argsort(eigen_val)[::-1] #burda bilmedim bubble sortnan eliyim yoxsa hazir funksiya olar 
    eigen_val = eigen_val[sorted_indices] #burdan eigen valuelari secirem 
    eigen_vect = eigen_vect[:, sorted_indices]#burdanda vectorlari

    V = eigen_vect # eigenvectorlar Vni form eliyir deye bele yaza bilerem
    VT=V.T #trnasposunu tapmaqcin
    sigma_vector = np.sqrt(np.maximum(eigen_val, 0)) # potential negative eigenvaluelarnan handle eliyirem

    r = np.sum(sigma_vector > 1e-9) #float deqiqliyine gore
    U_cols=[] #unun columnlari ucun
    for d in range (r):
      sigma_i=sigma_vector[d]
      v_i=V[:,d]
      u_i=(1/sigma_i)*A@v_i
      U_cols.append(u_i) #ele bildiymiz kimi Ununkileri hesablayib columna append eliyrik

    if U_cols: #unun row sayinin duzgun olub olmadiqini yoxlayiram
      U = np.column_stack(U_cols)
      if r < m:
          U_at, s_at, VT_at = np.linalg.svd(A.T)#evvel nullspacecin yazmisam deye daha burda scratchden yazmadim
          null_space_basis = U_at[:, np.isclose(s_at, 0)]
          U = np.hstack((U, null_space_basis))
    else:
        U = np.zeros((m, m)) #hec bir 0 singular valye yoxdusa


    return U, sigma_vector, VT # U, Sigma, VT

# --- 3.2: Application: Image Compression ---
print("\n--- 3.2: Application: Image Compression ---")
# Create a simple sample image to avoid file loading issues
def create_sample_image(size=(128, 128)):
    img = np.zeros(size, dtype=np.float32)
    # Create a cross shape
    img[size[0]//2 - 10 : size[0]//2 + 10, :] = 0.8
    img[:, size[1]//2 - 10 : size[1]//2 + 10] = 0.8
    # Create a circle
    cx, cy, r = size[0]//2, size[1]//2, size[0]//4
    x, y = np.ogrid[:size[0], :size[1]]
    mask = (x - cx)**2 + (y - cy)**2 < r**2
    img[mask] = 0.3
    return img

image_matrix = create_sample_image()
U, S_vals, VT = compute_svd(image_matrix)

def reconstruct_matrix(U, S_vals, VT, k):
    """
    Reconstructs a matrix from its SVD components using only the top k singular values.
    """
    # --- YOUR CODE HERE ---
    U_k = U[:, :k] #u matrixini slice elemekcin
    S_k = np.diag(S_vals[:k]) #ehazir funksiyanan yazdim scratchden yox
    VT_k = VT[:k, :] # VT ni slice eliyrem 
    A_k = U_k @ S_k @ VT_k #3 dene slice olunmusu bir birine vurdum
    # 1. Slice U to get the first k columns.
    # 2. Create a diagonal matrix from the first k singular values in S_vals.
    # 3. Slice VT to get the first k rows.
    # 4. Multiply the three sliced matrices together to get the approximation.

    return A_k # Replace with your result

if U is not None and S_vals is not None and VT is not None:
    ranks_to_show = [5, 15, 40]
    plt.figure(figsize=(len(ranks_to_show) * 4 + 4, 4))

    # Show Original
    plt.subplot(1, len(ranks_to_show) + 1, 1)
    plt.imshow(image_matrix, cmap='gray')
    plt.title('Original')
    plt.axis('off')

    for i, k in enumerate(ranks_to_show):
        reconstructed = reconstruct_matrix(U, S_vals, VT, k)
        if reconstructed is not None:
            plt.subplot(1, len(ranks_to_show) + 1, i + 2)
            plt.imshow(reconstructed.real, cmap='gray')
            plt.title(f'k = {k}')
            plt.axis('off')
    plt.show()
else:
    print("SVD components not computed, cannot perform image compression.")

# --- 3.3: Verification ---
print("\n--- 3.3: Verification ---")
test_matrix = np.array([[1., 2., 3.], [4., 5., 6.]])
print_matrix("Test Matrix for SVD", test_matrix)

U_scratch, S_vals_scratch, VT_scratch = compute_svd(test_matrix)
U_np, S_vals_np, VT_np = np.linalg.svd(test_matrix)

print_matrix("U (from scratch)", U_scratch)
print_matrix("U (from NumPy)", U_np)
print_matrix("Singular values (from scratch)", S_vals_scratch)
print_matrix("Singular values (from NumPy)", S_vals_np)
print_matrix("V^T (from scratch)", VT_scratch)
print_matrix("V^T (from NumPy)", VT_np)

if U_scratch is not None and S_vals_scratch is not None and VT_scratch is not None:
    # Reconstruct the matrix from the scratch SVD
    r_scratch = U_scratch.shape[1]
    # Place the singular values on the diagonal of a matrix with the same shape as the original matrix
    Sigma_diagonal = np.diag(S_vals_scratch[:r_scratch])
    VT_r = VT_scratch[:r_scratch, :]
    reconstructed_A = U_scratch @ Sigma_diagonal @ VT_scratch
    print_matrix("Reconstructed Matrix from Scratch SVD", reconstructed_A)
    print(f"Is it close to the original? {np.allclose(test_matrix, reconstructed_A)}")