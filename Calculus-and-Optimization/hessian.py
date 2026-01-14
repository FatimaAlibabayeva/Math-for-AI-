# Math4AI: Calculus & Optimization - Assignment 4
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment4_code.py

import numpy as np
import matplotlib.pyplot as plt
import sympy
import scipy.integrate as spi

# ====================================================================
# --- Helper Function for Pretty Printing ---
# ====================================================================

def print_result(problem, description, value):
    """
    Helper function to print a result with its problem number and description.
    """
    print(f"--- {problem} ---")
    print(f"{description}:")
    if value is None:
        print("None (Function not yet implemented)")
    else:
        # Set print options for numerical results
        if isinstance(value, (int, float, np.number)):
             print(f"{value:.10f}")
        else:
             print(value)
    print("-" * 40)

# ====================================================================
# Part 1: Multivariable Calculus
# ====================================================================

# --------------------------------------------------------------------
# Problem 1.1: Partial Derivatives and the Gradient Vector
# --------------------------------------------------------------------

def partial_derivative(f, point, var_index, h=1e-7):
    """
    Computes the partial derivative of a multivariable function f
    at a given point w.r.t. the variable at var_index.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point (e.g., [a1, a2, ...]) to evaluate at.
        var_index (int): The index of the variable to differentiate (e.g., 0 for x_0).
        h (float, optional): Step size. Defaults to 1e-7.

    Returns:
        float: The approximated partial derivative, or None.
    """
    # --- YOUR CODE HERE ---

    # TODO 1: Create a copy of the 'point' array.
    point_copy=np.copy(point)

    # TODO 2: Modify the copied array at 'var_index' by adding 'h'.
    point_copy[var_index] += h

    # TODO 3: Evaluate the function at the modified point.
    f_copied=f(point_copy)

    # TODO 4: Evaluate the function at the original, unmodified 'point'.
    f_original=f(point)

    # TODO 5: Apply the finite difference formula.
    partial_deriv=(f_copied-f_original)/h

    return partial_deriv

def compute_gradient(f, point, h=1e-7):
    """
    Computes the full gradient vector of a multivariable function f
    at a given point.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point to evaluate at.
        h (float, optional): Step size to pass to partial_derivative.

    Returns:
        np.ndarray: The gradient vector, or None.
    """
    # Ensure 'point' is a numpy array for easier handling
    point = np.asarray(point)

    # --- YOUR CODE HERE ---

    # TODO 1: Get the number of variables (n).
    n=len(point)

    # TODO 2: Initialize a zero vector for the gradient.
    gradient=np.zeros(n)

    # TODO 3: Loop through each variable.
    for i in range(n):
        pd_i=partial_derivative(f,point,i,h=h)
        gradient[i]=pd_i

    # TODO 4: In the loop, call 'partial_derivative' for the i-th variable.

    # TODO 5: Store the result in the i-th element of your gradient vector.

    return gradient

# --------------------------------------------------------------------
# Problem 1.2: The Hessian Matrix
# --------------------------------------------------------------------

def compute_hessian(f, point, h=1e-5):
    """
    Computes the Hessian matrix of a multivariable function f
    at a given point.

    Args:
        f (callable): The multivariable function.
        point (np.ndarray): The point to evaluate at.
        h (float, optional): Step size.

    Returns:
        np.ndarray: The (n x n) Hessian matrix, or None.
    """
    point = np.asarray(point)

    # --- YOUR CODE HERE ---

    # TODO 1: Get the number of variables (n).
    n=len(point)

    # TODO 2: Initialize an (n x n) zero matrix for the Hessian.
    hessian_matrix=np.zeros((n,n))

    # TODO 3: Implement a nested loop (for i, for j).
    for i in range(n):
        for j in range(n):
            def gj_func(x_vector): #burda bunu yaziram cunki mene hem birinci hemde 2ci dereceli toreme lazimdi
                return partial_derivative(f,x_vector,i,h=h)

            h_ij=partial_derivative(gj_func,point,i,h=h) #soramda 2ci toremeni tapiriq
            hessian_matrix[i,j]=h_ij

    # TODO 4: Inside the loop, approximate the second partial
    #         derivative H_ij = d/dx_i ( d/dx_j f ).
    #         (Hint: This will likely involve your 'partial_derivative' function).

    return hessian_matrix

# --------------------------------------------------------------------
# Problem 1.3: Numerical Double Integration
# --------------------------------------------------------------------

def double_integral(f, a, b, c, d, nx, ny):
    """
    Computes the double integral of f(x, y) over [a, b] x [c, d]
    using the nested midpoint rule.

    Args:
        f (callable): The function to integrate (must accept f(x, y)).
        a (float): Lower bound for x.
        b (float): Upper bound for x.
        c (float): Lower bound for y.
        d (float): Upper bound for y.
        nx (int): Number of subintervals for x.
        ny (int): Number of subintervals for y.

    Returns:
        float: The approximated double integral, or None.
    """

    # --- YOUR CODE HERE ---

    # TODO 1: Calculate delta_x and delta_y.
    delta_x=(b-a)/nx
    delta_y=(d-c)/ny

    # TODO 2: Initialize the total sum.
    total_integral=0

    # TODO 3: Implement a nested loop over nx and ny.
    for i in range(nx):
        for j in range(ny):
            mid_x=a+(i+0.5)*delta_x
            mid_y=c+(j+0.5)*delta_y #-
            f_mid=f(mid_x,mid_y)
            total_integral+=f_mid

    # TODO 4: Inside the loops, calculate the midpoints x_i* and y_j*.

    # TODO 5: Evaluate the function at the midpoint f(x_i*, y_j*).

    # TODO 6: Add this value to the total sum.

    # TODO 7: After the loops, multiply the total sum by the area
    #         of each small rectangle.
    integral_approx=total_integral*delta_x*delta_y

    return integral_approx

# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

if __name__ == "__main__":

    print("=====================================================")
    print("Math4AI: Assignment 4 Verification")
    print("=====================================================")

    # --- Problem 1.1 Verification ---

    # Define the function f(x, y) = x^2 + 2y^2
    def f_1_1(point):
        x = point[0]
        y = point[1]
        return x**2 + 2*y**2

    test_point_1_1 = np.array([1.0, 1.0])

    # TODO:
    # 1. Call your 'compute_gradient' function.
    # 2. Print the result.
    grad_scratch = compute_gradient(f_1_1,test_point_1_1,h=1e-7) # Replace this
    print_result("Problem 1.1", f"Gradient of x^2+2y^2 at {test_point_1_1} (Scratch)", grad_scratch)

    # TODO: Plotting (for the report)
    x_range = np.linspace(-3, 3, 50)
    y_range = np.linspace(-3, 3, 50)
    X, Y = np.meshgrid(x_range, y_range)
    Z = X**2 + 2*Y**2 # f_1_1(X, Y)
    points_x, points_y = np.meshgrid(np.arange(-2, 3, 1), np.arange(-2, 3, 1))
    U = np.zeros_like(points_x, dtype=float)
    V = np.zeros_like(points_y, dtype=float)
    for i in range(points_x.shape[0]):
      for j in range(points_x.shape[1]):
        pt = np.array([points_x[i, j], points_y[i, j]])
        g = compute_gradient(f_1_1, pt)
        U[i, j] = g[0]
        V[i, j] = g[1]
    plt.figure(figsize=(8, 6))
    plt.contour(X, Y, Z, levels=15, cmap='viridis')
    plt.colorbar(label='$f(x, y) = x^2 + 2y^2$ Contour')
    plt.quiver(points_x, points_y, U, V, color='red', scale=40, width=0.005, headwidth=5, headlength=7)
    plt.title("Problem 1.1: Gradient Field of $f(x, y)$")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.gca().set_aspect('equal', adjustable='box')
    # 1. Create a meshgrid for x and y.
    # 2. Plot the contours of f_1_1.
    # 3. Define a few test points.
    # 4. Compute the gradient at each point.
    # 5. Plot the gradient vectors using plt.quiver.
    # 6. Save and show the plot.
    print("Problem 1.1: Plotting TODO: See comments in code.")


    # --- Problem 1.2 Verification ---

    # TODO:
    # 1. Define a numerical function f(x, y) = x^2 - y^2.
    def f_1_2(point):
     x = point[0]
     y = point[1]
     return x ** 2 - y ** 2


    test_point_1_2 = np.array([5.0, -3.0])
    # 2. Define a test point.
    # 3. Call your 'compute_hessian' function and print the result.
    hessian_scratch = compute_hessian(f_1_2, test_point_1_2, h=1e-5) # Replace this
    print_result("Problem 1.2", "Hessian of x^2-y^2 (Scratch)", hessian_scratch)

    # TODO: SymPy Verification
    # 1. Define symbolic x, y and the symbolic function.
    x_sym, y_sym = sympy.symbols('x y')
    f_sym = x_sym**2 - y_sym**2
    # 2. Use sympy.hessian() to get the symbolic Hessian.
    hessian_sympy_sym = sympy.hessian(f_sym, (x_sym, y_sym))
    # 3. Print the symbolic Hessian.
    hessian_sympy = hessian_sympy_sym # Replace this
    print_result("Problem 1.2", "Hessian of x^2-y^2 (SymPy)", hessian_sympy)


    # --- Problem 1.3 Verification ---

    # 1. Define the function to integrate
    def f_1_3(x, y):
        return x * np.sin(y)

    # 2. Set integration parameters
    a, b = 0.0, 1.0  # x-range
    c, d = 0.0, np.pi # y-range
    nx, ny = 100, 100

    # 3. Call your 'from scratch' implementation
    # TODO: Call your 'double_integral' function.
    integral_scratch = double_integral(f_1_3, a, b, c, d, nx, ny) # Replace this
    print_result("Problem 1.3", "Double Integral of x*sin(y) (Scratch)", integral_scratch)

    # 4. Verify with SciPy
    # TODO:
    # 1. Use spi.dblquad() to get the "exact" value.
    # 2. Assign the value to 'integral_scipy'.
    # 3. Print the result.
    def f_scipy(y, x): # SciPy's dblquad expects the function signature f(y, x) for f(x, y)
        return x * np.sin(y)
    result_scipy = spi.dblquad(f_scipy, a, b, c, d)
    integral_scipy = result_scipy[0] # Replace this
    print_result("Problem 1.3", "Double Integral of x*sin(y) (SciPy)", integral_scipy)

    print("--- End of Verification ---")