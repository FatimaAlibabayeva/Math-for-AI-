import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import rosen, rosen_der # For verification

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
        elif isinstance(value, (np.ndarray)):
             np.set_printoptions(precision=6, suppress=True)
             print(value)
        else:
             print(value)
    print("-" * 40)

# ====================================================================
# Part 3: Advanced Second-Order and Quasi-Newton Methods
# ====================================================================

# --------------------------------------------------------------------
# Problem 3.2: The BFGS Algorithm
# --------------------------------------------------------------------

def bfgs_algorithm(f, f_grad, x0, max_iter=100, tol=1e-6):
    """
    Performs optimization using the BFGS (Broyden–Fletcher–Goldfarb–Shanno)
    algorithm from scratch.

    Args:
        f (callable): The objective function to minimize.
        f_grad (callable): The gradient of the objective function.
        x0 (np.ndarray): The starting point (e.g., np.array([1.0, 2.0])).
        max_iter (int): Maximum number of iterations.
        tol (float): Tolerance for the gradient norm to stop.

    Returns:
        (np.ndarray, list): A tuple of (final_point, path_history)
                            where path_history is a list of points.
    """
    #arg checks
    if not callable(f) or not callable(f_grad):
        raise ValueError("f and f_grad must be callable functions.")
    if not isinstance(max_iter, int) or max_iter <= 0:
        raise ValueError("max_iter must be a positive integer.")
    if not isinstance(tol, (int, float)) or tol <= 0:
        raise ValueError("tol must be a positive number.")
    
    x_k = np.asarray(x0, dtype=float).flatten()
    if x_k.ndim != 1:
        raise ValueError("x0 must be a 1D array-like object.")
    n = len(x0) # Number of variables
    x_k = np.asarray(x0, dtype=float)

    # Initialize B_k_inv (the inverse Hessian approximation) as the identity matrix
    B_k_inv = np.identity(n)

    # Store the path
    path_history = [x_k]

    # --- YOUR CODE HERE ---

    # TODO 1: Calculate the initial gradient at x_k.
    grad_k = f_grad(x_k)

    # TODO 2: Start the main iteration loop (up to max_iter).
    for i in range(max_iter):
        if np.linalg.norm(grad_k)< tol: # Corrected 'gard_k' to 'grad_k'
            break
        p_k=-np.dot(B_k_inv, grad_k)

        alpha_k=1.0
        rho=0.5
        c=1e-4
        while f(x_k + alpha_k * p_k) > f(x_k) + c * alpha_k * np.dot(grad_k, p_k):
            alpha_k *= rho      #sert ucun(backtracking)

        x_next=x_k +alpha_k*p_k
        grad_next=f_grad(x_next)
        s_k=x_next -x_k
        y_k=grad_next-grad_k #todo 8
        ys=np.dot(y_k,s_k)
        if abs(ys)>1e-10:
            I=np.identity(n)
            rho_k = 1.0 / ys
            term1 = I - rho_k * np.outer(s_k, y_k)
            term2 = I - rho_k * np.outer(y_k, s_k)
            B_k_inv = np.dot(term1, np.dot(B_k_inv, term2)) + rho_k * np.outer(s_k, s_k) #todo 10

        x_k = x_next
        grad_k = grad_next #for the next iteration

        path_history.append(x_k.copy())


        # TODO 3: Check for convergence using the gradient norm and 'tol'.

        # TODO 4: Calculate the search direction 'p_k'.

        # TODO 5: Perform a line search to find the step size 'alpha_k'.
        #         (A simple fixed step or backtracking search is fine).

        # TODO 6: Update the position to 'x_k_plus_1'.

        # TODO 7: Calculate the new gradient at 'x_k_plus_1'.

        # TODO 8: Calculate the update vectors 's_k' and 'y_k'.

        # TODO 9: Check if the update is valid (e.g., denominator not zero).

        # TODO 10: Update the inverse Hessian approximation 'B_k_inv'
        #          using the BFGS formula.

        # TODO 11: Update 'x_k' and 'grad_k' for the next iteration.

        # TODO 12: Append the new position to 'path_history'.


    # TODO 13: Return the final position and the path history.

    return x_k, path_history


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

if __name__ == "__main__":

    print("=====================================================")
    print("Math4AI: Assignment 6 Verification")
    print("=====================================================")

    # --- Problem 3.2 Verification ---

    # We will use the famous Rosenbrock function.
    # The global minimum is at (1, 1).

    start_point = np.array([-1.5, 1.0])

    # TODO:
    # 1. Call your 'bfgs_algorithm' function with 'rosen', 'rosen_der',
    #    and the 'start_point'.
    bfgs_result, bfgs_path = bfgs_algorithm(rosen, rosen_der, start_point)

    # 2. Print your final result.
    print_result("Problem 3.2", "BFGS Result (Scratch)", bfgs_result)

    # 3. (Optional) Use scipy.optimize.minimize(rosen, start_point, method='BFGS')
    #    to get a verification result and print it.
    from scipy.optimize import minimize
    res_scipy = minimize(rosen, start_point, method='BFGS')
    print_result("Scipy BFGS Result", "Result", res_scipy.x) #checking with spicy

    # 4. (For the report) Plot the convergence.
    #    - Plot the contours of the Rosenbrock function.
    #    - Plot the path taken by your algorithm (bfgs_path).
    #    - Add labels, a title, and save/show the plot.
    path = np.array(bfgs_path)
    x = np.linspace(-2.0, 2.0, 400)
    y = np.linspace(-1.0, 3.0, 400)
    X, Y = np.meshgrid(x, y)
    Z = rosen([X, Y])
    plt.figure(figsize=(10, 8))
    cp = plt.contour(X, Y, Z, levels=np.logspace(-0.5, 3.5, 20), cmap='viridis', alpha=0.6)
    plt.clabel(cp, inline=True, fontsize=8)
    plt.plot(path[:, 0], path[:, 1], 'r-o', markersize=4, linewidth=1.5, label='BFGS Path (Scratch)')

    plt.plot(start_point[0], start_point[1], 'go', label=f'Start {start_point}')
    plt.plot(1, 1, 'b*', markersize=15, label='Target Minimum (1, 1)')

    plt.title("BFGS Convergence on Rosenbrock Function", fontsize=14)
    plt.xlabel("x", fontsize=12)
    plt.ylabel("y", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.show() #gosdermeycin

    print("--- End of Verification ---")