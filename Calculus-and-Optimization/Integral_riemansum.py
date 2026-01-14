# Math4AI: Calculus & Optimization - Assignment 3
# Starter Code Template
#
# Your file should be named:
# Math4AI_Calculus&Optimization_FirstName_LastName_Assignment3_code.py

import numpy as np
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
# Part 1: Integral Calculus
# ====================================================================

# --------------------------------------------------------------------
# Problem 1.1: Numerical Integration using Riemann Sums
# --------------------------------------------------------------------

def riemann_integral(f, a, b, n):
    """
    Calculates the definite integral of a function f from a to b
    using the midpoint Riemann sum with n rectangles.

    Args:
        f (callable): The function to integrate.
        a (float): The lower bound of integration.
        b (float): The upper bound of integration.
        n (int): The number of rectangles (subintervals).

    Returns:
        float: The approximated integral, or None if not implemented.
    """

    # --- YOUR CODE HERE ---

    # TODO 1: Calculate the width of each rectangle.
    # Formula: delta_x = (b - a) / n [cite: 22]
    delta_x=(b-a)/n

    # TODO 2: Initialize the total sum.
    total_sum=0
    # TODO 3: Loop through each rectangle.
    # We use a 0-based index 'i' from 0 to n-1.
    for i in range(n):
        x_i= a + (i + 0.5) * delta_x
        total_sum += f(x_i)
    final_sum=total_sum*delta_x

    # TODO 4: Inside the loop, calculate the midpoint of the current rectangle.
    # Formula: x_i* = a + (i - 0.5) * delta_x (for i=1..n) [cite: 22]
    # For 0-based index 'i' (0..n-1), this becomes:
    # x_i* = a + ((i + 1) - 0.5) * delta_x = a + (i + 0.5) * delta_x

    # TODO 5: Evaluate the function at the midpoint.

    # TODO 6: Add the area of this rectangle to the total sum.
    # This loop implements the summation part of:
    # sum(f(x_i*)) [cite: 21]

    # TODO 7: Return the final total sum.
    # Multiply the sum of all heights by the common width (delta_x)
    # This completes the formula: sum(f(x_i*)) * delta_x [cite: 21]
    return final_sum


# ====================================================================
# --- Main Execution & Verification ---
# ====================================================================

# This block will only run when the script is executed directly
if __name__ == "__main__":
    print("=====================================================")
    print("Math4AI: Assignment 3 Verification")
    print("=====================================================")


    # --- Problem 1.1 Verification ---

    # 1. Define the Standard Normal PDF (phi(x)) [cite: 28]
    def phi(x):
        """Standard Normal Probability Density Function"""
        return (1 / np.sqrt(2 * np.pi)) * np.exp(-x ** 2 / 2)


    # 2. Set integration parameters [cite: 29]
    a_val = -1.0
    b_val = 1.0
    n_rects = 1000

    # 3. Call your 'from scratch' implementation [cite: 32]
    integral_scratch = riemann_integral(phi, a_val, b_val, n_rects)
    print_result("Problem 1.1", f"Integral of PDF from -1 to 1 (Scratch, n={n_rects})", integral_scratch)

    # 4. Verify with SciPy [cite: 33]
    # TODO:
    # 1. Use spi.quad() to compute the "exact" value of the integral.
    # 2. Extract the integral value from the result.
    # 3. Print the SciPy result.

    # spi.quad returns a tuple: (integral_value, error_estimate)
    integral_scipy, error_scipy = spi.quad(phi, a_val, b_val) # burda tuple qaytarir deye 1ci integralin qiymeti olur 2ci ise error(hemise bele olur)
    print_result("Problem 1.1", "Integral of PDF from -1 to 1 (SciPy)", integral_scipy)

    print("--- End of Verification ---")
