import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erf
from scipy.stats import binom # For verification only
import math # Import the math module

np.random.seed(42)

# ==========================================
# 1. Normal Distribution & Anomalies
# ==========================================

def gaussian_pdf(x, mu, sigma):
    """
    Implementation of univariate Gaussian PDF.
    """
    # TODO: Implement 1/(sigma * sqrt(2pi)) * exp(...)
    # <YOUR CODE HERE>
    sigma = max(sigma, 1e-9) #edge case di 0 ola bilmez 
    const = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return const*np.exp(exponent)

def gaussian_cdf(x, mu, sigma):
    """
    Implementation of CDF using Error Function.
    """
    # TODO: Implement 0.5 * (1 + erf(...))
    # <YOUR CODE HERE>
    arg = (x - mu) / (sigma * np.sqrt(2))

    return 0.5 * (1 + erf(arg))

def run_section_1():
    print("--- Section 1: Anomaly Detection ---")
    # Generate synthetic server temp data
    data = np.concatenate([
        np.random.normal(50, 5, 1000), # Normal operations
        [25, 80, 78, 20]               # Anomalies
    ])
    
    # TODO: Calculate MLE (Mean and Std) from data
    # <YOUR CODE HERE>
    mu_hat = np.mean(data)
    sigma_hat = np.std(data)
    
    print(f"Estimated Mean: {mu_hat:.4f}")    # Record in Report
    print(f"Estimated Std Dev: {sigma_hat:.4f}") # Record in Report
    
    # TODO: Detect Anomalies using 99% interval
    # (i.e., data points where CDF < 0.005 OR CDF > 0.995)
    # <YOUR CODE HERE>
    probs = gaussian_cdf(data, mu_hat, sigma_hat)
    mask = (probs < 0.005) | (probs > 0.995)
    anomalies = data[mask]
    
    print(f"Anomalies detected: {len(anomalies)}")
    
    # Plotting
    x_grid = np.linspace(min(data), max(data), 1000)
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=50, density=True, alpha=0.5, label='Data Hist')
    plt.plot(x_grid, gaussian_pdf(x_grid, mu_hat, sigma_hat), 'k', linewidth=2, label='Fitted PDF')
    
    # Plot red dots for anomalies
    if len(anomalies) > 0:
        plt.scatter(anomalies, np.zeros_like(anomalies), color='red', s=50, zorder=5, label='Anomalies')
        
    plt.title('Server Temperatures: Fit & Anomalies')
    plt.legend()
    plt.show()

# ==========================================
# 2. Poisson vs Binomial Limit
# ==========================================

def poisson_pmf(k, lam):
    """
    Implementation of Poisson PMF.
    P(X=k) = (lam^k * e^-lam) / k!
    """
    # TODO: Implement formula. 
    # Hint: Use np.math.factorial(k) or compute iteratively.
    # <YOUR CODE HERE>
    if k < 0: return 0.0 #k menfi ola bilmez
    if lam < 0: raise ValueError("lambda minus ola bilmez")
    P=(np.exp(-lam)*(lam**k))/(math.factorial(k))
    return P

def run_section_2():
    print("\n--- Section 2: Poisson Limit ---")
    n = 10000
    p = 0.001
    lam = n * p
    target_k = 15
    
    # 1. Empirical (Simulation)
    # Simulate n coin flips, repeat 5000 times to get distribution of successes
    trials = 5000
    successes = np.random.binomial(n, p, trials) # Using numpy's generator for speed
    empirical_prob = np.mean(successes == target_k)
    
    # 2. Exact Binomial (SciPy)
    exact_prob = binom.pmf(target_k, n, p)
    
    # 3. Poisson Approximation (Your Implementation)
    approx_prob = poisson_pmf(target_k, lam)
    
    print(f"For k={target_k}:")
    print(f"  Empirical: {empirical_prob:.6f}") # Record in Report
    print(f"  Exact:     {exact_prob:.6f}")     # Record in Report
    print(f"  Poisson:   {approx_prob:.6f}")    # Record in Report
    
    # Comparison Plot
    k_range = np.arange(0, 30)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, binom.pmf(k_range, n, p), 'bo', label='Binomial Exact')
    plt.plot(k_range, [poisson_pmf(k, lam) for k in k_range], 'r-', alpha=0.6, label='Poisson Approx')
    plt.title(f'Limit Theorem: n={n}, p={p}')
    plt.xlabel('Number of Clicks (k)')
    plt.legend()
    plt.show()

# ==========================================
# 3. Bivariate Distributions
# ==========================================

def bivariate_normal_pdf(x, y, mux, muy, sigx, sigy, rho):
    """
    Returns PDF value at (x,y)
    """
    # TODO: Implement Bivariate Gaussian Formula
    # z = ((x-mux)/sigx)^2 - 2*rho*... + ...
    # norm_const = 1 / (2 * pi * sigx * sigy * sqrt(1-rho^2))
    # <YOUR CODE HERE>
    rho = np.clip(rho, -0.9999, 0.9999) #rho tam 1 or -1 olsa islemir
    sigx = max(sigx, 1e-9)
    sigy = max(sigy, 1e-9)

    const = 1 / (2 * np.pi * sigx * sigy * np.sqrt(1 - rho**2))
    term_x = ((x - mux) / sigx)**2
    term_y = ((y - muy) / sigy)**2
    term_rho = 2 * rho * ((x - mux) / sigx) * ((y - muy) / sigy)
    z = term_x - term_rho + term_y
    return const * np.exp(-z / (2 * (1 - rho**2)))

def run_section_3():
    print("\n--- Section 3: Bivariate & Marginals ---")
    
    # Setup parameters
    mux, muy = 0, 0
    sigx, sigy = 1, 1
    rho = 0.7 # Correlation
    
    # Create grid
    x = np.linspace(-3, 3, 100)
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)
    
    # Calculate Joint PDF
    for i in range(len(x)):
        for j in range(len(y)):
            Z[j, i] = bivariate_normal_pdf(x[i], y[j], mux, muy, sigx, sigy, rho)
            
    # TODO: Compute Marginal P(X) numerically
    # Sum over the Y axis (rows) and normalize
    marginal_x = np.sum(Z, axis=0)
    marginal_x /= np.sum(marginal_x) * (x[1] - x[0]) # Normalize by area
    
    # Theoretical univariate for comparison
    theory_x = gaussian_pdf(x, mux, sigx)
    
    # Plotting (Contour + Side Marginals)
    # Note: This is a complex plot, we have set it up for you.
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2,  width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    
    ax = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)
    
    # Contour
    ax.contourf(X, Y, Z, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Marginal X
    ax_histx.plot(x, theory_x, 'k--', label='Theory')
    ax_histx.plot(x, marginal_x, 'r-', label='Computed')
    ax_histx.set_title(f"Joint PDF (rho={rho})")
    ax_histx.legend(fontsize='small')
    
    # Marginal Y (Visual placeholder)
    ax_histy.plot(np.sum(Z, axis=1), y, 'r-')
    
    plt.show()

if __name__ == "__main__":
    run_section_1()
    run_section_2()
    run_section_3()