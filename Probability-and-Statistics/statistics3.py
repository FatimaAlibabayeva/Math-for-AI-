"""
Math4AI: Probability & Statistics
Assignment 3: Moments, Stability, and Correlation
Starter Code

Note to Students:
- Task 3.3 requires Matrix Algebra. Do not use loops for Covariance calculation.
- Reproducibility (np.random.seed) is essential.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # For Heatmap

np.random.seed(42)

# ==========================================
# 1. Law of Large Numbers (Geometric Dist)
# ==========================================

class MomentEstimator:
    @staticmethod
    def geometric_theoretical_stats(p):
        """
        Returns theoretical Mean and Variance for Geometric distribution.
        """
        # TODO: Implement formulas for Mean (1/p) and Variance ((1-p)/p^2)
        # <YOUR CODE HERE>
        mu = 1/p
        var = (1-p)/(p**2)
        
        return mu, var

    @staticmethod
    def analyze_convergence(data):
        """
        Computes running sample means to visualize convergence.
        Returns a list/array of means where index i is mean of data[0...i].
        """
        # Hint: np.cumsum() is very efficient for running totals!
        # running_mean = cumsum / (index + 1)
        # <YOUR CODE HERE>
        running_mean=np.cumsum(data) / np.arange(1, len(data) + 1) #np.arange() range in numpy versionu kimidi ve daha suretlidi
        
        return running_mean

def run_section_1():
    print("--- Section 1: Law of Large Numbers ---")
    p = 0.2
    N = 10000
    
    # 1. Theoretical
    mu_theory, var_theory = MomentEstimator.geometric_theoretical_stats(p)
    
    # 2. Simulation (Generate N samples)
    # NumPy's geometric is usually number of trials to success
    data = np.random.geometric(p, N)
    
    # 3. Convergence
    running_means = MomentEstimator.analyze_convergence(data)
    final_mean = running_means[-1]
    
    print(f"Theoretical Mean: {mu_theory:.4f}")
    print(f"Theoretical Var:  {var_theory:.4f}")
    print(f"Final Sample Mean (N={N}): {final_mean:.4f}")
    
    # 4. Error Analysis
    ns = np.arange(1, N + 1)
    errors = np.abs(running_means - mu_theory)
    expected_error_scale = np.sqrt(var_theory) / np.sqrt(ns) # SE = sigma / sqrt(n)
    
    # Plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot A: Convergence
    ax1.plot(ns, running_means, label='Running Sample Mean', color='blue', alpha=0.8)
    ax1.axhline(mu_theory, color='red', linestyle='--', label='Theoretical Mean')
    ax1.set_title("LLN Convergence: Geometric(p=0.2)")
    ax1.set_xlabel("Sample Size (N)")
    ax1.set_ylabel("Mean Value")
    ax1.legend()
    
    # Plot B: Error Rates (Log-Log)
    ax2.loglog(ns, errors, label='Absolute Error', alpha=0.5)
    ax2.loglog(ns, expected_error_scale, 'r--', label=r'Theory $1/\sqrt{N}$')
    ax2.set_title("Standard Error Decay")
    ax2.set_xlabel("Sample Size (N)")
    ax2.set_ylabel("Absolute Error")
    ax2.legend()
    
    plt.tight_layout()
    plt.show()

# ==========================================
# 2. Higher-Order Moments (Robustness)
# ==========================================

def compute_central_moment(data, k):
    """
    Computes the k-th central moment: E[(X - mu)^k] / sigma^k
    """
    # TODO:
    # <YOUR CODE HERE>
    mean=np.mean(data)
    std=np.std(data,ddof=1)  #biz standart deviantioni(datalarin meanden orta hesabla ne qeder uzaqlasidiqi) n-1(ddof) bolmeyiki ne yox cunki n e bolende netice hmeise realdan daha az cixir
    z=(data-mean)/(std +1e-15)#Ps.1e-15 i 0 deivsion olmasin deye yaziram   #bunu edirik cunki ferqli datalari(meselen boy,ceki ve sair) eyni tereziye getiremk isdeyirik
    moment=np.mean(z**k)
    # 1. Calculate sample mean and std (use ddof=1 for std)
    # 2. Standardize the data: z = (x - mean) / std
    # 3. Compute mean of z^k
    
    return moment

def run_section_2():
    print("\n--- Section 2: Skewness & Kurtosis ---")
    N = 5000
    
    # Dataset A: Normal
    data_a = np.random.normal(0, 1, N)
    
    # Dataset B: Student-t (Heavy tails)
    # df=3 implies finite variance but infinite skew/kurtosis theoretically, 
    # but we will measure sample stats.
    data_b = np.random.standard_t(df=3, size=N)
    
    # Compute Moments
    skew_a = compute_central_moment(data_a, 3)
    kurt_a = compute_central_moment(data_a, 4)
    
    skew_b = compute_central_moment(data_b, 3)
    kurt_b = compute_central_moment(data_b, 4)
    
    print(f"{'Statistic':<15} | {'Normal':<10} | {'Student-t':<10}")
    print("-" * 45)
    print(f"{'Skewness':<15} | {skew_a:7.4f}    | {skew_b:7.4f}")
    print(f"{'Kurtosis':<15} | {kurt_a:7.4f}    | {kurt_b:7.4f}")
    
    # Simple histogram comparison
    plt.figure(figsize=(10, 5))
    plt.hist(data_a, bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(data_b, bins=50, alpha=0.5, label='Student-t (df=3)', density=True)
    plt.title("Visualizing Heavy Tails")
    plt.yscale('log') # Log scale makes tails visible
    plt.legend()
    plt.show()

# ==========================================
# 3. Covariance & Matrix Algebra
# ==========================================

class MultivariateEstimator:
    @staticmethod
    def covariance_matrix(X):
        """
        Computes Covariance Matrix of X (shape N x D) using Matrix Algebra.
        Must return shape (D, D).
        """
        N, D = X.shape
        
        # TODO: 1. Compute column means (shape D,)
        # <YOUR CODE HERE>
        mean=np.mean(X,axis=0) #axis=0 sutun uzre tapir
        
        # TODO: 2. Center the data (X_centered = X - means)
        # Note: Broadcasting handles the shape automatically.
        X_centered=X-mean #burda eger cavabi menfialiriqsa demeli ortalamdan boyuk eks haldasa kicikdi boy
        # TODO: 3. Compute (X_c.T @ X_c) / (N - 1)
        # Use np.dot or the @ operator
        Sigma = (X_centered.T@X_centered)/(N-1)
        
        return Sigma

    @staticmethod
    def correlation_matrix(Sigma):
        """
        Converts Covariance Matrix Sigma to Correlation Matrix.
        Corr_ij = Sig_ij / (std_i * std_j)
        """
        # TODO: Extract variances (diagonal) and compute std devs
        # Hint: np.diag(Sigma) gives the variance array
        variances=np.diag(Sigma) #diagnola elementleri bize vairanci i kenar elemletler ise covrarianci verir
        # TODO: Compute Outer Product of standard deviations
        # outer_std = np.outer(std, std)
        std_devs = np.sqrt(variances) #bu ise standart deviationdir
        # TODO: Element-wise division
        outer_std = np.outer(std_devs, std_devs) +1e-15 #np.outer birbasa 2 siyahini bir birine vuraraq bir matrix yaradir
        Corr = Sigma/outer_std

        return Corr

def run_section_3():
    print("\n--- Section 3: Multivariate Analysis ---")
    
    # Generate Synthetic Correlated Data
    # X1 = random
    # X2 = 2*X1 + noise (Strongly correlated)
    # X3 = random (Independent)
    N = 1000
    x1 = np.random.randn(N)
    x2 = 2 * x1 + np.random.randn(N) * 0.5
    x3 = np.random.randn(N)
    
    # Stack into Matrix X (N x 3)
    X = np.column_stack([x1, x2, x3])
    
    # 1. Compute Covariance
    Sigma = MultivariateEstimator.covariance_matrix(X)
    print("Covariance Matrix:")
    print(np.round(Sigma, 4))
    
    # 2. Compute Correlation
    Corr = MultivariateEstimator.correlation_matrix(Sigma)
    
    # 3. Heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(Corr, annot=True, cmap='coolwarm', vmin=-1, vmax=1,
                xticklabels=['X1', 'X2', 'X3'], yticklabels=['X1', 'X2', 'X3'])
    plt.title("Correlation Matrix Heatmap")
    plt.show()

if __name__ == "__main__":
    run_section_1()
    run_section_2()
    run_section_3()