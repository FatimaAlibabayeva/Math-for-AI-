"""
Math4AI: Probability & Statistics — Assignment 6 (Diamond Starter)
Inference, MLE, and Bootstrapping

Student Instructions:
- Complete the TODOs in Tasks 6.1–6.3.
- Run this file to generate required figures:
    variance_bias_N5.png
    variance_bias_N50.png
    gumbel_fit.png
    bootstrap_median.png
"""

from __future__ import annotations

import math
from typing import Callable, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Task 6.2 allowed: SciPy (for optimization + comparison fit)
from scipy.optimize import minimize
from scipy.stats import gumbel_r


# ============================================================
# Provided datasets (do NOT change)
# ============================================================

EXTREME_SENSOR_READINGS = np.array([20.2947, 20.582, 18.3123, 19.2273, 26.459, 8.7197, 11.1025, 10.9172, 16.9677, 13.2832, 13.9234, 9.5977, 11.9202, 10.6293, 13.3364, 15.3891, 12.5307, 18.7974, 10.5894, 12.0031, 11.1248, 14.0652, 8.8467, 9.9882, 10.975, 15.8258, 13.1602, 19.7643, 16.4658, 11.6528, 11.2211, 8.9209, 14.3268, 13.9263, 13.1391, 15.9008, 16.9301, 13.0935, 15.3932, 11.7434, 13.3846, 10.5475, 11.5341, 14.4554, 10.551, 10.7731, 13.7825, 14.696, 11.6566, 16.7339, 15.7511, 24.2692, 10.9106, 11.7773, 11.5001, 10.9574, 13.2187, 12.4328, 16.733, 17.2668, 11.753, 13.1277, 12.4569, 11.0743, 11.9823, 12.5376, 12.4986, 14.538, 20.6601, 13.3879, 15.5518, 13.6102, 10.3691, 15.306, 19.0306, 14.7684, 14.642, 11.7973, 12.5137, 10.9336, 11.7809, 13.6275, 10.6999, 16.2499, 21.4335, 17.9015, 11.3801, 13.1964, 16.345, 12.9087, 16.5008, 11.5614, 13.3154, 13.8366, 14.5622, 12.0125, 14.0011, 17.9722, 17.1872, 9.04, 9.8194, 11.538, 14.9355, 8.8828, 10.9723, 11.4184, 13.2909, 14.8659, 17.7227, 9.8865, 13.2423, 15.7169, 14.5181, 12.3607, 16.093, 10.3404, 11.1217, 11.4004, 13.4238, 12.0327, 12.3272, 11.8795, 18.0697, 13.552, 19.8953, 12.9597, 14.289, 16.6432, 17.5376, 12.303, 16.191, 9.6191, 12.3481, 14.1334, 12.2806, 21.4233, 9.1051, 13.045, 10.9423, 18.1233, 13.0131, 12.9836, 9.4458, 12.4122, 13.1099, 14.9233, 14.2732, 12.7685, 13.371, 21.559, 10.6004, 9.9562, 16.7243, 12.5345, 17.4085, 11.7268, 14.77, 11.8143, 11.3474, 11.0474, 17.4289, 9.7323, 15.3519, 20.1669, 12.5288, 13.9224, 10.5715, 10.7457, 14.4093, 9.2077, 14.6694, 12.8084, 15.0463, 9.4714, 16.289, 19.7005, 13.4005, 8.039, 10.0033, 11.1934, 10.0124, 9.9848, 12.7813, 14.421, 11.0225, 11.799, 13.899, 17.7758, 11.2065, 14.9732, 9.4604, 15.221, 17.0819, 10.5605, 16.4835, 16.0543, 12.2228, 10.174, 15.8002, 14.4753], dtype=float)

SERVER_LATENCY_MS = np.array([332.53, 127.604, 220.848, 58.348, 185.256, 88.158, 35.638, 40.151, 48.572, 75.231, 34.321, 55.187, 46.087, 48.682, 77.553, 65.913, 87.19, 51.724, 42.795, 50.484, 59.434, 58.079, 37.355, 56.355, 59.138, 131.778, 105.309, 40.503, 49.374, 32.714, 44.401, 60.975, 83.267, 42.301, 43.426, 25.75, 51.577, 37.643, 45.363, 40.169, 52.826, 29.512, 32.673, 115.035, 34.793, 37.193, 103.847, 150.924, 36.232, 47.996, 61.531, 99.987, 38.652, 50.107, 71.67, 63.572, 47.863, 52.1, 33.743, 50.231, 49.738, 59.22, 44.954, 64.395, 77.824, 57.651, 61.751, 55.623, 54.6, 42.413, 60.994, 52.77, 113.592, 94.696, 62.492, 41.801, 36.99, 82.839, 59.857, 64.589, 29.648, 75.536, 64.01, 37.016, 46.292, 59.878, 55.61, 49.291, 52.656, 49.989, 57.593, 91.38, 22.235, 50.255, 58.078, 60.558, 47.934, 29.522, 61.24, 99.94, 31.917, 73.872, 48.668, 53.439, 37.769, 48.567, 86.057, 66.949, 100.114, 82.442, 63.668, 100.522, 63.666, 72.952, 49.215, 55.885, 42.773, 77.197, 36.147, 71.796], dtype=float)


# ============================================================
# Task 6.1 — Bias & consistency of variance estimators
# ============================================================

def simulate_variance_estimators(
    K: int,
    N: int,
    seed: int = 0
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulate K datasets of size N from N(0,1). For each dataset compute:
      - Biased variance estimator (MLE): (1/N) * sum (xi - xbar)^2
      - Unbiased corrected estimator:    (1/(N-1)) * sum (xi - xbar)^2

    Returns:
        sigma2_mle: shape (K,)
        s2_unbiased: shape (K,)
    """
    rng = np.random.default_rng(seed)
    if N < 2 : #edge case
        raise ValueError("N en azi 2 olmalidir")
    # TODO 1: Generate samples of shape (K, N) from standard normal.

    samples=rng.standard_normal((K,N)) #K*N formali cedvel yaradir ve normal distrubtipndan ora ededler getirib yazir

    # TODO 2: Compute the per-row mean xbar (shape (K, 1) or (K,)).

    xbar=samples.mean(axis=1,keepdims=True)  #axis 1 edirik cunki rowun meanini tapmaq isdeyirik ve her setrdeki elementlerden oz meanlerini cixaciq, bu butun setirlercin getsin deye keepdims lazimdi
    
    # TODO 3: Compute sum of squared deviations per row:
    #   ss = sum((samples - xbar)^2, axis=1)

    ss=((samples-xbar)**2).sum(axis=1) #her setir ucun etmekcin axis=1 yazram, her rowdaki ededlerden meani cixib sonra onlari toplayri ve row ucun bir gosterici olur

    # TODO 4: Compute the two estimators from ss.
    # sigma2_mle = ...
    # s2_unbiased = ...

    sigma_mle=ss/N # N e bolende daha kicik cixir(biased) Ps.Normal distde ss/N Mle dir
    s_unbiased=ss/(N-1) #burda ise unbiased olur
    return sigma_mle,s_unbiased


def plot_variance_bias(
    sigma2_mle: np.ndarray,
    s2_unbiased: np.ndarray,
    N: int,
    filename: str
) -> None:
    """
    Save an overlaid histogram plot of the two estimators.
    """
    plt.figure(figsize=(9, 5))
    bins = 40
    plt.hist(sigma2_mle, bins=bins, density=True, alpha=0.6, label=r"$\hat{\sigma}^2_{MLE}$")
    plt.hist(s2_unbiased, bins=bins, density=True, alpha=0.6, label=r"$S^2$ (unbiased)")
    plt.axvline(1.0, linestyle="--", linewidth=2, label="True variance = 1")

    plt.title(f"Variance Estimators under N(0,1) — N={{N}}")
    plt.xlabel("Estimate")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    plt.show()


# ============================================================
# Task 6.2 — Numerical MLE for Gumbel(mu, beta)
# ============================================================

def gumbel_neg_log_likelihood(params: np.ndarray, data: np.ndarray) -> float:
    """
    Negative log-likelihood for Gumbel(loc=mu, scale=beta), beta>0:

        NLL(mu,beta) = N*log(beta) + sum_i [ z_i + exp(-z_i) ]
        where z_i = (x_i - mu)/beta

    Returns +inf if beta <= 0.
    """
    mu, beta = float(params[0]), float(params[1])

    # TODO 5: Enforce beta>0. If invalid, return np.inf.
    # if beta <= 0: return np.inf
    if len(data) == 0:
        raise ValueError("data cannot be empty")
    if beta<=0:
        return np.inf
    # TODO 6: Compute z = (data - mu)/beta
    z=(data-mu)/beta
    # TODO 7: Return N*log(beta) + sum(z + exp(-z))
    N=len(data)
    # Hint: use np.exp, np.log, and vectorized operations.

    return N*np.log(beta) + np.sum(z+np.exp(-z)) #logluyanda log(x)=


def fit_gumbel_mle(data: np.ndarray) -> Tuple[float, float]:
    """
    Fit (mu, beta) by minimizing NLL using scipy.optimize.minimize.

    Returns:
        mu_hat, beta_hat
    """
    # Reasonable initial guess (students may change): mu ~ mean, beta ~ std
    mu0 = float(np.mean(data))
    beta0 = float(np.std(data, ddof=0)) + 1e-6
    x0 = np.array([mu0, beta0], dtype=float)

    # TODO 8: Call scipy.optimize.minimize on gumbel_neg_log_likelihood.
    # Recommended: method="L-BFGS-B" with bounds:
    #   mu: (None, None)
    #   beta: (1e-8, None)
    # result = ...
    
    result = minimize(
       gumbel_neg_log_likelihood,
       x0,
       args=(data,),
       method="L-BFGS-B",
       bounds=[(None, None), (1e-8, None)]
    )
    # TODO 9: Extract mu_hat, beta_hat from result.x and return.
    mu_hat, beta_hat = float(result.x[0]), float(result.x[1])
    return mu_hat, beta_hat


def plot_gumbel_fit(data: np.ndarray, mu_hat: float, beta_hat: float, filename: str) -> None:
    """
    Save histogram with fitted Gumbel PDF overlay.
    """
    plt.figure(figsize=(9, 5))
    plt.hist(data, bins=30, density=True, alpha=0.6, label="Data")

    xs = np.linspace(np.min(data) - 1.0, np.max(data) + 1.0, 400)
    pdf = gumbel_r.pdf(xs, loc=mu_hat, scale=beta_hat)
    plt.plot(xs, pdf, linewidth=2, label=f"Fitted Gumbel PDF (mu={{mu_hat:.3f}}, beta={{beta_hat:.3f}})")

    plt.title("Gumbel MLE Fit (Numerical Optimization)")
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    plt.show()


# ============================================================
# Task 6.3 — Bootstrap CI for the median
# ============================================================

def bootstrap_ci(
    data: np.ndarray,
    statistic_func: Callable[[np.ndarray], float],
    n_bootstraps: int = 10_000,
    ci_level: float = 0.95,
    seed: int = 0
) -> Tuple[float, float, np.ndarray]:
    """
    Percentile bootstrap confidence interval for a statistic.

    Returns:
        (ci_low, ci_high, boot_stats)
    """
    rng = np.random.default_rng(seed)
    data = np.asarray(data, dtype=float)
    n = data.size

    # TODO 10: Generate bootstrap resamples of indices (shape: (n_bootstraps, n)).
    # idx = ...
    if n_bootstraps < 1:
        raise ValueError("n_bootstraps should be positive") #edge case
    idx=rng.integers(0,n,size=(n_bootstraps,n)) #tek meqsedimiz confidence intervali hesablamaqdi amma datadan hazir dusturnan yox
    samples=data[idx]
    boot_stats = np.apply_along_axis(statistic_func, 1, samples)
    # TODO 11: Build bootstrap samples and compute statistic on each.
    # Hint: samples = data[idx]  gives shape (B, n)
    # boot_stats = np.apply_along_axis(statistic_func, 1, samples)
    # boot_stats = ...
    # TODO 12: Compute percentile CI endpoints.
    # alpha = (1 - ci_level)/2
    # ci_low = np.percentile(boot_stats, 100*alpha)
    # ci_high = np.percentile(boot_stats, 100*(1-alpha))

    alpha = (1 - ci_level) / 2
    ci_low  = np.percentile(boot_stats, 100 * alpha)
    ci_high = np.percentile(boot_stats, 100 * (1 - alpha))
    return ci_low, ci_high, boot_stats


def plot_bootstrap_hist(
    boot_stats: np.ndarray,
    ci_low: float,
    ci_high: float,
    filename: str
) -> None:
    plt.figure(figsize=(9, 5))
    plt.hist(boot_stats, bins=40, density=True, alpha=0.7, label="Bootstrap statistics")
    plt.axvline(ci_low, linestyle="--", linewidth=2, label=f"CI low = {ci_low:.3f}")
    plt.axvline(ci_high, linestyle="--", linewidth=2, label=f"CI high = {ci_high:.3f}")
    plt.title("Bootstrap Distribution (Median) with 95% CI")
    plt.xlabel("Bootstrap median")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()
    plt.show()


# ============================================================
# Main (generates required outputs once TODOs are done)
# ============================================================

def main() -> None:
    # ---- Task 6.1 ----
    for N, out in [(5, "variance_bias_N5.png"), (50, "variance_bias_N50.png")]:
        sigma2_mle, s2_unbiased = simulate_variance_estimators(K=10_000, N=N, seed=0)

        print(f"[Task 6.1] N={N}")
        print("  mean(MLE)      =", float(np.mean(sigma2_mle)))
        print("  bias(MLE)      =", float(np.mean(sigma2_mle) - 1.0))
        print("  mean(unbiased) =", float(np.mean(s2_unbiased)))
        print("  bias(unbiased) =", float(np.mean(s2_unbiased) - 1.0))

        plot_variance_bias(sigma2_mle, s2_unbiased, N=N, filename=out)
        print("  saved:", out)

    # ---- Task 6.2 ----
    mu_hat, beta_hat = fit_gumbel_mle(EXTREME_SENSOR_READINGS)
    mu_scipy, beta_scipy = gumbel_r.fit(EXTREME_SENSOR_READINGS)

    print("\n[Task 6.2] Gumbel MLE (numerical)")
    print("  mu_hat, beta_hat =", mu_hat, beta_hat)
    print("  scipy fit         =", mu_scipy, beta_scipy)

    plot_gumbel_fit(EXTREME_SENSOR_READINGS, mu_hat, beta_hat, filename="gumbel_fit.png")
    print("  saved: gumbel_fit.png")

    # ---- Task 6.3 ----
    ci_low, ci_high, boot_stats = bootstrap_ci(
        SERVER_LATENCY_MS,
        statistic_func=np.median,
        n_bootstraps=10_000,
        ci_level=0.95,
        seed=0,
    )
    print("\n[Task 6.3] Bootstrap CI for median (server latency)")
    print("  CI =", (ci_low, ci_high))

    plot_bootstrap_hist(boot_stats, ci_low, ci_high, filename="bootstrap_median.png")
    print("  saved: bootstrap_median.png")


if __name__ == "__main__":
    main()
