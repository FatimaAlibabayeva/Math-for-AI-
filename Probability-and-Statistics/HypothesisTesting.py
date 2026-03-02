"""
Math4AI: Probability & Statistics — Assignment 5 (Student Starter)
Hypothesis Testing & A/B Testing

Fill in the TODOs. Keep function signatures unchanged.

Expected outputs (when main is run):
- power_curve.png
- confidence_intervals.png
- printed test results for Tasks 5.1–5.3
"""

from __future__ import annotations

import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Save all generated figures next to this file (robust to different working directories)
OUTPUT_DIR = Path(__file__).resolve().parent


# You may use scipy.stats for CDF/PPF/SF utilities (recommended).
try:
    from scipy import stats
except Exception:  # pragma: no cover
    stats = None


# ============================================================
# Provided Data (DO NOT CHANGE)
# ============================================================

# Task 5.1 example dataset for z-test (e.g., latency measurements in ms)
Z_TEST_DATA = np.array([
    101.2, 98.7, 103.5, 99.1, 100.4, 102.0, 97.9, 101.8, 99.9, 100.7,
    98.4, 101.1, 102.9, 99.3, 100.2, 101.5, 98.8, 100.0, 99.6, 102.2
], dtype=float)
MU0_Z = 100.0
SIGMA_Z = 15.0  # assumed known population sigma


# Task 5.2: model accuracies across random seeds (A/B test)
MODEL_A_ACCURACIES = np.array([
    0.812, 0.805, 0.809, 0.814, 0.803, 0.811, 0.807, 0.810, 0.808, 0.806,
    0.813, 0.804, 0.809, 0.807, 0.812, 0.806, 0.810, 0.808, 0.805, 0.811
], dtype=float)

MODEL_B_ACCURACIES = np.array([
    0.821, 0.816, 0.819, 0.823, 0.815, 0.822, 0.818, 0.820, 0.817, 0.816,
    0.824, 0.814, 0.820, 0.818, 0.823, 0.816, 0.821, 0.819, 0.815, 0.822
], dtype=float)

# Task 5.3: contingency table (rows = user segment, cols = churn status)
# Columns: [Retained, Churned]
OBSERVED_CONTINGENCY = np.array([
    [50, 30],   # Segment 1
    [45, 55],   # Segment 2
    [20, 60],   # Segment 3
], dtype=float)


# ============================================================
# Small distribution helpers (optional)
# ============================================================

def _norm_cdf(x: float) -> float:
    """Standard normal CDF."""
    if stats is not None:
        return float(stats.norm.cdf(x))
    # Fallback using erf
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Standard normal inverse CDF (quantile)."""
    if stats is None:
        raise RuntimeError("scipy is required for norm.ppf in this assignment.")
    return float(stats.norm.ppf(p))


# ============================================================
# Task 5.1 — Z-test and Power
# ============================================================

def z_test(data: np.ndarray, mu_0: float, sigma: float) -> tuple[float, float]:
    """
    Two-sided z-test for mean with known population sigma.
    Returns: (z_statistic, p_value_two_sided)
    """
    # TODO: compute sample mean, z-statistic, and two-sided p-value
    n=len(data)
    if n == 0: #edge case bos data
        return 0.0, 1.0
    if sigma <= 0: #sigma 0 olsa
        return (0.0, 0.0) if np.mean(data) == mu_0 else (np.inf, 0.0)
    mean=np.mean(data)
    z=(mean-mu_0)/(sigma/np.sqrt(n))
    p_val=2*(1-_norm_cdf(abs(z)))

    return float(z), float(p_val)


def compute_power(effect_size: float, alpha: float, n: int, sigma: float) -> float:
    """
    Compute power of a two-sided z-test with:
      H0: mu = mu0, H1: mu = mu0 + effect_size

    Under H1, the z-stat has mean:
      mean_z = effect_size / (sigma / sqrt(n))
    and variance 1.

    Returns power = P(reject H0 | H1 true).
    """
    # TODO: compute z_crit, mean_z, then power under N(mean_z, 1)
    z_crit=_norm_ppf(1-alpha/2)
    mean_z=effect_size/(sigma/np.sqrt(n))
    power=(1-_norm_cdf(z_crit-mean_z)) + _norm_cdf(-z_crit-mean_z) #power, citict pointden sagda ve solda qalan arealarin cemidi
    return float(power)


def plot_power_curve(effect_size: float, alpha: float, sigma: float, n_values: np.ndarray,
                     filename: str = "power_curve.png") -> None:
    """
    Plot power vs. n and save to disk.
    """
    powers = []
    for n in n_values:
        powers.append(compute_power(effect_size, alpha, int(n), sigma))
    powers = np.array(powers, dtype=float)

    plt.figure(figsize=(9, 5))
    plt.plot(n_values, powers, linewidth=2)
    plt.axhline(0.8, linestyle="--", linewidth=1)
    plt.xlabel("Sample size n")
    plt.ylabel("Power (1 - beta)")
    plt.title("Power Curve (Two-sided z-test)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.show()
    plt.close()


# ============================================================
# Task 5.2 — Welch's t-test and Confidence Interval
# ============================================================

def welch_t_test(data_a: np.ndarray, data_b: np.ndarray) -> tuple[float, float, float]:
    """
    Welch's two-sample t-test (unequal variances), two-sided.
    Returns: (t_statistic, dof, p_value_two_sided)
    """
    # TODO: compute t, dof (Welch-Satterthwaite), and p-value (two-sided)
    n1, n2 = len(data_a), len(data_b)
    if n1 < 2 or n2 < 2: #data lazimi qeder degilse
        return 0.0, 1.0, 1.0
    m1, m2 = np.mean(data_a), np.mean(data_b)
    v1, v2 = np.var(data_a, ddof=1), np.var(data_b, ddof=1)

    if v1 == 0 and v2 == 0:#eyer butun qruplardaki datalar samedise
        return (0.0, float(n1 + n2 - 2), 1.0) if m1 == m2 else (np.inf, float(n1 + n2 - 2), 0.0)
    t_stat = (m1 - m2) / np.sqrt(v1/n1 + v2/n2) 
    numerator = (v1/n1 + v2/n2)**2
    denominator = (v1/n1)**2 / (n1-1) + (v2/n2)**2 / (n2-1)
    dof = numerator / denominator
    p_val = 2 * stats.t.sf(np.abs(t_stat), dof)

    return float(t_stat), float(dof), float(p_val)


def compute_confidence_interval(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """
    Two-sided confidence interval for the mean using t distribution.
    Returns: (mean, ci_low, ci_high)
    """
    # TODO: compute mean, standard error, t critical value, and CI bounds
    n=len(data)
    mean=np.mean(data)
    std_err=stats.sem(data)
    t_crit = stats.t.ppf((1 + confidence) / 2, n - 1)
    ci_low = mean - t_crit * std_err
    ci_high = mean + t_crit * std_err
    
    return float(mean), float(ci_low), float(ci_high)


def plot_confidence_intervals(
    mean_a: float, ci_a: tuple[float, float],
    mean_b: float, ci_b: tuple[float, float],
    filename: str = "confidence_intervals.png"
) -> None:
    """
    Error bar plot for two mean confidence intervals.
    """
    means = np.array([mean_a, mean_b], dtype=float)
    lows = np.array([ci_a[0], ci_b[0]], dtype=float)
    highs = np.array([ci_a[1], ci_b[1]], dtype=float)
    yerr = np.vstack([means - lows, highs - means])

    plt.figure(figsize=(7, 4))
    plt.errorbar([0, 1], means, yerr=yerr, fmt="o", capsize=8)
    plt.xticks([0, 1], ["Model A", "Model B"])
    plt.ylabel("Accuracy")
    plt.title("95% Confidence Intervals for Mean Accuracy")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=200)
    plt.show()
    plt.close()


# ============================================================
# Task 5.3 — Chi-square test (vectorized expected matrix)
# ============================================================

def chi_squared_test(observed_matrix: np.ndarray) -> tuple[float, int, float]:
    """
    Pearson chi-square test of independence.
    Returns: (chi2_statistic, dof, p_value)

    REQUIREMENT: compute E using np.outer(row_sums, col_sums) / N.
    """
    # TODO: compute row sums, col sums, expected matrix E (vectorized),
    #       chi-square statistic, degrees of freedom, and p-value.
    row_sums = observed_matrix.sum(axis=1)
    col_sums = observed_matrix.sum(axis=0)
    total_n = observed_matrix.sum()
    if total_n == 0:
        return 0.0, 0, 1.0
    expected = np.outer(row_sums, col_sums) / total_n
    chi2_stat = np.sum((observed_matrix - expected)**2 / expected)
    dof = (observed_matrix.shape[0] - 1) * (observed_matrix.shape[1] - 1)
    p_val = stats.chi2.sf(chi2_stat, dof)
    
    return float(chi2_stat), int(dof), float(p_val)


# ============================================================
# Main (runs tasks + saves figures)
# ============================================================

def main() -> None:
    # ---------------- Task 5.1 ----------------
    z, pz = z_test(Z_TEST_DATA, MU0_Z, SIGMA_Z)
    print("Task 5.1 — Z-test")
    print("  z-stat:", z)
    print("  p-val :", pz)

    n_values = np.arange(5, 301, 5)
    plot_power_curve(effect_size=5.0, alpha=0.05, sigma=15.0, n_values=n_values, filename="power_curve.png")
    print("  saved: power_curve.png")

    # ---------------- Task 5.2 ----------------
    t, dof, pt = welch_t_test(MODEL_A_ACCURACIES, MODEL_B_ACCURACIES)
    mean_a, lo_a, hi_a = compute_confidence_interval(MODEL_A_ACCURACIES, confidence=0.95)
    mean_b, lo_b, hi_b = compute_confidence_interval(MODEL_B_ACCURACIES, confidence=0.95)

    print("\nTask 5.2 — Welch's t-test")
    print("  mean(A):", mean_a, "CI:", (lo_a, hi_a))
    print("  mean(B):", mean_b, "CI:", (lo_b, hi_b))
    print("  t-stat :", t)
    print("  dof    :", dof)
    print("  p-val  :", pt)

    plot_confidence_intervals(mean_a, (lo_a, hi_a), mean_b, (lo_b, hi_b), filename="confidence_intervals.png")
    print("  saved: confidence_intervals.png")

    # ---------------- Task 5.3 ----------------
    chi2, dof_c, pchi = chi_squared_test(OBSERVED_CONTINGENCY)
    print("\nTask 5.3 — Chi-square test of independence")
    print("  chi2  :", chi2)
    print("  dof   :", dof_c)
    print("  p-val :", pchi)


if __name__ == "__main__":
    main()
