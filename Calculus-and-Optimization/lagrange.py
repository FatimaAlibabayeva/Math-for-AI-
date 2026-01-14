"""
Math4AI — Assignment 7 (Starter)
Constrained Optimization & Portfolio Theory (Bonus: KKT view of SVM support vectors)

What you must implement (see TODOs):
  Part 1 (Equality constraints / Lagrange):
    - solve_portfolio_lagrange(mu, Sigma, target_return)

  Part 2 (Inequality constraints / KKT with w >= 0):
    - solve_portfolio_kkt(mu, Sigma, target_return)

What we provide:
  - helper functions for return/risk
  - efficient frontier loop + plotting helper (so you don’t fight matplotlib)

Allowed: numpy, matplotlib
Required for Part 2: scipy.optimize.minimize with method="SLSQP"
Not allowed (unless optional verification): CVXPy / high-level QP solvers

Student: <YOUR NAME>
"""

from __future__ import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None  # Part 2 requires SciPy; we raise a helpful error in that function.


# ============================================================
# Helpers
# ============================================================
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected return: mu^T w."""
    return float(np.dot(mu, w))


def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Variance (risk): w^T Sigma w."""
    return float(w.T @ Sigma @ w)


def portfolio_risk_std(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Standard deviation risk: sqrt(w^T Sigma w)."""
    var = portfolio_variance(w, Sigma)
    return float(np.sqrt(max(var, 0.0)))


def feasible_target_grid(mu: np.ndarray, num: int = 25, eps: float = 1e-9) -> np.ndarray:
    """
    Under w >= 0 and sum(w)=1, achievable returns lie in [min(mu), max(mu)].
    We generate a grid in that range.
    """
    lo = float(np.min(mu)) + eps
    hi = float(np.max(mu)) - eps
    if hi <= lo:
        return np.array([float(np.mean(mu))])
    return np.linspace(lo, hi, num=num)


# ============================================================
# Part 1: Equality constraints (Lagrange multipliers)
# ============================================================
def solve_portfolio_lagrange(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float
) -> tuple[np.ndarray, float, float]:
    """
    Solve the equality-constrained Markowitz problem (short selling allowed):

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return

    Derivation leads to the block KKT system:

        [ Sigma   1    mu ] [  w  ]   [ 0 ]
        [ 1^T     0     0 ] [ l1  ] = [ 1 ]
        [ mu^T    0     0 ] [ l2  ]   [ R ]

    Returns:n        w  : (n,) optimal weights
        l1 : multiplier for the budget constraint
        l2 : multiplier for the return constraint (shadow price)
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size

    ones = np.ones((n, 1))

    if not np.allclose(Sigma, Sigma.T): #simmetrik olmaidi
        Sigma = (Sigma + Sigma.T) / 2

    if np.any(np.linalg.eigvals(Sigma) <= 0): #musbet olmalidi
        Sigma = Sigma + np.eye(n) * 1e-11 #menfi eigenvlaue varsa bele edrik

    # TODO 1: Construct the (n+2)x(n+2) block matrix A using np.block (or vstack/hstack)
    # A = [
    #     [Sigma,            ones,             mu.reshape(-1, 1)],
    #     [ones.T,           np.zeros((1, 1)), np.zeros((1, 1))],
    #     [mu.reshape(1,-1), np.zeros((1, 1)), np.zeros((1, 1))]
    # ]
    mu_col = mu.reshape(-1, 1)  #in order to make it with matrix operation : it's actually f-g*lambda
    A = np.block([
        [Sigma,    ones,    mu_col],
        [ones.T,   np.zeros((1, 1)), np.zeros((1, 1))],
        [mu_col.T, np.zeros((1, 1)), np.zeros((1, 1))]
    ])

    # TODO 2: Construct RHS vector b = [0,...,0, 1, target_return]^T
    b = np.zeros(n+2)
    b[n]=1.0
    b[n+1]=target_return


    # TODO 3: Solve z = np.linalg.solve(A, b)
    z = np.linalg.solve(A,b)

    # TODO 4: Extract w, l1, l2 from z
    w = z[:n] #ne kimi olan hisse w di deye
    l1 = z[n]
    l2 = z[n+1]

    if not np.isclose(np.sum(w), 1.0, atol=1e-6):
        print(f"[Warning] Portfel çəkilərinin cəmi tam 1 deyil: {np.sum(w)}")
    return w, l1, l2


# ============================================================
# Part 2: Inequality constraints (No short selling, w >= 0)
# ============================================================
def solve_portfolio_kkt(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    tol: float = 1e-9,
    maxiter: int = 500
) -> np.ndarray:

    if target_return < np.min(mu) - 1e-7 or target_return > np.max(mu) + 1e-7:
        raise ValueError(f"Hedef gelr ({target_return}) mümkün degil"
                         f"serhedder: [{np.min(mu):.4f}, {np.max(mu):.4f}]")
    """
    Solve the no-short portfolio problem:

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return
                    w >= 0

    This cannot be solved with a single linear solve because of complementarity.
    Use scipy.optimize.minimize with method='SLSQP' and bounds.

    Returns:
        w : (n,) optimal weights
    """
    if minimize is None:
        raise RuntimeError(
            "SciPy is required for Part 2. Install scipy or use the provided environment."
        )

    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    target_return = float(target_return)

    # TODO 5: Define objective f(w) = 0.5 * w^T Sigma w
    def objective(w):
        return 0.5 * w.T @ Sigma @ w
    # (Optional) Also define gradient: grad(w) = Sigma @ w
    def grad(w):
        return Sigma @ w

    # TODO 6: Define equality constraints:
    #   (1) sum(w) = 1
    #   (2) mu^T w = target_return
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.dot(mu, w) - target_return}
    ]

    # TODO 7: Define bounds for w >= 0 (you may use upper bound 1.0 to help SLSQP)
    bounds = [(0, 1.0) for _ in range(n)]

    # TODO 8: Choose an initial guess w0 (e.g., uniform)
    w0 = np.ones(n)/n #baslanqiccin

    # TODO 9: Call minimize(..., method="SLSQP", bounds=bounds, constraints=constraints, jac=grad)
    # res = minimize(...)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds,
                   constraints=constraints, jac=grad)

    # TODO 10: Check res.success; if failed, raise RuntimeError(res.message)
    # TODO 11: Return res.x as a numpy array
    if not res.success:
        raise RuntimeError(res.message)
    w = res.x # for cheching before returning
    if not np.isclose(np.sum(w), 1.0, atol=1e-6):
        print(f"[Warning] Portfel çəkilərinin cəmi tam 1 deyil: {np.sum(w)}")
    return res.x


# ============================================================
# Efficient frontier + plotting (provided)
# ============================================================
def compute_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_returns: np.ndarray,
    solver_fn
) -> list[tuple[float, float]]:
    """
    Computes a (risk, return) list for a set of target returns using solver_fn.
    solver_fn signature: solver_fn(mu, Sigma, target_return) -> w
    """
    pts: list[tuple[float, float]] = []
    for R in target_returns:
        try:
            w = solver_fn(mu, Sigma, float(R))
            w = np.asarray(w, dtype=float).reshape(-1)
            risk = portfolio_risk_std(w, Sigma)
            ret = portfolio_return(w, mu)
            pts.append((risk, ret))
        except Exception as e:
            # Skip infeasible/failed targets but keep going
            print(f"[warn] Skipping R={float(R):.6f}: {e}")
            continue
    return pts


def plot_efficient_frontier(
    lagrange_pts: list[tuple[float, float]],
    kkt_pts: list[tuple[float, float]],
    filename: str = "efficient_frontier.png"
) -> None:
    """
    Plots the Efficient Frontier curves and saves to filename.
    lagrange_pts / kkt_pts are lists of (risk_std, return).
    """
    if len(lagrange_pts) == 0 or len(kkt_pts) == 0:
        print("[warn] Not enough points to plot.")
        return

    l_risk, l_ret = zip(*lagrange_pts)
    k_risk, k_ret = zip(*kkt_pts)

    plt.figure(figsize=(10, 6))
    plt.plot(l_risk, l_ret, linestyle="--", linewidth=2, label="Unconstrained (Lagrange)")
    plt.plot(k_risk, k_ret, marker="o", linewidth=2, label="No short selling (SLSQP)")

    plt.xlabel(r"Risk $\sigma(w)=\sqrt{w^T\Sigma w}$")
    plt.ylabel(r"Return $\mu^T w$")
    plt.title("Efficient Frontier: Impact of the No-Short-Selling Constraint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# ============================================================
# Demo / sanity run (you can modify)
# ============================================================
def example_problem() -> tuple[np.ndarray, np.ndarray]:
    """
    Example from the assignment handout (3 assets).
    Returns (mu, Sigma).
    """
    mu = np.array([0.05, 0.06, 0.08], dtype=float)
    Sigma = np.array(
        [
            [0.010, 0.002, 0.001],
            [0.002, 0.015, 0.005],
            [0.001, 0.005, 0.020],
        ],
        dtype=float,
    )
    return mu, Sigma


if __name__ == "__main__":
    mu, Sigma = example_problem()

    # Target returns grid (feasible for no-short case)
    target_returns = feasible_target_grid(mu, num=25)

    # Part 1 frontier (short selling allowed)
    lagrange_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=lambda mu_, Sigma_, R_: solve_portfolio_lagrange(mu_, Sigma_, R_)[0],  # returns (w, l1, l2)
    )

    # Part 2 frontier (no short selling)
    kkt_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=solve_portfolio_kkt,
    )

    plot_efficient_frontier(lagrange_pts, kkt_pts, filename="efficient_frontier.png") annotations

import numpy as np
import matplotlib.pyplot as plt

try:
    from scipy.optimize import minimize
except Exception:
    minimize = None  # Part 2 requires SciPy; we raise a helpful error in that function.


# ============================================================
# Helpers
# ============================================================
def portfolio_return(w: np.ndarray, mu: np.ndarray) -> float:
    """Expected return: mu^T w."""
    return float(np.dot(mu, w))


def portfolio_variance(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Variance (risk): w^T Sigma w."""
    return float(w.T @ Sigma @ w)


def portfolio_risk_std(w: np.ndarray, Sigma: np.ndarray) -> float:
    """Standard deviation risk: sqrt(w^T Sigma w)."""
    var = portfolio_variance(w, Sigma)
    return float(np.sqrt(max(var, 0.0)))


def feasible_target_grid(mu: np.ndarray, num: int = 25, eps: float = 1e-9) -> np.ndarray:
    """
    Under w >= 0 and sum(w)=1, achievable returns lie in [min(mu), max(mu)].
    We generate a grid in that range.
    """
    lo = float(np.min(mu)) + eps
    hi = float(np.max(mu)) - eps
    if hi <= lo:
        return np.array([float(np.mean(mu))])
    return np.linspace(lo, hi, num=num)


# ============================================================
# Part 1: Equality constraints (Lagrange multipliers)
# ============================================================
def solve_portfolio_lagrange(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float
) -> tuple[np.ndarray, float, float]:
    """
    Solve the equality-constrained Markowitz problem (short selling allowed):

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return

    Derivation leads to the block KKT system:

        [ Sigma   1    mu ] [  w  ]   [ 0 ]
        [ 1^T     0     0 ] [ l1  ] = [ 1 ]
        [ mu^T    0     0 ] [ l2  ]   [ R ]

    Returns:
        w  : (n,) optimal weights
        l1 : multiplier for the budget constraint
        l2 : multiplier for the return constraint (shadow price)
    """
    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size

    ones = np.ones((n, 1))

    # TODO 1: Construct the (n+2)x(n+2) block matrix A using np.block (or vstack/hstack)
    # A = [
    #     [Sigma,            ones,             mu.reshape(-1, 1)],
    #     [ones.T,           np.zeros((1, 1)), np.zeros((1, 1))],
    #     [mu.reshape(1,-1), np.zeros((1, 1)), np.zeros((1, 1))]
    # ] 
    mu_col = mu.reshape(-1, 1)  #in order to make it with matrix operation : it's actually f-g*lambda 
    A = np.block([
        [Sigma,    ones,    mu_col],
        [ones.T,   np.zeros((1, 1)), np.zeros((1, 1))],
        [mu_col.T, np.zeros((1, 1)), np.zeros((1, 1))]
    ])

    # TODO 2: Construct RHS vector b = [0,...,0, 1, target_return]^T
    b = np.zeros(n+2)
    b[n]=1.0
    b[n+1]=target_return


    # TODO 3: Solve z = np.linalg.solve(A, b)
    z = np.linalg.solve(A,b)

    # TODO 4: Extract w, l1, l2 from z
    w = z[:n] #ne kimi olan hisse w di deye 
    l1 = z[n]
    l2 = z[n+1]

    raise NotImplementedError("TODO: implement solve_portfolio_lagrange")


# ============================================================
# Part 2: Inequality constraints (No short selling, w >= 0)
# ============================================================
def solve_portfolio_kkt(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_return: float,
    tol: float = 1e-9,
    maxiter: int = 500
) -> np.ndarray:
    """
    Solve the no-short portfolio problem:

        minimize    0.5 * w^T Sigma w
        subject to  1^T w = 1
                    mu^T w = target_return
                    w >= 0

    This cannot be solved with a single linear solve because of complementarity.
    Use scipy.optimize.minimize with method='SLSQP' and bounds.

    Returns:
        w : (n,) optimal weights
    """
    if minimize is None:
        raise RuntimeError(
            "SciPy is required for Part 2. Install scipy or use the provided environment."
        )

    mu = np.asarray(mu, dtype=float).reshape(-1)
    Sigma = np.asarray(Sigma, dtype=float)
    n = mu.size
    target_return = float(target_return)

    # TODO 5: Define objective f(w) = 0.5 * w^T Sigma w
    # def objective(w): ...
    # (Optional) Also define gradient: grad(w) = Sigma @ w
    objective = 0.5 * w.T @ Sigma @ w 
    grad = Sigma @ w

    # TODO 6: Define equality constraints:
    #   (1) sum(w) = 1
    #   (2) mu^T w = target_return
    constraints = constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},
        {'type': 'eq', 'fun': lambda w: np.dot(mu, w) - target_return}
    ]

    # TODO 7: Define bounds for w >= 0 (you may use upper bound 1.0 to help SLSQP)
    bounds = [(0, 1.0) for _ in range(n)]

    # TODO 8: Choose an initial guess w0 (e.g., uniform)
    w0 = np.ones(n)/n #baslanqiccin

    # TODO 9: Call minimize(..., method="SLSQP", bounds=bounds, constraints=constraints, jac=grad)
    # res = minimize(...)
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, 
                   constraints=constraints, jac=grad)

    # TODO 10: Check res.success; if failed, raise RuntimeError(res.message)
    # TODO 11: Return res.x as a numpy array
    if not res.success:
        raise RuntimeError(res.message)
    return res.x


# ============================================================
# Efficient frontier + plotting (provided)
# ============================================================
def compute_frontier(
    mu: np.ndarray,
    Sigma: np.ndarray,
    target_returns: np.ndarray,
    solver_fn
) -> list[tuple[float, float]]:
    """
    Computes a (risk, return) list for a set of target returns using solver_fn.
    solver_fn signature: solver_fn(mu, Sigma, target_return) -> w
    """
    pts: list[tuple[float, float]] = []
    for R in target_returns:
        try:
            w = solver_fn(mu, Sigma, float(R))
            w = np.asarray(w, dtype=float).reshape(-1)
            risk = portfolio_risk_std(w, Sigma)
            ret = portfolio_return(w, mu)
            pts.append((risk, ret))
        except Exception as e:
            # Skip infeasible/failed targets but keep going
            print(f"[warn] Skipping R={float(R):.6f}: {e}")
            continue
    return pts


def plot_efficient_frontier(
    lagrange_pts: list[tuple[float, float]],
    kkt_pts: list[tuple[float, float]],
    filename: str = "efficient_frontier.png"
) -> None:
    """
    Plots the Efficient Frontier curves and saves to filename.
    lagrange_pts / kkt_pts are lists of (risk_std, return).
    """
    if len(lagrange_pts) == 0 or len(kkt_pts) == 0:
        print("[warn] Not enough points to plot.")
        return

    l_risk, l_ret = zip(*lagrange_pts)
    k_risk, k_ret = zip(*kkt_pts)

    plt.figure(figsize=(10, 6))
    plt.plot(l_risk, l_ret, linestyle="--", linewidth=2, label="Unconstrained (Lagrange)")
    plt.plot(k_risk, k_ret, marker="o", linewidth=2, label="No short selling (SLSQP)")

    plt.xlabel(r"Risk $\sigma(w)=\sqrt{w^T\Sigma w}$")
    plt.ylabel(r"Return $\mu^T w$")
    plt.title("Efficient Frontier: Impact of the No-Short-Selling Constraint")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()


# ============================================================
# Demo / sanity run (you can modify)
# ============================================================
def example_problem() -> tuple[np.ndarray, np.ndarray]:
    """
    Example from the assignment handout (3 assets).
    Returns (mu, Sigma).
    """
    mu = np.array([0.05, 0.06, 0.08], dtype=float)
    Sigma = np.array(
        [
            [0.010, 0.002, 0.001],
            [0.002, 0.015, 0.005],
            [0.001, 0.005, 0.020],
        ],
        dtype=float,
    )
    return mu, Sigma


if __name__ == "__main__":
    mu, Sigma = example_problem()

    # Target returns grid (feasible for no-short case)
    target_returns = feasible_target_grid(mu, num=25)

    # Part 1 frontier (short selling allowed)
    lagrange_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=lambda mu_, Sigma_, R_: solve_portfolio_lagrange(mu_, Sigma_, R_)[0],  # returns (w, l1, l2)
    )

    # Part 2 frontier (no short selling)
    kkt_pts = compute_frontier(
        mu,
        Sigma,
        target_returns,
        solver_fn=solve_portfolio_kkt,
    )

    plot_efficient_frontier(lagrange_pts, kkt_pts, filename="efficient_frontier.png")
