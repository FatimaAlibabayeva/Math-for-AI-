"""
Math4AI: Probability & Statistics — Assignment 4 (Student Starter)
Bayesian Inference & Networks

You must complete the TODOs. Do NOT use probabilistic programming libraries.
Allowed: numpy, matplotlib, Python stdlib.

Deliverables:
- bayesian_update.png
- Printouts for Task 4.2 and Task 4.3
"""

from __future__ import annotations

import math
import itertools
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt


# Output directory (save outputs next to this file)
OUT_DIR = Path(__file__).resolve().parent

# ================================================================
# Task 4.1: Beta-Binomial Model
# ================================================================

def _log_beta(a: float, b: float) -> float:
    """log(Beta(a,b)) using log-gamma for numerical stability."""
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def beta_pdf(theta: np.ndarray, a: float, b: float) -> np.ndarray:
    """
    Compute Beta(a,b) PDF on an array of theta values in (0,1).
    Uses a numerically stable log-space computation.
    """
    theta = np.asarray(theta, dtype=float)
    eps = 1e-12
    t = np.clip(theta, eps, 1 - eps)
    logp = (a - 1) * np.log(t) + (b - 1) * np.log(1 - t) - _log_beta(a, b)
    return np.exp(logp)


@dataclass
class BetaBinomialModel:
    """
    Maintains a Beta(alpha, beta) belief about theta = P(Heads).
    """
    alpha: float
    beta: float

    def update(self, heads: int, tails: int) -> None:
        """
        Perform the conjugate Bayesian update.

        TODO 1:
          Update self.alpha and self.beta given heads and tails.
          Posterior: Beta(alpha + heads, beta + tails).
        """
        self.alpha+=heads
        self.beta+=tails

    def mean(self) -> float:
        """Posterior mean E[theta] = alpha / (alpha + beta)."""
        return float(self.alpha / (self.alpha + self.beta))

    def map(self) -> float:
        """
        MAP estimate for Beta(alpha,beta):
            (alpha-1)/(alpha+beta-2)  when alpha>1 and beta>1.
        If not defined (alpha<=1 or beta<=1), return the mean.
        """
        if self.alpha > 1 and self.beta > 1:
            return float((self.alpha - 1) / (self.alpha + self.beta - 2))
        return self.mean()


def simulate_coin_flips(theta_true: float, n_flips: int, rng: np.random.Generator) -> Tuple[int, int]:
    """Return (heads, tails) from n_flips Bernoulli trials."""
    flips = rng.random(n_flips) < theta_true
    heads = int(np.sum(flips))
    tails = int(n_flips - heads)
    return heads, tails


def plot_belief_evolution(thetas: np.ndarray, snapshots: Dict[str, Tuple[float, float]], filename: str) -> None:
    """
    Plot PDFs for (alpha,beta) snapshots. 'snapshots' maps label -> (alpha,beta).
    """
    plt.figure(figsize=(10, 6))
    for label, (a, b) in snapshots.items():
        plt.plot(thetas, beta_pdf(thetas, a, b), label=label)
    plt.xlabel(r"$\theta$ (Probability of Heads)")
    plt.ylabel("Density")
    plt.title("Bayesian Update: Beta Prior -> Beta Posterior")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.show()
    plt.close()



# ================================================================
# Task 4.2: MLE vs MAP
# ================================================================

def compute_estimates(heads: int, tails: int, alpha_prior: float, beta_prior: float) -> Dict[str, float]:
    """
    TODO 2:
      Return a dictionary with keys:
        - 'mle'
        - 'map'
      using:
        mle = heads / (heads + tails)
        map = (alpha_prior + heads - 1) / (alpha_prior + beta_prior + heads + tails - 2)
      If the MAP formula is not valid (denominator<=0 or numerator out of range),
      fall back to posterior mean:
        mean = (alpha_prior + heads) / (alpha_prior + beta_prior + heads + tails)
    """
    mle=heads/(heads+tails) #for mle

    alpha_post=alpha_prior+heads  #posteriorlarin hesablanmasi 
    beta_post=beta_prior+tails
    #mexrecimiz 2den boyuk olmalidiki biz deye bilekki max varda
    if alpha_post>1 and beta_post>1:
        map_est=(alpha_post-1)/(beta_post+alpha_post-2)
    else:
        map_est=alpha_post/(alpha_post+beta_post) #sert ondemese meane qayidiriq
    return {'mle':mle, 'map':map_est}


# ================================================================
# Task 4.3: Bayesian Network (Wet Grass)
# ================================================================

Bool = bool
Assignment = Tuple[Bool, Bool, Bool]  # (R, S, W)


class SimpleBayesNet:
    """
    A tiny Bayes Net for Wet Grass with variables:
      R (Rain), S (Sprinkler), W (WetGrass)

    You will compute the full joint distribution over 2^3 assignments
    and answer queries by marginalization (summing over hidden vars).
    """

    def __init__(self):
        # Priors / CPTs (given by the assignment)
        self.p_rain_true = 0.2

        # P(S=True | R)
        self.p_s_true_given_r = {True: 0.01, False: 0.4}

        # P(W=True | S, R)
        self.p_w_true_given_s_r = {
            (True, True): 0.99,
            (True, False): 0.90,
            (False, True): 0.80,
            (False, False): 0.0,
        }

        self.joint: Dict[Assignment, float] = {}

    def p_r(self, r: Bool) -> float:
        return self.p_rain_true if r else 1.0 - self.p_rain_true

    def p_s_given_r(self, s: Bool, r: Bool) -> float:
        p_true = self.p_s_true_given_r[r]
        return p_true if s else 1.0 - p_true

    def p_w_given_s_r(self, w: Bool, s: Bool, r: Bool) -> float:
        p_true = self.p_w_true_given_s_r[(s, r)]
        return p_true if w else 1.0 - p_true

    def compute_joint_distribution(self) -> Dict[Assignment, float]:
        """
        TODO 3:
          Fill self.joint with entries for all (R,S,W) assignments:
            P(R,S,W) = P(R) * P(S|R) * P(W|S,R)
          Return the dict.

        Recommended:
          After computing, assert probabilities sum to ~1.
        """
        for r,s,w in itertools.product([True,False],repeat=3): # 3 defe r,s,w ucun ayri ayri true false olaraq dovre girir 
            p=self.p_r(r)*self.p_s_given_r(s,r)*self.p_w_given_s_r(w,s,r)
            self.joint[(r,s,w)]=p

        assert math.isclose(sum(self.joint.values()),1.0) #problarin cemi 1 olmalidi
        return self.joint

    def query(self, query_var: str, evidence: Dict[str, Bool]) -> float:
        """
        Compute P(query_var=True | evidence).

        query_var is one of {'R','S','W'}.
        evidence maps variable name -> bool, e.g. {'W': True}.

        TODO 4:
          Implement marginalization using the full joint table:
            P(Q=True | E) = sum_{assignments consistent with Q=True and E} P(assign)
                            -------------------------------------------------------
                            sum_{assignments consistent with E} P(assign)
        """
        numerator=0.0
        denominator=0.0
        var_map={'R':0,'S':1,'W':2} #deyisenlerin yerini bilmek ucun
        q_idx=var_map[query_var] #meselen R sorusulsa 0 cixir
        for assignment,prob in self.joint.items():
            is_const=True
            for var_name, value in evidence.items():
                if assignment[var_map[var_name]] !=value:
                    is_const=False
                    break
            if is_const:
                denominator+=prob
                if assignment[q_idx] is True:
                    numerator+=prob
        return numerator/denominator if denominator>0 else 0.0


# ================================================================
# Main (runs all tasks and produces required outputs)
# ================================================================

def main() -> None:
    # ----------------------------
    # Task 4.1: belief evolution
    # ----------------------------
    rng = np.random.default_rng(0)
    theta_true = 0.8
    model = BetaBinomialModel(alpha=1.0, beta=1.0)

    thetas = np.linspace(0.001, 0.999, 600)
    snapshots = {"Prior (0 flips)": (model.alpha, model.beta)}

    total_heads = 0
    total_tails = 0

    for step in range(1, 11):  # 10 batches x 50 = 500 flips
        heads, tails = simulate_coin_flips(theta_true, 50, rng)
        total_heads += heads
        total_tails += tails

        # TODO 5: Call model.update(heads, tails)
        # model.update(heads, tails)
        model.update(heads,tails)
        snapshots[f"Posterior ({50*step} flips)"] = (model.alpha, model.beta)

    plot_belief_evolution(thetas, snapshots, filename=str(OUT_DIR / "bayesian_update.png"))
    print(f"Saved plot: {OUT_DIR / 'bayesian_update.png'}")
    print(f"Final posterior alpha={model.alpha:.1f}, beta={model.beta:.1f}, mean={model.mean():.4f}, MAP={model.map():.4f}")
    print(f"Total heads={total_heads}, total tails={total_tails}")

    # ----------------------------
    # Task 4.2: MLE vs MAP
    # ----------------------------
    est = compute_estimates(heads=5, tails=0, alpha_prior=10.0, beta_prior=10.0)
    print("\nTask 4.2 estimates (5H,0T, prior Beta(10,10)):")
    print(est)

    # ----------------------------
    # Task 4.3: Bayes net query
    # ----------------------------
    bn = SimpleBayesNet()
    bn.compute_joint_distribution()
    p_r_given_w = bn.query("R", {"W": True})
    print("\nTask 4.3 query:")
    print(f"P(R=True | W=True) = {p_r_given_w:.6f}")

    # Top-3 joint states
    top3 = sorted(bn.joint.items(), key=lambda kv: kv[1], reverse=True)[:3]
    print("\nTop-3 joint assignments (R,S,W):")
    for (r,s,w), p in top3:
        print((r, s, w), f"{p:.6f}")


if __name__ == "__main__":
    main()
