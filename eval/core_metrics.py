"""Method-agnostic metrics for the empirical ex-ante-core test."""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.optimize import linprog


def max_coalition_improvement(
    utilities: np.ndarray,
    distribution: np.ndarray,
    *,
    tolerance: float = 1e-8,
) -> float:
    """Return alpha*, where alpha > 1 means some coalition blocks a policy.

    For each nonempty coalition S of size r, solve the LP from Section 4.2:
    maximize alpha subject to a probability-mass budget r/n and
    U_i^T p' >= alpha * U_i^T p for every i in S.
    """

    utilities = np.asarray(utilities, dtype=np.float64)
    distribution = np.asarray(distribution, dtype=np.float64)
    num_agents, num_leaves = utilities.shape
    if not np.all(np.isfinite(utilities)):
        raise ValueError("utilities must be finite")
    if not np.all(np.isfinite(distribution)):
        raise ValueError("distribution must be finite")
    if np.any(distribution < -tolerance):
        raise ValueError("distribution must be non-negative")

    distribution = np.maximum(distribution, 0.0)
    total_mass = float(distribution.sum())
    if total_mass <= tolerance:
        raise ValueError("distribution has no probability mass")
    distribution = distribution / total_mass

    current = utilities @ distribution
    current_safe = np.maximum(current, tolerance)
    best_alpha = 0.0

    for r in range(1, num_agents + 1):
        budget = r / num_agents
        for coalition in combinations(range(num_agents), r):
            # Variables are q_0...q_{m-1}, alpha. linprog minimizes c^T x.
            c = np.zeros(num_leaves + 1, dtype=np.float64)
            c[-1] = -1.0

            a_eq = np.zeros((1, num_leaves + 1), dtype=np.float64)
            a_eq[0, :num_leaves] = 1.0
            b_eq = np.array([budget], dtype=np.float64)

            a_ub = []
            b_ub = []
            for agent in coalition:
                row = np.zeros(num_leaves + 1, dtype=np.float64)
                # Scale by the current utility so HiGHS sees alpha with a
                # well-conditioned coefficient: alpha <= U_i(q) / U_i(p).
                row[:num_leaves] = -(utilities[agent] / current_safe[agent])
                row[-1] = 1.0
                a_ub.append(row)
                b_ub.append(0.0)

            result = linprog(
                c,
                A_ub=np.asarray(a_ub),
                b_ub=np.asarray(b_ub),
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(0.0, None)] * num_leaves + [(0.0, 1.0 / tolerance)],
                method="highs",
            )
            if not result.success:
                raise RuntimeError(f"Coalition LP failed for {coalition}: {result.message}")
            best_alpha = max(best_alpha, float(result.x[-1]))

    if abs(best_alpha - 1.0) <= tolerance:
        return 1.0
    return best_alpha
