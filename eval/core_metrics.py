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
    current = utilities @ distribution
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
                row[:num_leaves] = -utilities[agent]
                row[-1] = current[agent]
                a_ub.append(row)
                b_ub.append(0.0)

            result = linprog(
                c,
                A_ub=np.asarray(a_ub),
                b_ub=np.asarray(b_ub),
                A_eq=a_eq,
                b_eq=b_eq,
                bounds=[(0.0, None)] * num_leaves + [(0.0, None)],
                method="highs",
            )
            if not result.success:
                raise RuntimeError(f"Coalition LP failed for {coalition}: {result.message}")
            best_alpha = max(best_alpha, float(result.x[-1]))

    if abs(best_alpha - 1.0) <= tolerance:
        return 1.0
    return best_alpha
