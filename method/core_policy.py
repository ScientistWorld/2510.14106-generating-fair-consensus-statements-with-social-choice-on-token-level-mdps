"""Nash-welfare stochastic policy for token-level MDP leaves."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True)
class NashPolicyResult:
    distribution: np.ndarray
    nash_welfare: float
    success: bool
    message: str


def solve_nash_welfare_distribution(
    utilities: np.ndarray,
    *,
    max_iter: int = 2000,
    tolerance: float = 1e-10,
) -> NashPolicyResult:
    """Solve max_p sum_i log(u_i^T p) over the probability simplex."""

    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.ndim != 2:
        raise ValueError("utilities must have shape (num_agents, num_leaves)")
    if np.any(utilities < 0):
        raise ValueError("Nash-welfare utilities must be non-negative")

    num_leaves = utilities.shape[1]
    eps = 1e-300

    def objective(p: np.ndarray) -> float:
        expected = utilities @ p
        return -float(np.sum(np.log(np.maximum(expected, eps))))

    def gradient(p: np.ndarray) -> np.ndarray:
        expected = np.maximum(utilities @ p, eps)
        return -np.sum(utilities / expected[:, None], axis=0)

    constraints = ({"type": "eq", "fun": lambda p: float(np.sum(p) - 1.0)},)
    bounds = [(0.0, 1.0) for _ in range(num_leaves)]
    initial = np.full(num_leaves, 1.0 / num_leaves, dtype=np.float64)
    result = minimize(
        objective,
        initial,
        jac=gradient,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"maxiter": max_iter, "ftol": tolerance, "disp": False},
    )

    p = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    total = float(p.sum())
    if total <= 0:
        raise RuntimeError(f"Nash optimization returned zero mass: {result.message}")
    p = p / total
    expected = np.maximum(utilities @ p, eps)
    return NashPolicyResult(
        distribution=p,
        nash_welfare=float(np.prod(expected)),
        success=bool(result.success),
        message=str(result.message),
    )


def induce_prefix_policy(
    leaves: np.ndarray,
    distribution: np.ndarray,
    branching: int,
) -> dict[str, list[float]]:
    """Convert leaf lottery p* into prefix conditional masses pi*(a|s)."""

    leaves = np.asarray(leaves, dtype=np.int64)
    distribution = np.asarray(distribution, dtype=np.float64)
    depth = leaves.shape[1]
    prefix_policy: dict[str, list[float]] = {}

    for prefix_len in range(depth):
        prefixes = {tuple(row[:prefix_len]) for row in leaves}
        for prefix in prefixes:
            prefix_mask = np.all(leaves[:, :prefix_len] == np.asarray(prefix), axis=1)
            prefix_mass = float(distribution[prefix_mask].sum())
            probabilities = []
            for action in range(branching):
                action_mask = prefix_mask & (leaves[:, prefix_len] == action)
                action_mass = float(distribution[action_mask].sum())
                probabilities.append(action_mass / prefix_mass if prefix_mass > 0 else 0.0)
            prefix_policy[",".join(map(str, prefix))] = probabilities

    return prefix_policy
