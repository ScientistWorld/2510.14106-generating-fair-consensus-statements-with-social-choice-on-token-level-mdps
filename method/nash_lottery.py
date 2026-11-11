"""Nash-welfare lottery over complete statements and the induced stochastic token policy.

Algorithm 1 of the paper: build u_i(j) = U_i^prob(X_j), solve
    p* = argmax_{p in simplex} sum_i log(u_i^T p)
with Frank-Wolfe (coordinate linear oracle, exact line search; away steps so the duality gap
can be driven to the tolerance), then induce pi*(a|s) = P*(C(s,a)) / P*(C(s)) (Definition 2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass
class NashResult:
    distribution: np.ndarray
    gap: float
    iterations: int


def leaf_logutils(logp: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """log U_i^prob(X) = sum_t log pi_i(a_t | s_{t-1}) for every agent and leaf, shape (n, m)."""

    leaves = np.asarray(leaves, dtype=np.int64)
    out = np.zeros((logp.shape[0], leaves.shape[0]))
    for t in range(logp.shape[1]):
        out += logp[:, t, leaves[:, t]]
    return out


def all_leaves(branching: int, depth: int) -> np.ndarray:
    return np.asarray(list(product(range(branching), repeat=depth)), dtype=np.int64)


def _line_search(a: np.ndarray, d: np.ndarray, gmax: float) -> float:
    """Exact maximiser of the concave phi(g) = sum_i log(a_i + g d_i) on [0, gmax]."""

    if float(np.sum(d / (a + gmax * (1.0 - 1e-15) * d))) >= 0.0:
        return gmax
    lo, hi, g = 0.0, gmax, 0.0
    for _ in range(80):
        r = d / (a + g * d)
        f1 = float(r.sum())
        if f1 > 0:
            lo = g
        else:
            hi = g
        f2 = -float((r * r).sum())
        newton = g - f1 / f2 if f2 < 0 else 0.5 * (lo + hi)
        g_new = newton if lo < newton < hi else 0.5 * (lo + hi)
        if abs(f1) <= 1e-12 * float(np.abs(r).sum()) or abs(g_new - g) <= 1e-15 + 1e-12 * abs(g_new) \
                or hi - lo <= 1e-15 * max(1.0, gmax):
            return g_new
        g = g_new
    return g


def solve_nash_welfare(
    utilities: np.ndarray,
    *,
    tol: float = 1e-10,
    max_iter: int = 50_000,
    init: np.ndarray | None = None,
) -> NashResult:
    """Maximise sum_i log(u_i^T p) over the simplex by away-step Frank-Wolfe.

    ``utilities`` is (agents x leaves) and may be rescaled per agent (this only shifts the
    objective). Stops when the Frank-Wolfe duality gap max_j g_j - g^T p <= tol, where
    g = grad = sum_i u_i / (u_i^T p); since g^T p = n, the gap also bounds the core:
    no coalition can improve by more than a factor 1 + gap / n.
    """

    v = np.asarray(utilities, dtype=np.float64)
    v = v / v.max(axis=1, keepdims=True)
    n, m = v.shape
    vt = np.ascontiguousarray(v.T)
    if init is None and m > 2000:
        # Large candidate sets: start at the best single statement (a simplex vertex), the
        # usual Frank-Wolfe start, so the iterate stays sparse.
        p = np.zeros(m)
        p[int(np.argmax(np.log(v).sum(axis=0)))] = 1.0
    elif init is None:
        p = np.full(m, 1.0 / m)
    else:
        p = np.maximum(np.asarray(init, dtype=np.float64), 0.0)
        p /= p.sum()
    a = v @ p
    gap = math.inf
    it = 0
    for it in range(1, max_iter + 1):
        if it % 200 == 0:
            a = v @ p
        g = vt @ (1.0 / a)
        gp = float(g @ p)
        j_fw = int(np.argmax(g))
        gap = float(g[j_fw] - gp)
        if gap <= tol:
            break
        active = np.flatnonzero(p > 0)
        j_aw = int(active[np.argmin(g[active])])
        if gap >= gp - float(g[j_aw]):
            d = v[:, j_fw] - a
            gamma = _line_search(a, d, 1.0)
            p *= 1.0 - gamma
            p[j_fw] += gamma
        else:
            pa = float(p[j_aw])
            gmax = pa / (1.0 - pa)
            d = a - v[:, j_aw]
            gamma = _line_search(a, d, gmax)
            p *= 1.0 + gamma
            p[j_aw] -= gamma
            if gamma >= gmax * (1.0 - 1e-12):
                p[j_aw] = 0.0
        np.maximum(p, 0.0, out=p)
        p /= p.sum()
        a = a + gamma * d
    return NashResult(distribution=p, gap=gap, iterations=it)


def induce_prefix_policy(leaves: np.ndarray, distribution: np.ndarray, branching: int) -> dict:
    """pi*(a|s) = P*(C(s,a)) / P*(C(s)) for every prefix s with P*(C(s)) > 0 (Definition 2).

    Prefix masses are accumulated from the leaves in one pass; prefixes with zero mass are
    unreachable under pi* (their action probabilities are 0 by Definition 2) and are omitted.
    """

    leaves = np.asarray(leaves, dtype=np.int64)
    probs = np.asarray(distribution, dtype=np.float64)
    depth = leaves.shape[1]
    mass: dict[tuple, float] = {}
    for leaf, q in zip(map(tuple, leaves.tolist()), probs.tolist()):
        if q <= 0.0:
            continue
        for t in range(depth + 1):
            key = leaf[:t]
            mass[key] = mass.get(key, 0.0) + q
    policy = {}
    for prefix, total in mass.items():
        if len(prefix) == depth or total <= 0.0:
            continue
        row = [mass.get(prefix + (a,), 0.0) / total for a in range(branching)]
        s = sum(row)
        policy[",".join(map(str, prefix))] = [x / s for x in row]
    return policy
