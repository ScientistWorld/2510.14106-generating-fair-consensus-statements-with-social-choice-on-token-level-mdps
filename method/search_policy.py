"""Deterministic egalitarian search on finite token-level MDP trees."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class PathSearchResult:
    name: str
    path: tuple[int, ...]
    score: float


def prefix_log_utilities(
    leaves: np.ndarray,
    utilities: np.ndarray,
    prefix: tuple[int, ...],
    *,
    eps: float = 1e-300,
) -> np.ndarray:
    """Return each agent's additive log utility for a prefix.

    Leaf utilities are complete-path probabilities under each agent policy.
    For a fixed-depth tree, summing the probabilities of all leaves below a
    prefix recovers that prefix probability under the corresponding policy.
    """

    leaves = np.asarray(leaves, dtype=np.int64)
    utilities = np.asarray(utilities, dtype=np.float64)
    if len(prefix) == 0:
        return np.zeros(utilities.shape[0], dtype=np.float64)
    prefix_array = np.asarray(prefix, dtype=np.int64)
    mask = np.all(leaves[:, : len(prefix)] == prefix_array, axis=1)
    if not np.any(mask):
        raise ValueError(f"prefix {prefix!r} is not present in the leaf set")
    return np.log(np.maximum(utilities[:, mask].sum(axis=1), eps))


def egalitarian_score(leaves: np.ndarray, utilities: np.ndarray, path: tuple[int, ...]) -> float:
    """Paper's deterministic objective: min_i U_i^log(path)."""

    return float(np.min(prefix_log_utilities(leaves, utilities, path)))


def finite_lookahead_search(
    leaves: np.ndarray,
    utilities: np.ndarray,
    *,
    branching: int,
    depth: int,
    lookahead_depth: int,
) -> PathSearchResult:
    """Algorithm 2: rolling finite-lookahead maximization of egalitarian welfare."""

    prefix: tuple[int, ...] = ()
    while len(prefix) < depth:
        horizon = min(lookahead_depth, depth - len(prefix))
        candidates = []
        for suffix in product(range(branching), repeat=horizon):
            candidate = prefix + tuple(suffix)
            candidates.append((egalitarian_score(leaves, utilities, candidate), candidate))
        _, best_candidate = max(candidates, key=lambda item: (item[0], item[1]))
        prefix = prefix + (best_candidate[len(prefix)],)
    return PathSearchResult(
        name=f"finite_lookahead_d{lookahead_depth}",
        path=prefix,
        score=egalitarian_score(leaves, utilities, prefix),
    )


def beam_search(
    leaves: np.ndarray,
    utilities: np.ndarray,
    *,
    branching: int,
    depth: int,
    beam_width: int,
) -> PathSearchResult:
    """Algorithm 3: retain the top-width prefixes under egalitarian welfare."""

    beam: list[tuple[int, ...]] = [()]
    for _ in range(depth):
        candidates = [prefix + (action,) for prefix in beam for action in range(branching)]
        candidates.sort(
            key=lambda path: (egalitarian_score(leaves, utilities, path), path),
            reverse=True,
        )
        beam = candidates[:beam_width]
    best_path = max(beam, key=lambda path: (egalitarian_score(leaves, utilities, path), path))
    return PathSearchResult(
        name=f"beam_search_w{beam_width}",
        path=best_path,
        score=egalitarian_score(leaves, utilities, best_path),
    )


def exhaustive_egalitarian_optimum(
    leaves: np.ndarray,
    utilities: np.ndarray,
) -> PathSearchResult:
    """Exact finite-tree optimum used as a diagnostic upper bound."""

    leaf_paths = [tuple(map(int, leaf)) for leaf in np.asarray(leaves, dtype=np.int64)]
    best_path = max(leaf_paths, key=lambda path: (egalitarian_score(leaves, utilities, path), path))
    return PathSearchResult(
        name="exhaustive_egalitarian_oracle",
        path=best_path,
        score=egalitarian_score(leaves, utilities, best_path),
    )


def best_of_n_uniform(
    leaves: np.ndarray,
    utilities: np.ndarray,
    *,
    samples: int,
    seed: int,
) -> PathSearchResult:
    """Best-of-N baseline: sample complete paths uniformly and select by EW."""

    rng = np.random.default_rng(seed)
    leaves = np.asarray(leaves, dtype=np.int64)
    indices = rng.integers(0, leaves.shape[0], size=samples)
    sample_paths = [tuple(map(int, leaves[index])) for index in indices]
    best_path = max(sample_paths, key=lambda path: (egalitarian_score(leaves, utilities, path), path))
    return PathSearchResult(
        name=f"best_of_{samples}_uniform",
        path=best_path,
        score=egalitarian_score(leaves, utilities, best_path),
    )
