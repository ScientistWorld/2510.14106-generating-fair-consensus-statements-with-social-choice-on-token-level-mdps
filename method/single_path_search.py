"""Deterministic single-path search for egalitarian welfare on token trees.

The paper's single-statement objective maximizes the worst agent's additive
log utility over one completed path.  This module works with a finite token
tree represented by its leaves and per-agent leaf log utilities.  Bounded
partial-prefix searches rank prefixes by aggregating the complete leaves still
reachable from that prefix, then return and score one completed leaf.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Literal

import numpy as np


PrefixAggregation = Literal["mean", "best", "worst"]


@dataclass(frozen=True)
class PathSelectionResult:
    """Selected deterministic path and welfare diagnostics."""

    name: str
    path: tuple[int, ...]
    leaf_index: int
    agent_log_utilities: tuple[float, ...]
    egalitarian_welfare: float
    utilitarian_welfare: float
    metadata: dict[str, int | float | str]


def leaf_log_utilities_from_prob_utilities(
    utilities: np.ndarray,
    *,
    floor: float = 1e-300,
) -> np.ndarray:
    """Convert non-negative multiplicative leaf utilities to log utilities."""

    utilities = np.asarray(utilities, dtype=np.float64)
    if utilities.ndim != 2:
        raise ValueError("utilities must have shape (num_agents, num_leaves)")
    if np.any(utilities < 0):
        raise ValueError("probability utilities must be non-negative")
    return np.log(np.maximum(utilities, floor))


def egalitarian_welfare(agent_log_utilities: np.ndarray) -> float:
    """Worst-agent additive log utility for one path."""

    values = np.asarray(agent_log_utilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("agent_log_utilities must be one-dimensional")
    return float(np.min(values))


def utilitarian_welfare(agent_log_utilities: np.ndarray) -> float:
    """Sum of additive log utilities for one path."""

    values = np.asarray(agent_log_utilities, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("agent_log_utilities must be one-dimensional")
    return float(np.sum(values))


def select_egalitarian_path(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    *,
    name: str = "egalitarian_exhaustive",
) -> PathSelectionResult:
    """Select the complete leaf with maximum egalitarian welfare."""

    leaves, leaf_log_utilities = _validate_inputs(leaves, leaf_log_utilities)
    scores = np.min(leaf_log_utilities, axis=0)
    return _result_for_leaf(name, _stable_argmax(scores, leaves), leaves, leaf_log_utilities)


def select_utilitarian_path(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    *,
    name: str = "utilitarian",
) -> PathSelectionResult:
    """Select the complete leaf maximizing total agent log utility."""

    leaves, leaf_log_utilities = _validate_inputs(leaves, leaf_log_utilities)
    scores = np.sum(leaf_log_utilities, axis=0)
    return _result_for_leaf(name, _stable_argmax(scores, leaves), leaves, leaf_log_utilities)


def select_greedy_path(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    branching: int,
    *,
    aggregation: PrefixAggregation = "mean",
    name: str = "greedy_egalitarian",
) -> PathSelectionResult:
    """Greedily choose the next token with highest prefix egalitarian value."""

    leaves, leaf_log_utilities = _validate_inputs(leaves, leaf_log_utilities)
    _validate_branching(branching)
    prefix: tuple[int, ...] = ()
    for _ in range(leaves.shape[1]):
        candidates = [(action,) for action in range(branching)]
        prefix = _choose_best_extension(
            prefix,
            candidates,
            leaves,
            leaf_log_utilities,
            aggregation,
        )
    return _result_for_prefix(
        name,
        prefix,
        leaves,
        leaf_log_utilities,
        {"aggregation": aggregation},
    )


def select_finite_lookahead_path(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    branching: int,
    *,
    lookahead: int = 2,
    aggregation: PrefixAggregation = "mean",
    name: str = "finite_lookahead_egalitarian",
) -> PathSelectionResult:
    """Rolling-horizon egalitarian search with a bounded token lookahead."""

    leaves, leaf_log_utilities = _validate_inputs(leaves, leaf_log_utilities)
    _validate_branching(branching)
    if lookahead < 1:
        raise ValueError("lookahead must be at least 1")

    depth = leaves.shape[1]
    prefix: tuple[int, ...] = ()
    while len(prefix) < depth:
        horizon = min(lookahead, depth - len(prefix))
        extensions = product(range(branching), repeat=horizon)
        best_extension = _choose_best_extension(
            prefix,
            extensions,
            leaves,
            leaf_log_utilities,
            aggregation,
        )
        prefix = prefix + (best_extension[len(prefix)],)

    return _result_for_prefix(
        name,
        prefix,
        leaves,
        leaf_log_utilities,
        {"lookahead": lookahead, "aggregation": aggregation},
    )


def select_beam_search_path(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    branching: int,
    *,
    beam_width: int = 4,
    aggregation: PrefixAggregation = "mean",
    name: str = "beam_search_egalitarian",
) -> PathSelectionResult:
    """Beam search that keeps the top prefixes by egalitarian prefix value."""

    leaves, leaf_log_utilities = _validate_inputs(leaves, leaf_log_utilities)
    _validate_branching(branching)
    if beam_width < 1:
        raise ValueError("beam_width must be at least 1")

    beam: list[tuple[int, ...]] = [()]
    for _ in range(leaves.shape[1]):
        candidates = [
            prefix + (action,)
            for prefix in beam
            for action in range(branching)
            if _has_matching_leaf(prefix + (action,), leaves)
        ]
        ranked = sorted(
            candidates,
            key=lambda prefix: (
                -_prefix_egalitarian_value(prefix, leaves, leaf_log_utilities, aggregation),
                prefix,
            ),
        )
        beam = ranked[:beam_width]

    final_scores = np.asarray(
        [
            _leaf_score_for_prefix(prefix, leaves, leaf_log_utilities)
            for prefix in beam
        ],
        dtype=np.float64,
    )
    best_prefix = beam[_stable_argmax_prefix_scores(final_scores, beam)]
    return _result_for_prefix(
        name,
        best_prefix,
        leaves,
        leaf_log_utilities,
        {"beam_width": beam_width, "aggregation": aggregation},
    )


def run_all_selectors(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    branching: int,
    *,
    lookahead: int = 2,
    beam_width: int = 4,
    aggregation: PrefixAggregation = "mean",
) -> list[PathSelectionResult]:
    """Run the bounded deterministic selectors used in the secondary experiment."""

    return [
        select_egalitarian_path(leaves, leaf_log_utilities),
        select_greedy_path(
            leaves,
            leaf_log_utilities,
            branching,
            aggregation=aggregation,
        ),
        select_finite_lookahead_path(
            leaves,
            leaf_log_utilities,
            branching,
            lookahead=lookahead,
            aggregation=aggregation,
        ),
        select_beam_search_path(
            leaves,
            leaf_log_utilities,
            branching,
            beam_width=beam_width,
            aggregation=aggregation,
        ),
        select_utilitarian_path(leaves, leaf_log_utilities),
    ]


def _validate_inputs(
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    leaves = np.asarray(leaves, dtype=np.int64)
    leaf_log_utilities = np.asarray(leaf_log_utilities, dtype=np.float64)
    if leaves.ndim != 2:
        raise ValueError("leaves must have shape (num_leaves, depth)")
    if leaf_log_utilities.ndim != 2:
        raise ValueError("leaf_log_utilities must have shape (num_agents, num_leaves)")
    if leaf_log_utilities.shape[1] != leaves.shape[0]:
        raise ValueError("leaf_log_utilities and leaves disagree on num_leaves")
    if not np.all(np.isfinite(leaf_log_utilities)):
        raise ValueError("leaf_log_utilities must be finite")
    return leaves, leaf_log_utilities


def _validate_branching(branching: int) -> None:
    if branching < 1:
        raise ValueError("branching must be at least 1")


def _stable_argmax(scores: np.ndarray, leaves: np.ndarray) -> int:
    best_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))
    return min((tuple(leaves[index]), int(index)) for index in tied)[1]


def _stable_argmax_prefix_scores(
    scores: np.ndarray,
    prefixes: list[tuple[int, ...]],
) -> int:
    best_score = float(np.max(scores))
    tied = np.flatnonzero(np.isclose(scores, best_score, rtol=0.0, atol=1e-12))
    return min((prefixes[index], int(index)) for index in tied)[1]


def _matching_leaf_indices(prefix: tuple[int, ...], leaves: np.ndarray) -> np.ndarray:
    if len(prefix) == 0:
        return np.arange(leaves.shape[0])
    return np.flatnonzero(np.all(leaves[:, : len(prefix)] == np.asarray(prefix), axis=1))


def _has_matching_leaf(prefix: tuple[int, ...], leaves: np.ndarray) -> bool:
    return bool(len(_matching_leaf_indices(prefix, leaves)))


def _prefix_egalitarian_value(
    prefix: tuple[int, ...],
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    aggregation: PrefixAggregation,
) -> float:
    indices = _matching_leaf_indices(prefix, leaves)
    if len(indices) == 0:
        return -np.inf
    descendant_utilities = leaf_log_utilities[:, indices]
    if aggregation == "mean":
        agent_values = np.mean(descendant_utilities, axis=1)
    elif aggregation == "best":
        agent_values = descendant_utilities[:, _stable_argmax(np.min(descendant_utilities, axis=0), leaves[indices])]
    elif aggregation == "worst":
        agent_values = np.min(descendant_utilities, axis=1)
    else:
        raise ValueError(f"unknown aggregation: {aggregation}")
    return egalitarian_welfare(agent_values)


def _choose_best_extension(
    prefix: tuple[int, ...],
    extensions,
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    aggregation: PrefixAggregation,
) -> tuple[int, ...]:
    candidates = [prefix + tuple(extension) for extension in extensions]
    candidates = [candidate for candidate in candidates if _has_matching_leaf(candidate, leaves)]
    if not candidates:
        raise RuntimeError(f"no valid extensions from prefix {prefix}")
    return min(
        candidates,
        key=lambda candidate: (
            -_prefix_egalitarian_value(candidate, leaves, leaf_log_utilities, aggregation),
            candidate,
        ),
    )


def _leaf_score_for_prefix(
    prefix: tuple[int, ...],
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
) -> float:
    indices = _matching_leaf_indices(prefix, leaves)
    if len(indices) != 1:
        raise RuntimeError(f"prefix {prefix} does not identify one complete leaf")
    return egalitarian_welfare(leaf_log_utilities[:, int(indices[0])])


def _result_for_prefix(
    name: str,
    prefix: tuple[int, ...],
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    metadata: dict[str, int | float | str],
) -> PathSelectionResult:
    indices = _matching_leaf_indices(prefix, leaves)
    if len(indices) != 1:
        raise RuntimeError(f"prefix {prefix} does not identify one complete leaf")
    return _result_for_leaf(name, int(indices[0]), leaves, leaf_log_utilities, metadata)


def _result_for_leaf(
    name: str,
    leaf_index: int,
    leaves: np.ndarray,
    leaf_log_utilities: np.ndarray,
    metadata: dict[str, int | float | str] | None = None,
) -> PathSelectionResult:
    agent_values = leaf_log_utilities[:, leaf_index]
    return PathSelectionResult(
        name=name,
        path=tuple(int(token) for token in leaves[leaf_index]),
        leaf_index=int(leaf_index),
        agent_log_utilities=tuple(float(value) for value in agent_values),
        egalitarian_welfare=egalitarian_welfare(agent_values),
        utilitarian_welfare=utilitarian_welfare(agent_values),
        metadata=dict(metadata or {}),
    )
