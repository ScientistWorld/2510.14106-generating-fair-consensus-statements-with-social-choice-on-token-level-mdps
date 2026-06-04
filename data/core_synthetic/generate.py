"""Synthetic token-tree environment from Section 4.2 of the paper."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np


@dataclass(frozen=True)
class CoreSyntheticInstance:
    branching: int
    depth: int
    num_agents: int
    embedding_dim: int
    rho: float
    seed: int
    leaves: np.ndarray
    utilities: np.ndarray


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp = np.exp(shifted)
    return exp / exp.sum()


def build_instance(
    rho: float,
    seed: int,
    branching: int = 3,
    depth: int = 4,
    num_agents: int = 5,
    embedding_dim: int = 8,
) -> CoreSyntheticInstance:
    """Build the B-ary token tree with prompted-policy analogues.

    The paper's empirical core test samples normalized token vectors v_{t,a},
    normalized agent vectors w_i, uses z(s)=sum_tau v_{tau,a_tau}, and defines
    pi_i(a|s)=softmax_a(rho * w_i^T(z + v_{t,a})).
    """

    rng = np.random.default_rng(seed)
    token_vectors = _normalize_rows(
        rng.normal(size=(depth, branching, embedding_dim))
    )
    agent_vectors = _normalize_rows(rng.normal(size=(num_agents, embedding_dim)))

    leaves = np.asarray(list(product(range(branching), repeat=depth)), dtype=np.int64)
    utilities = np.ones((num_agents, len(leaves)), dtype=np.float64)

    for leaf_index, path in enumerate(leaves):
        state = np.zeros(embedding_dim, dtype=np.float64)
        for t, action in enumerate(path):
            for i in range(num_agents):
                action_logits = np.array(
                    [
                        rho * float(agent_vectors[i] @ (state + token_vectors[t, a]))
                        for a in range(branching)
                    ],
                    dtype=np.float64,
                )
                probabilities = _softmax(action_logits)
                utilities[i, leaf_index] *= probabilities[action]
            state = state + token_vectors[t, action]

    return CoreSyntheticInstance(
        branching=branching,
        depth=depth,
        num_agents=num_agents,
        embedding_dim=embedding_dim,
        rho=rho,
        seed=seed,
        leaves=leaves,
        utilities=utilities,
    )
