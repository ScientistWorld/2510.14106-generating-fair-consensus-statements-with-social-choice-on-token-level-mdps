"""Held-out test instances. Private to eval/test/: nothing outside this directory can regenerate them.

The instance families are the public ones (same generating process as the training grid), but
the seeds below are used only here, and the generator is a private copy so that editing public
data code cannot change the test ground truth.
"""

from __future__ import annotations

import numpy as np

EMPIRICAL = "empirical_core_test"
DEEP = "deep_tree_core_test"

SETTINGS = {
    EMPIRICAL: {
        "branching": 3,
        "depth": 4,
        "num_agents": 6,
        "embedding_dim": 8,
        "rhos": [float(r) for r in np.linspace(0.6, 5.0, 20)],
    },
    DEEP: {
        "branching": 3,
        "depth": 24,
        "num_agents": 6,
        "embedding_dim": 8,
        "rhos": [0.1, 0.2, 0.4, 0.8],
    },
}

_SEEDS = {
    EMPIRICAL: [731954, 284117, 905362, 467203, 118846],
    DEEP: [640271, 352908, 819433, 207615, 974382, 561049, 438726, 95813, 713364, 286590],
}


def _vectors(seed: int, branching: int, depth: int, embedding_dim: int, num_agents: int):
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(depth, branching, embedding_dim))
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-12
    w = rng.normal(size=(num_agents, embedding_dim))
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    return v, w


def _token_logprobs(v: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    logits = rho * np.einsum("nd,lbd->nlb", w, v)
    m = logits.max(axis=2, keepdims=True)
    s = logits - m
    return s - np.log(np.exp(s).sum(axis=2, keepdims=True))


def test_instances() -> list[dict]:
    instances = []
    for experiment in (EMPIRICAL, DEEP):
        cfg = SETTINGS[experiment]
        counter = 0
        for seed in _SEEDS[experiment]:
            v, w = _vectors(seed, cfg["branching"], cfg["depth"], cfg["embedding_dim"], cfg["num_agents"])
            for rho in cfg["rhos"]:
                instances.append(
                    {
                        "instance_id": f"test_{'core' if experiment == EMPIRICAL else 'deep'}_{counter:03d}",
                        "experiment": experiment,
                        "branching": cfg["branching"],
                        "depth": cfg["depth"],
                        "num_agents": cfg["num_agents"],
                        "embedding_dim": cfg["embedding_dim"],
                        "rho": float(rho),
                        "token_vectors": v.tolist(),
                        "agent_vectors": w.tolist(),
                        "agent_token_logprobs": _token_logprobs(v, w, rho).tolist(),
                    }
                )
                counter += 1
    return instances
