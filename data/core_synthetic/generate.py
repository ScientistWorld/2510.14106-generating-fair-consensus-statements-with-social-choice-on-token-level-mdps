"""Synthetic token-tree preference environments (paper Section 4.2, and a deep prefix-dependent tree).

Each agent i has a next-token policy on a B-ary token tree of depth L,

    pi_i(a | s) = softmax_a( rho * w_i^T (z(s) + v_{s,a}) ),

with unit-normalised Gaussian agent vectors w_i and token vectors v. The state term z(s) adds the
same logit to every action at a prefix and cancels in the softmax.

* ``empirical_core_test`` is the paper's Section 4.2 environment: B=3, L=4, six agents, d=8, token
  vectors indexed by depth only (v_{s,a} = v_{t,a} for every prefix s of length t), polarization
  rho on linspace(0.6, 5, 20). The tree has 40 internal prefixes and 81 complete statements.
* ``deep_tree_core_test``: B=3, L=12 (531,441 complete statements), six agents, d=8, and a fresh
  token vector for every prefix (each prefix has its own B candidate continuations, as when a base
  model proposes the top-B tokens of each prefix), so next-token distributions depend on the
  whole prefix. rho in {0.4, 0.8, 1.6, 3.2} spans moderate to strong polarization (rho * L from
  4.8 to 38; the paper uses rho * L from 2.4 to 20 at L=4).

Solvers never see these parameters. They see the agents' policies only through the evaluator's
query oracle (see ``oracle_client.py``): one query returns log pi_i(. | s) for every agent at one
prefix s, and every instance has a query budget. On the small trees the budget covers every
prefix; on the deep trees it covers about 0.6% of them.

This module is the public generator: it defines both instance families and the public training
grid, and is used by the visible training evaluator. It is method-agnostic.
"""

from __future__ import annotations

import numpy as np

EMPIRICAL = "empirical_core_test"
DEEP = "deep_tree_core_test"

EXPERIMENT_SETTINGS = {
    EMPIRICAL: {
        "branching": 3,
        "depth": 4,
        "num_agents": 6,
        "embedding_dim": 8,
        "token_vectors": "per_depth",
        "rhos": [float(r) for r in np.linspace(0.6, 5.0, 20)],
        "query_budget": 40,
    },
    DEEP: {
        "branching": 3,
        "depth": 12,
        "num_agents": 6,
        "embedding_dim": 8,
        "token_vectors": "per_prefix",
        "rhos": [0.4, 0.8, 1.6, 3.2],
        "query_budget": 1500,
    },
}

# Public training grid. Seeds 42-44 are the seeds used by the authors' core-test script.
TRAIN_SEEDS = {
    EMPIRICAL: [42, 43, 44],
    DEEP: [1001, 1002, 1003, 1004, 1005],
}


def num_internal_prefixes(branching: int, depth: int) -> int:
    return (branching ** depth - 1) // (branching - 1)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def generate_vectors(seed, cfg: dict):
    """Agent vectors w (n, d) and token vectors per depth: list of arrays (P_t, B, d).

    P_t = 1 for per-depth token vectors (shared by every prefix of length t) and B^t for
    per-prefix token vectors (row k belongs to the prefix whose base-B digits spell k).
    """

    rng = np.random.default_rng(seed)
    b, depth, d, n = cfg["branching"], cfg["depth"], cfg["embedding_dim"], cfg["num_agents"]
    if cfg["token_vectors"] == "per_depth":
        v = _unit(rng.normal(size=(depth, b, d)))
        w = _unit(rng.normal(size=(n, d)))
        return w, [v[t][None] for t in range(depth)]
    w = _unit(rng.normal(size=(n, d)))
    return w, [_unit(rng.normal(size=(b ** t, b, d))) for t in range(depth)]


def level_logprobs(w: np.ndarray, v_levels: list, rho: float) -> list:
    """log pi_i(a | s) per depth: list of arrays (n, P_t, B)."""

    out = []
    for vt in v_levels:
        logits = rho * np.einsum("nd,pbd->npb", w, vt)
        m = logits.max(axis=2, keepdims=True)
        s = logits - m
        out.append(s - np.log(np.exp(s).sum(axis=2, keepdims=True)))
    return out


def instance_specs(seeds_by_experiment: dict, prefix: str) -> list[dict]:
    specs = []
    for experiment in (EMPIRICAL, DEEP):
        cfg = EXPERIMENT_SETTINGS[experiment]
        counter = 0
        for seed in seeds_by_experiment.get(experiment, []):
            for rho in cfg["rhos"]:
                specs.append({
                    "instance_id": f"{prefix}_{'core' if experiment == EMPIRICAL else 'deep'}_{counter:03d}",
                    "experiment": experiment,
                    "seed": seed,
                    "rho": float(rho),
                    **{k: cfg[k] for k in ("branching", "depth", "num_agents", "embedding_dim",
                                           "token_vectors", "query_budget")},
                })
                counter += 1
    return specs


def train_specs() -> list[dict]:
    return instance_specs(TRAIN_SEEDS, "train")


_VECTOR_CACHE: dict = {}


def build_levels(spec: dict) -> list:
    """Per-depth next-token log-probabilities of every agent for one instance spec."""

    key = (spec["experiment"], spec["seed"])
    if key not in _VECTOR_CACHE:
        _VECTOR_CACHE.clear()  # keep one seed's vectors (consecutive specs share a seed)
        _VECTOR_CACHE[key] = generate_vectors(spec["seed"], EXPERIMENT_SETTINGS[spec["experiment"]])
    w, v = _VECTOR_CACHE[key]
    return level_logprobs(w, v, spec["rho"])


if __name__ == "__main__":
    for experiment, cfg in EXPERIMENT_SETTINGS.items():
        print(experiment, {k: v for k, v in cfg.items() if k != "rhos"}, "rhos", cfg["rhos"],
              "internal prefixes", num_internal_prefixes(cfg["branching"], cfg["depth"]))
