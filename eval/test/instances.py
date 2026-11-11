"""Held-out test instances. Private to eval/test/: nothing outside this directory can regenerate them.

Same instance families and generating process as the public training grid (paper Section 4.2
trees and the deep prefix-dependent trees), but the seeds below are 128-bit secrets used only
here, and this is a private copy of the generator, so editing public data code cannot change the
test ground truth. Solvers never see these parameters: the evaluator serves the agents' policies
through its query oracle.
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
        "rhos": [0.6, 1.2, 2.4, 4.8],
        "query_budget": 1500,
    },
}

_SEEDS = {
    EMPIRICAL: [
        187467005423487668866679900654475066387,
        92602817955918559778605316817005890609,
        100241760611236698106173466815995678012,
        124948664797661884906316732608950276223,
        182009451633985238586055078217946590121,
    ],
    DEEP: [
        188290670554331227035960578107643378456,
        281395383811491243988366189654405500131,
        175219895180530813842909839042816699184,
        127448278942929321549038229309549593272,
        221785200858676058664248933179552991978,
        205723567811725313417570178369291303828,
        20187137560332875398768747202203598692,
        197394862045449568635260320257412926585,
        320211363870170874381208516279710707816,
        29788871434089837403370034451414407663,
    ],
}


def _unit(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def _vectors(seed: int, cfg: dict):
    rng = np.random.default_rng(seed)
    b, depth, d, n = cfg["branching"], cfg["depth"], cfg["embedding_dim"], cfg["num_agents"]
    if cfg["token_vectors"] == "per_depth":
        v = _unit(rng.normal(size=(depth, b, d)))
        w = _unit(rng.normal(size=(n, d)))
        return w, [v[t][None] for t in range(depth)]
    w = _unit(rng.normal(size=(n, d)))
    return w, [_unit(rng.normal(size=(b ** t, b, d))) for t in range(depth)]


def _level_logprobs(w: np.ndarray, v_levels: list, rho: float) -> list:
    out = []
    for vt in v_levels:
        logits = rho * np.einsum("nd,pbd->npb", w, vt)
        m = logits.max(axis=2, keepdims=True)
        s = logits - m
        out.append(s - np.log(np.exp(s).sum(axis=2, keepdims=True)))
    return out


def test_specs() -> list[dict]:
    specs = []
    for experiment in (EMPIRICAL, DEEP):
        cfg = SETTINGS[experiment]
        counter = 0
        for seed in _SEEDS[experiment]:
            for rho in cfg["rhos"]:
                specs.append({
                    "instance_id": f"test_{'core' if experiment == EMPIRICAL else 'deep'}_{counter:03d}",
                    "experiment": experiment,
                    "seed": seed,
                    "rho": float(rho),
                    **{k: cfg[k] for k in ("branching", "depth", "num_agents", "embedding_dim",
                                           "token_vectors", "query_budget")},
                })
                counter += 1
    return specs


_CACHE: dict = {}


def build_levels(spec: dict) -> list:
    key = (spec["experiment"], spec["seed"])
    if key not in _CACHE:
        _CACHE.clear()
        _CACHE[key] = _vectors(spec["seed"], SETTINGS[spec["experiment"]])
    w, v = _CACHE[key]
    return _level_logprobs(w, v, spec["rho"])
