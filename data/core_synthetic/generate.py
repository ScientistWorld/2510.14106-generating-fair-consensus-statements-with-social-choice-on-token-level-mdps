"""Synthetic token-tree preference environments (paper Section 4.2 and a deep-tree scale-up).

Each agent i has a next-token policy over a B-ary token tree of depth L,

    pi_i(a | s) = softmax_a( rho * w_i^T (z(s) + v_{t,a}) ),   z(s) = sum_{tau<t} v_{tau, a_tau},

with unit-normalised Gaussian token vectors v_{t,a} and agent vectors w_i, exactly as in
Section 4.2 of the paper (and the authors' public ``core.py``). The state term z(s) adds the
same logit to every action at a prefix, so the next-token distribution at depth t is identical
for every prefix of that depth; ``agent_token_logprobs[i][t][a]`` therefore gives
log pi_i(a | s) for every prefix s of length t. ``next_token_logprobs`` evaluates the formula
literally (with z(s)) for checking.

This module is the public generator: it defines the two instance families and the public
training grid. It is method-agnostic.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

EMPIRICAL = "empirical_core_test"
DEEP = "deep_tree_core_test"

# Section 4.2 setting from the paper and the authors' code: B=3, L=4, d=8, six agents,
# rho on linspace(0.6, 5.0, 20).
# Deep-tree setting: same branching, agents and embedding, depth 24 (3^24 ~ 2.8e11 complete
# statements). Polarization is rescaled by 4/24 so that rho * L spans the paper's range
# (2.4 ... 20) at the longer statement length.
EXPERIMENT_SETTINGS = {
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

# Public training grid. Seeds 42-44 are the seeds used by the authors' core-test script.
TRAIN_SEEDS = {
    EMPIRICAL: [42, 43, 44],
    DEEP: [1001, 1002, 1003],
}


def generate_vectors(seed: int, branching: int, depth: int, embedding_dim: int, num_agents: int):
    """Token vectors v (L, B, d) and agent vectors w (n, d), unit-normalised."""

    rng = np.random.default_rng(seed)
    v = rng.normal(size=(depth, branching, embedding_dim))
    v /= np.linalg.norm(v, axis=2, keepdims=True) + 1e-12
    w = rng.normal(size=(num_agents, embedding_dim))
    w /= np.linalg.norm(w, axis=1, keepdims=True) + 1e-12
    return v, w


def _log_softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(x, axis=axis, keepdims=True)
    s = x - m
    return s - np.log(np.sum(np.exp(s), axis=axis, keepdims=True))


def token_logprobs(v: np.ndarray, w: np.ndarray, rho: float) -> np.ndarray:
    """log pi_i(a | s) for any prefix s at depth t, shape (n, L, B)."""

    logits = rho * np.einsum("nd,lbd->nlb", w, v)
    return _log_softmax(logits, axis=2)


def next_token_logprobs(v: np.ndarray, w: np.ndarray, rho: float, prefix) -> np.ndarray:
    """Literal evaluation of the paper's policy at an explicit prefix, shape (n, B)."""

    t = len(prefix)
    z = np.zeros(v.shape[2])
    for tau, a in enumerate(prefix):
        z = z + v[tau, int(a)]
    logits = rho * (w @ (z[None, :] + v[t]).T)
    return _log_softmax(logits, axis=1)


def build_instance(experiment: str, seed: int, rho: float, instance_id: str) -> dict:
    cfg = EXPERIMENT_SETTINGS[experiment]
    v, w = generate_vectors(
        seed, cfg["branching"], cfg["depth"], cfg["embedding_dim"], cfg["num_agents"]
    )
    return {
        "instance_id": instance_id,
        "experiment": experiment,
        "branching": cfg["branching"],
        "depth": cfg["depth"],
        "num_agents": cfg["num_agents"],
        "embedding_dim": cfg["embedding_dim"],
        "rho": float(rho),
        "token_vectors": v.tolist(),
        "agent_vectors": w.tolist(),
        "agent_token_logprobs": token_logprobs(v, w, rho).tolist(),
    }


def build_grid(seeds_by_experiment: dict, prefix: str) -> list[dict]:
    instances = []
    for experiment in (EMPIRICAL, DEEP):
        counter = 0
        for seed in seeds_by_experiment.get(experiment, []):
            for rho in EXPERIMENT_SETTINGS[experiment]["rhos"]:
                instance_id = f"{prefix}_{'core' if experiment == EMPIRICAL else 'deep'}_{counter:03d}"
                instances.append(build_instance(experiment, seed, rho, instance_id))
                counter += 1
    return instances


def train_instances() -> list[dict]:
    return build_grid(TRAIN_SEEDS, "train")


def write_instances(instances: list[dict], directory: Path) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for inst in instances:
        name = f"{inst['instance_id']}.json"
        (directory / name).write_text(json.dumps(inst), encoding="utf-8")
        entries.append({"instance_id": inst["instance_id"], "experiment": inst["experiment"], "file": name})
    (directory / "manifest.json").write_text(json.dumps({"instances": entries}, indent=1), encoding="utf-8")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Write public training instances for solver development.")
    parser.add_argument("--output-dir", default="checkpoints/train_instances")
    args = parser.parse_args()
    write_instances(train_instances(), Path(args.output_dir))
    print(f"wrote {len(train_instances())} instances to {args.output_dir}")


if __name__ == "__main__":
    main()
