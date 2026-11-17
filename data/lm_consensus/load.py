"""Public training instances of the language-model consensus-search experiment (method-agnostic).

Each instance is a token tree built by ``build_trees.py`` from one scenario in
``scenarios_train.json``: branching B=2 (the base model's two most likely next tokens at every
prefix), depth L=16, and n participants whose next-token log-probabilities come from the same
language model prompted with each participant's opinion. Solvers never see the text; they see
the participants' (and the base model's) log-probabilities of the B candidates at the prefixes
they query, within the instance's query budget.

``build_lm_levels(spec)`` returns {"agents": [ (n, B^t, B) ], "base": [ (B^t, B) ]} for
t = 0..L-1, in the same row layout as the synthetic trees (row k of level t is the prefix whose
candidate indices spell k in base B).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

LM = "lm_consensus_search"
HERE = Path(__file__).resolve().parent
SCENARIOS = HERE / "scenarios_train.json"
TREE_DIR = HERE / "trees"
SETTINGS = {"branching": 2, "depth": 16, "query_budget": 150}


def lm_train_specs() -> list[dict]:
    scenarios = json.loads(SCENARIOS.read_text(encoding="utf-8"))
    return [{
        "instance_id": sc["id"],
        "experiment": LM,
        "seed": None,
        "rho": None,
        "branching": SETTINGS["branching"],
        "depth": SETTINGS["depth"],
        "num_agents": len(sc["opinions"]),
        "query_budget": SETTINGS["query_budget"],
        "tree_file": str(TREE_DIR / f"{sc['id']}.npz"),
    } for sc in scenarios]


def build_lm_levels(spec: dict) -> dict:
    with np.load(spec["tree_file"]) as z:
        depth, b, n = int(spec["depth"]), int(spec["branching"]), int(spec["num_agents"])
        agents = [z[f"agents_{t}"].astype(np.float64) for t in range(depth)]
        base = [z[f"base_{t}"].astype(np.float64) for t in range(depth)]
    for t in range(depth):
        if agents[t].shape != (n, b ** t, b) or base[t].shape != (b ** t, b):
            raise ValueError(f"{spec['tree_file']}: unexpected shape at level {t}")
    return {"agents": agents, "base": base}


def token_ids(spec: dict) -> list:
    """Token ids of the candidates at every prefix (for decoding statements; not served)."""

    with np.load(spec["tree_file"]) as z:
        return [z[f"ids_{t}"] for t in range(int(spec["depth"]))]
