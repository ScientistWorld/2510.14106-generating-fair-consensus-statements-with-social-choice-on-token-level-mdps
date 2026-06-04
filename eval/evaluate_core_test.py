#!/usr/bin/env python3
"""Evaluate artifacts from the empirical core-test experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval.core_metrics import max_coalition_improvement


def _load_records(input_dir: Path) -> list[dict]:
    manifest = input_dir / "manifest.json"
    if manifest.exists():
        paths = [Path(p) for p in json.loads(manifest.read_text())["artifacts"]]
    else:
        paths = sorted(input_dir.glob("core_rho*_seed*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def evaluate(input_dir: Path) -> dict:
    records = _load_records(input_dir)
    if not records:
        raise FileNotFoundError(f"No core-test artifacts found in {input_dir}")

    by_policy: dict[str, list[float]] = {}
    per_instance = []
    for record in records:
        utilities = np.asarray(record["utilities"], dtype=np.float64)
        instance_scores = {"rho": record["rho"], "seed": record["seed"], "policies": {}}
        for policy in record["policies"]:
            name = policy["name"]
            if name != "nash_welfare_policy":
                continue
            distribution = np.asarray(policy["distribution"], dtype=np.float64)
            alpha = max_coalition_improvement(utilities, distribution)
            by_policy.setdefault(name, []).append(alpha)
            instance_scores["policies"][name] = {
                "max_coalition_improvement": alpha,
                "core_constraint_satisfied": 1.0 if alpha <= 1.0 + 1e-6 else 0.0,
            }
        per_instance.append(instance_scores)

    results = {}
    for policy_name, alphas in by_policy.items():
        if policy_name != "nash_welfare_policy":
            continue
        values = np.asarray(alphas, dtype=np.float64)
        results[policy_name] = {
            "max_coalition_improvement": float(values.max()),
            "core_constraint_satisfied": 1.0 if float(values.max()) <= 1.0 + 1e-6 else 0.0,
        }

    return {
        "experiments": {
            "empirical_core_test": {
                "_details": {
                    "num_instances": len(records),
                    "per_instance": per_instance,
                },
                "results": results,
            }
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="checkpoints/core_test")
    parser.add_argument("--scores", default="scoring/scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = evaluate(Path(args.input_dir))
    output = Path(args.scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores["experiments"]["empirical_core_test"]["results"], indent=2))


if __name__ == "__main__":
    main()
