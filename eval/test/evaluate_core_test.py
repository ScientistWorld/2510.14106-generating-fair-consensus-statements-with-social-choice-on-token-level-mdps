#!/usr/bin/env python3
"""Held-out test-slice scoring for empirical core-test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from eval.core_metrics import max_coalition_improvement


SLICE = "test"
SPLIT_RULE = "sort by (rho, seed, filename), then keep odd-indexed artifacts"


def _resolve_artifact_path(input_dir: Path, artifact: str) -> Path:
    path = Path(artifact)
    if path.is_absolute() or path.exists():
        return path
    return input_dir / path


def _load_records(input_dir: Path) -> list[dict]:
    manifest = input_dir / "manifest.json"
    if manifest.exists():
        manifest_data = json.loads(manifest.read_text(encoding="utf-8"))
        paths = [
            _resolve_artifact_path(input_dir, artifact)
            for artifact in manifest_data["artifacts"]
        ]
    else:
        paths = sorted(input_dir.glob("core_rho*_seed*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _record_sort_key(record: dict) -> tuple[float, int, str]:
    return (
        float(record["rho"]),
        int(record["seed"]),
        str(record.get("artifact", "")),
    )


def _select_test(records: Iterable[dict]) -> list[dict]:
    ordered = sorted(records, key=_record_sort_key)
    return [record for index, record in enumerate(ordered) if index % 2 == 1]


def _expected_methods(train_scores: Path) -> set[str]:
    if not train_scores.exists():
        return set()
    payload = json.loads(train_scores.read_text(encoding="utf-8"))
    return set(
        payload["experiments"]["empirical_core_test"]["results"].keys()
    )


def evaluate(input_dir: Path, train_scores: Path) -> dict:
    records = _select_test(_load_records(input_dir))
    if not records:
        raise FileNotFoundError(f"No test-slice core-test artifacts found in {input_dir}")

    by_policy: dict[str, list[float]] = {}
    by_policy_rho: dict[str, dict[float, list[float]]] = {}
    per_instance = []
    for record in records:
        utilities = np.asarray(record["utilities"], dtype=np.float64)
        instance_scores = {
            "slice": SLICE,
            "rho": record["rho"],
            "seed": record["seed"],
            "policies": {},
        }
        for policy in record["policies"]:
            name = policy["name"]
            distribution = np.asarray(policy["distribution"], dtype=np.float64)
            alpha = max_coalition_improvement(utilities, distribution)
            by_policy.setdefault(name, []).append(alpha)
            by_policy_rho.setdefault(name, {}).setdefault(float(record["rho"]), []).append(alpha)
            instance_scores["policies"][name] = {
                "max_coalition_improvement": alpha,
                "core_constraint_satisfied": 1.0 if alpha <= 1.0 + 1e-6 else 0.0,
            }
        per_instance.append(instance_scores)

    expected = _expected_methods(train_scores)
    missing = sorted(expected.difference(by_policy))
    if missing:
        raise ValueError(
            "Test slice is missing methods present in train scores: "
            + ", ".join(missing)
        )

    results = {}
    result_names = sorted(expected) if expected else sorted(by_policy)
    for policy_name in result_names:
        alphas = by_policy[policy_name]
        values = np.asarray(alphas, dtype=np.float64)
        results[policy_name] = {
            "max_coalition_improvement": float(values.max()),
            "core_constraint_satisfied": 1.0 if float(values.max()) <= 1.0 + 1e-6 else 0.0,
        }
        for rho, rho_alphas in sorted(by_policy_rho[policy_name].items()):
            rho_values = np.asarray(rho_alphas, dtype=np.float64)
            rho_key = f"rho_{rho:g}_max_coalition_improvement"
            results[policy_name][rho_key] = float(rho_values.max())

    return {
        "experiments": {
            "empirical_core_test": {
                "_details": {
                    "slice": SLICE,
                    "split_rule": SPLIT_RULE,
                    "num_instances": len(records),
                    "required_methods_from_train": sorted(expected),
                    "per_instance": per_instance,
                },
                "results": results,
            }
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="checkpoints/core_test")
    parser.add_argument("--scores", default="scoring/scores_test.json")
    parser.add_argument("--train-scores", default="scoring/scores_train.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = evaluate(Path(args.input_dir), Path(args.train_scores))
    output = Path(args.scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores["experiments"]["empirical_core_test"]["results"], indent=2))


if __name__ == "__main__":
    main()
