#!/usr/bin/env python3
"""Train-slice scoring for empirical core-test artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np

from data.core_synthetic.generate import build_instance
from eval.core_metrics import max_coalition_improvement


SLICE = "train"
SPLIT_RULE = "fixed rho/seed grid in ascending order, then keep even-indexed keys"
EXPECTED_BRANCHING = 3
EXPECTED_DEPTH = 4
EXPECTED_AGENTS = 5
EXPECTED_EMBEDDING_DIM = 8
EXPECTED_RHOS = (0.0, 1.0, 2.0, 4.0, 8.0)
EXPECTED_SEEDS = (202510, 202511, 202512, 202513, 202514)
TOLERANCE = 1e-6


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


def _record_key(record: dict) -> tuple[float, int]:
    return (float(record["rho"]), int(record["seed"]))


def _expected_keys() -> list[tuple[float, int]]:
    return [(rho, seed) for rho in EXPECTED_RHOS for seed in EXPECTED_SEEDS]


def _expected_train_keys() -> set[tuple[float, int]]:
    return {key for index, key in enumerate(_expected_keys()) if index % 2 == 0}


def _check_expected_records(records: list[dict]) -> None:
    seen: set[tuple[float, int]] = set()
    duplicates: set[tuple[float, int]] = set()
    expected = set(_expected_keys())
    for record in records:
        key = _record_key(record)
        if key in seen:
            duplicates.add(key)
        seen.add(key)
    unknown = seen.difference(expected)
    if duplicates:
        raise ValueError(f"Duplicate core-test artifacts: {sorted(duplicates)}")
    if unknown:
        raise ValueError(f"Unexpected core-test artifacts: {sorted(unknown)}")


def _select_train(records: Iterable[dict]) -> list[dict]:
    record_list = list(records)
    _check_expected_records(record_list)
    train_keys = _expected_train_keys()
    selected = [record for record in record_list if _record_key(record) in train_keys]
    missing = train_keys.difference(_record_key(record) for record in selected)
    if missing:
        raise FileNotFoundError(f"Missing train-slice core-test artifacts: {sorted(missing)}")
    return sorted(selected, key=_record_sort_key)


def _expected_instance(record: dict):
    """Load the fixed benchmark instance instead of trusting artifact utilities."""

    for key, expected in (
        ("branching", EXPECTED_BRANCHING),
        ("depth", EXPECTED_DEPTH),
        ("num_agents", EXPECTED_AGENTS),
        ("embedding_dim", EXPECTED_EMBEDDING_DIM),
    ):
        if key not in record or int(record[key]) != expected:
            raise ValueError(
                f"Artifact for rho={record.get('rho')} seed={record.get('seed')} "
                f"has {key}={record.get(key)}, expected {expected}"
            )

    instance = build_instance(
        rho=float(record["rho"]),
        seed=int(record["seed"]),
        branching=EXPECTED_BRANCHING,
        depth=EXPECTED_DEPTH,
        num_agents=EXPECTED_AGENTS,
        embedding_dim=EXPECTED_EMBEDDING_DIM,
    )
    if "leaves" in record:
        artifact_leaves = np.asarray(record["leaves"], dtype=np.int64)
        if artifact_leaves.shape != instance.leaves.shape or not np.array_equal(
            artifact_leaves, instance.leaves
        ):
            raise ValueError(
                f"Artifact leaves do not match fixed benchmark instance for "
                f"rho={record['rho']} seed={record['seed']}"
            )
    return instance


def _distribution_stats(distribution: np.ndarray, num_leaves: int) -> tuple[np.ndarray, float]:
    if distribution.shape != (num_leaves,):
        raise ValueError(
            f"Policy distribution has shape {distribution.shape}, expected {(num_leaves,)}"
        )
    if not np.all(np.isfinite(distribution)):
        raise ValueError("Policy distribution must contain only finite values")
    negative_mass = float(np.abs(np.minimum(distribution, 0.0)).sum())
    simplex_l1_error = abs(float(distribution.sum()) - 1.0) + negative_mass
    if simplex_l1_error <= TOLERANCE:
        simplex_l1_error = 0.0
    clipped = np.maximum(distribution, 0.0)
    total = float(clipped.sum())
    if total <= TOLERANCE:
        raise ValueError("Policy distribution has no positive probability mass")
    return clipped / total, simplex_l1_error


def evaluate(input_dir: Path) -> dict:
    records = _select_train(_load_records(input_dir))
    if not records:
        raise FileNotFoundError(f"No train-slice core-test artifacts found in {input_dir}")

    by_policy: dict[str, list[float]] = {}
    by_policy_rho: dict[str, dict[float, list[float]]] = {}
    by_policy_simplex_error: dict[str, list[float]] = {}
    by_policy_min_utility: dict[str, list[float]] = {}
    per_instance = []
    for record in records:
        instance = _expected_instance(record)
        utilities = np.asarray(instance.utilities, dtype=np.float64)
        instance_scores = {
            "slice": SLICE,
            "rho": record["rho"],
            "seed": record["seed"],
            "policies": {},
        }
        for policy in record["policies"]:
            name = policy["name"]
            raw_distribution = np.asarray(policy["distribution"], dtype=np.float64)
            distribution, simplex_l1_error = _distribution_stats(
                raw_distribution, utilities.shape[1]
            )
            expected_utilities = utilities @ distribution
            alpha = max_coalition_improvement(utilities, distribution)
            by_policy.setdefault(name, []).append(alpha)
            by_policy_rho.setdefault(name, {}).setdefault(float(record["rho"]), []).append(alpha)
            by_policy_simplex_error.setdefault(name, []).append(simplex_l1_error)
            by_policy_min_utility.setdefault(name, []).append(float(expected_utilities.min()))
            instance_scores["policies"][name] = {
                "max_coalition_improvement": alpha,
                "mean_coalition_improvement": alpha,
                "coalition_violation_rate": 1.0 if alpha > 1.0 + TOLERANCE else 0.0,
                "core_constraint_satisfied": 1.0 if alpha <= 1.0 + TOLERANCE else 0.0,
                "distribution_simplex_l1_error": simplex_l1_error,
                "min_agent_expected_utility": float(expected_utilities.min()),
            }
        per_instance.append(instance_scores)

    results = {}
    for policy_name, alphas in by_policy.items():
        values = np.asarray(alphas, dtype=np.float64)
        simplex_errors = np.asarray(by_policy_simplex_error[policy_name], dtype=np.float64)
        min_utilities = np.asarray(by_policy_min_utility[policy_name], dtype=np.float64)
        results[policy_name] = {
            "max_coalition_improvement": float(values.max()),
            "mean_coalition_improvement": float(values.mean()),
            "coalition_violation_rate": float(np.mean(values > 1.0 + TOLERANCE)),
            "core_constraint_satisfied": 1.0 if float(values.max()) <= 1.0 + TOLERANCE else 0.0,
            "distribution_simplex_l1_error": float(simplex_errors.max()),
            "min_agent_expected_utility": float(min_utilities.min()),
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
                    "per_instance": per_instance,
                },
                "results": results,
            }
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="checkpoints/core_test")
    parser.add_argument("--scores", default="scoring/scores_train.json")
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
