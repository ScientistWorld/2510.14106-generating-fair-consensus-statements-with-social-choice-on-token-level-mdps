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
    by_policy_rho: dict[str, dict[float, list[float]]] = {}
    simplex_error_by_policy: dict[str, list[float]] = {}
    min_utility_by_policy: dict[str, list[float]] = {}
    path_by_method: dict[str, list[dict[str, float]]] = {}
    per_instance = []
    for record in records:
        utilities = np.asarray(record["utilities"], dtype=np.float64)
        instance_scores = {
            "rho": record["rho"],
            "seed": record["seed"],
            "policies": {},
            "paths": {},
        }
        for policy in record["policies"]:
            name = policy["name"]
            distribution = np.asarray(policy["distribution"], dtype=np.float64)
            simplex_error = float(abs(distribution.sum() - 1.0) + np.maximum(-distribution, 0.0).sum())
            alpha = max_coalition_improvement(utilities, distribution)
            expected_utilities = utilities @ (np.maximum(distribution, 0.0) / np.maximum(distribution.sum(), 1e-300))
            by_policy.setdefault(name, []).append(alpha)
            by_policy_rho.setdefault(name, {}).setdefault(float(record["rho"]), []).append(alpha)
            simplex_error_by_policy.setdefault(name, []).append(simplex_error)
            min_utility_by_policy.setdefault(name, []).append(float(expected_utilities.min()))
            instance_scores["policies"][name] = {
                "max_coalition_improvement": alpha,
                "core_constraint_satisfied": 1.0 if alpha <= 1.0 + 1e-6 else 0.0,
                "distribution_simplex_l1_error": simplex_error,
                "min_agent_expected_utility": float(expected_utilities.min()),
            }
        for path_record in record.get("paths", []):
            name = path_record["name"]
            metrics = {
                "egalitarian_log_welfare": float(path_record["egalitarian_log_welfare"]),
                "egalitarian_perplexity": float(path_record["egalitarian_perplexity"]),
            }
            path_by_method.setdefault(name, []).append(metrics)
            instance_scores["paths"][name] = metrics
        per_instance.append(instance_scores)

    results = {}
    for policy_name, alphas in by_policy.items():
        values = np.asarray(alphas, dtype=np.float64)
        results[policy_name] = {
            "max_coalition_improvement": float(values.max()),
            "mean_coalition_improvement": float(values.mean()),
            "coalition_violation_rate": float(np.mean(values > 1.0 + 1e-6)),
            "core_constraint_satisfied": 1.0 if float(values.max()) <= 1.0 + 1e-6 else 0.0,
            "distribution_simplex_l1_error": float(np.max(simplex_error_by_policy[policy_name])),
            "min_agent_expected_utility": float(np.min(min_utility_by_policy[policy_name])),
        }
        for rho, rho_alphas in sorted(by_policy_rho[policy_name].items()):
            rho_values = np.asarray(rho_alphas, dtype=np.float64)
            rho_key = f"rho_{rho:g}_max_coalition_improvement"
            results[policy_name][rho_key] = float(rho_values.max())

    path_results = {}
    for method_name, metrics in path_by_method.items():
        log_welfare = np.asarray([row["egalitarian_log_welfare"] for row in metrics], dtype=np.float64)
        perplexity = np.asarray([row["egalitarian_perplexity"] for row in metrics], dtype=np.float64)
        path_results[method_name] = {
            "mean_egalitarian_perplexity": float(perplexity.mean()),
            "max_egalitarian_perplexity": float(perplexity.max()),
            "mean_egalitarian_log_welfare": float(log_welfare.mean()),
            "min_egalitarian_log_welfare": float(log_welfare.min()),
        }

    return {
        "experiments": {
            "empirical_core_test": {
                "_details": {
                    "num_instances": len(records),
                    "per_instance": per_instance,
                },
                "results": results,
            },
            "synthetic_single_statement_search": {
                "_details": {
                    "num_instances": len(records),
                    "description": "Deterministic path search artifacts saved by method.run_core_test.",
                },
                "results": path_results,
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
