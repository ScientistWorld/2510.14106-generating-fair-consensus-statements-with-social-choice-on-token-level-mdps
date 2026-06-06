#!/usr/bin/env python3
"""Evaluate deterministic-search artifacts without importing method code."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np


def _load_records(input_dir: Path) -> list[dict]:
    manifest = input_dir / "manifest.json"
    if manifest.exists():
        paths = [Path(p) for p in json.loads(manifest.read_text())["artifacts"]]
    else:
        paths = sorted(input_dir.glob("search_rho*_seed*.json"))
    return [json.loads(path.read_text(encoding="utf-8")) for path in paths]


def _path_index(leaves: np.ndarray, path: list[int]) -> int:
    path_array = np.asarray(path, dtype=np.int64)
    matches = np.where(np.all(leaves == path_array, axis=1))[0]
    if len(matches) != 1:
        raise ValueError(f"path {path!r} does not identify exactly one leaf")
    return int(matches[0])


def _path_metrics(leaves: np.ndarray, utilities: np.ndarray, path: list[int], depth: int) -> dict:
    index = _path_index(leaves, path)
    agent_log_utilities = np.log(np.maximum(utilities[:, index], 1e-300))
    egalitarian_log_utility = float(agent_log_utilities.min())
    mean_egalitarian_log_utility = egalitarian_log_utility / depth
    return {
        "egalitarian_log_utility": egalitarian_log_utility,
        "egalitarian_perplexity": float(math.exp(-mean_egalitarian_log_utility)),
        "mean_agent_log_utility": float(agent_log_utilities.mean()),
        "path_index": index,
    }


def evaluate(input_dir: Path) -> dict:
    records = _load_records(input_dir)
    if not records:
        raise FileNotFoundError(f"No deterministic-search artifacts found in {input_dir}")

    values_by_method: dict[str, list[dict]] = {}
    per_instance = []
    for record in records:
        leaves = np.asarray(record["leaves"], dtype=np.int64)
        utilities = np.asarray(record["utilities"], dtype=np.float64)
        depth = int(record["depth"])
        instance = {"rho": record["rho"], "seed": record["seed"], "paths": {}}
        oracle_eppl = None
        for item in record["paths"]:
            name = item["name"]
            metrics = _path_metrics(leaves, utilities, item["path"], depth)
            if name == "exhaustive_egalitarian_oracle":
                oracle_eppl = metrics["egalitarian_perplexity"]
            values_by_method.setdefault(name, []).append(metrics)
            instance["paths"][name] = metrics
        if oracle_eppl is not None:
            for metrics in instance["paths"].values():
                metrics["eppl_gap_to_oracle"] = metrics["egalitarian_perplexity"] - oracle_eppl
        per_instance.append(instance)

    results = {}
    for method_name, metric_rows in values_by_method.items():
        eppl = np.asarray([row["egalitarian_perplexity"] for row in metric_rows], dtype=np.float64)
        log_utility = np.asarray([row["egalitarian_log_utility"] for row in metric_rows], dtype=np.float64)
        gaps = np.asarray([row.get("eppl_gap_to_oracle", np.nan) for row in metric_rows], dtype=np.float64)
        results[method_name] = {
            "mean_egalitarian_perplexity": float(eppl.mean()),
            "max_egalitarian_perplexity": float(eppl.max()),
            "mean_egalitarian_log_utility": float(log_utility.mean()),
            "mean_eppl_gap_to_oracle": float(np.nanmean(gaps)),
        }

    return {
        "experiments": {
            "synthetic_deterministic_search": {
                "_details": {
                    "num_instances": len(records),
                    "note": (
                        "Supplemental scaled experiment; not part of scoring/reference.json "
                        "because the paper's reported EPPL uses Llama/Habermas data."
                    ),
                    "per_instance": per_instance,
                },
                "results": results,
            }
        }
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="checkpoints/deterministic_search")
    parser.add_argument("--scores", default="scoring/deterministic_search_scores.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scores = evaluate(Path(args.input_dir))
    output = Path(args.scores)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scores, indent=2), encoding="utf-8")
    print(json.dumps(scores["experiments"]["synthetic_deterministic_search"]["results"], indent=2))


if __name__ == "__main__":
    main()
