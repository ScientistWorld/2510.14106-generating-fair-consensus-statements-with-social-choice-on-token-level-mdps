#!/usr/bin/env python3
"""Run the paper's empirical core-test method on synthetic token trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from data.core_synthetic.generate import build_instance
from method.core_policy import induce_prefix_policy, solve_nash_welfare_distribution
from method.single_path_search import (
    leaf_log_utilities_from_prob_utilities,
    run_all_selectors,
)


def _distribution_payload(name: str, distribution: np.ndarray) -> dict:
    return {
        "name": name,
        "distribution": distribution.tolist(),
        "expected_utilities": None,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for rho in args.rhos:
        for seed in range(args.seeds):
            instance = build_instance(
                rho=rho,
                seed=args.seed_offset + seed,
                branching=args.branching,
                depth=args.depth,
                num_agents=args.agents,
                embedding_dim=args.embedding_dim,
            )
            nash = solve_nash_welfare_distribution(instance.utilities)
            if not nash.success:
                raise RuntimeError(f"Nash optimization failed: {nash.message}")

            uniform = np.full(instance.leaves.shape[0], 1.0 / instance.leaves.shape[0])
            utilitarian = np.zeros(instance.leaves.shape[0], dtype=np.float64)
            utilitarian[int(np.argmax(instance.utilities.sum(axis=0)))] = 1.0

            policies = [
                _distribution_payload("nash_welfare_policy", nash.distribution),
                _distribution_payload("uniform_policy", uniform),
                _distribution_payload("utilitarian_policy", utilitarian),
            ]
            for policy in policies:
                p = np.asarray(policy["distribution"], dtype=np.float64)
                policy["expected_utilities"] = (instance.utilities @ p).tolist()
            leaf_log_utilities = leaf_log_utilities_from_prob_utilities(instance.utilities)
            path_results = run_all_selectors(
                instance.leaves,
                leaf_log_utilities,
                instance.branching,
                lookahead=args.lookahead_depth,
                beam_width=args.beam_width,
                aggregation=args.path_aggregation,
            )

            record = {
                "experiment": "empirical_core_test",
                "rho": rho,
                "seed": args.seed_offset + seed,
                "branching": instance.branching,
                "depth": instance.depth,
                "num_agents": instance.num_agents,
                "embedding_dim": instance.embedding_dim,
                "leaves": instance.leaves.tolist(),
                "utilities": instance.utilities.tolist(),
                "policies": policies,
                "induced_prefix_policy": induce_prefix_policy(
                    instance.leaves, nash.distribution, instance.branching
                ),
                "nash_welfare": nash.nash_welfare,
                "paths": [
                    {
                        "name": result.name,
                        "path": list(result.path),
                        "egalitarian_log_welfare": result.egalitarian_welfare,
                        "egalitarian_perplexity": float(
                            np.exp(-result.egalitarian_welfare / instance.depth)
                        ),
                        "utilitarian_log_welfare": result.utilitarian_welfare,
                        "metadata": result.metadata,
                    }
                    for result in path_results
                ],
            }
            path = output_dir / f"core_rho{rho:g}_seed{args.seed_offset + seed}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(str(path))

    manifest = {"artifacts": records}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/core_test")
    parser.add_argument("--rhos", nargs="+", type=float, default=[0.0, 1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=202510)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--lookahead-depth", type=int, default=4)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--path-aggregation",
        choices=["mean", "best", "worst"],
        default="mean",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
