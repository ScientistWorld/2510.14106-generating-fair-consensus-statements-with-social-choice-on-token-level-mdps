#!/usr/bin/env python3
"""Run deterministic egalitarian search algorithms on synthetic token trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.core_synthetic.generate import build_instance
from method.search_policy import (
    beam_search,
    best_of_n_uniform,
    exhaustive_egalitarian_optimum,
    finite_lookahead_search,
)


def _result_payload(result) -> dict:
    return {
        "name": result.name,
        "path": list(result.path),
        "method_reported_egalitarian_log_utility": result.score,
    }


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for rho in args.rhos:
        for seed_index in range(args.seeds):
            seed = args.seed_offset + seed_index
            instance = build_instance(
                rho=rho,
                seed=seed,
                branching=args.branching,
                depth=args.depth,
                num_agents=args.agents,
                embedding_dim=args.embedding_dim,
            )
            results = [
                _result_payload(
                    finite_lookahead_search(
                        instance.leaves,
                        instance.utilities,
                        branching=instance.branching,
                        depth=instance.depth,
                        lookahead_depth=1,
                    )
                ),
                _result_payload(
                    finite_lookahead_search(
                        instance.leaves,
                        instance.utilities,
                        branching=instance.branching,
                        depth=instance.depth,
                        lookahead_depth=args.lookahead_depth,
                    )
                ),
                _result_payload(
                    beam_search(
                        instance.leaves,
                        instance.utilities,
                        branching=instance.branching,
                        depth=instance.depth,
                        beam_width=args.beam_width,
                    )
                ),
                _result_payload(
                    best_of_n_uniform(
                        instance.leaves,
                        instance.utilities,
                        samples=args.best_of_n,
                        seed=seed + 10_000,
                    )
                ),
                _result_payload(exhaustive_egalitarian_optimum(instance.leaves, instance.utilities)),
            ]
            record = {
                "experiment": "synthetic_deterministic_search",
                "rho": rho,
                "seed": seed,
                "branching": instance.branching,
                "depth": instance.depth,
                "num_agents": instance.num_agents,
                "embedding_dim": instance.embedding_dim,
                "leaves": instance.leaves.tolist(),
                "utilities": instance.utilities.tolist(),
                "paths": results,
                "deviation_from_paper": (
                    "Uses the paper's finite-tree token-level MDP objective and "
                    "search algorithms, but substitutes the Section 4.2 synthetic "
                    "policy family for the Llama/Habermas text-generation stack."
                ),
            }
            path = output_dir / f"search_rho{rho:g}_seed{seed}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            records.append(str(path))

    (output_dir / "manifest.json").write_text(
        json.dumps({"artifacts": records}, indent=2), encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/deterministic_search")
    parser.add_argument("--rhos", nargs="+", type=float, default=[1.0, 2.0, 4.0, 8.0])
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--seed-offset", type=int, default=202610)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--depth", type=int, default=6)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--lookahead-depth", type=int, default=3)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--best-of-n", type=int, default=4)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
