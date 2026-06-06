#!/usr/bin/env python3
"""Run bounded deterministic EW path-selection on synthetic token trees."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data.core_synthetic.generate import build_instance
from method.single_path_search import (
    leaf_log_utilities_from_prob_utilities,
    run_all_selectors,
)


def run(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[str] = []

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
            leaf_log_utilities = leaf_log_utilities_from_prob_utilities(
                instance.utilities
            )
            results = run_all_selectors(
                instance.leaves,
                leaf_log_utilities,
                instance.branching,
                lookahead=args.lookahead,
                beam_width=args.beam_width,
                aggregation=args.aggregation,
            )

            record = {
                "experiment": "single_path_egalitarian_synthetic",
                "rho": rho,
                "seed": seed,
                "branching": instance.branching,
                "depth": instance.depth,
                "num_agents": instance.num_agents,
                "embedding_dim": instance.embedding_dim,
                "lookahead": args.lookahead,
                "beam_width": args.beam_width,
                "aggregation": args.aggregation,
                "selectors": [asdict(result) for result in results],
            }
            path = output_dir / f"single_path_rho{rho:g}_seed{seed}.json"
            path.write_text(json.dumps(record, indent=2), encoding="utf-8")
            artifacts.append(str(path))

    (output_dir / "manifest.json").write_text(
        json.dumps({"artifacts": artifacts}, indent=2),
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/single_path_search")
    parser.add_argument("--rhos", nargs="+", type=float, default=[1.0, 4.0, 8.0])
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--seed-offset", type=int, default=202510)
    parser.add_argument("--branching", type=int, default=3)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--agents", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=8)
    parser.add_argument("--lookahead", type=int, default=2)
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument(
        "--aggregation",
        choices=["mean", "best", "worst"],
        default="mean",
        help="How bounded prefix searches score descendant leaf log utilities.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
