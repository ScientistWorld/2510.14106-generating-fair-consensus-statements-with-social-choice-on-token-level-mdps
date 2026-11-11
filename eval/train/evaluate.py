#!/usr/bin/env python3
"""Visible train-slice scoring: run every solver on disk on the public training instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.core_synthetic.generate import build_levels, train_specs
from eval.train.core_eval import run_evaluation

WORKSPACE = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring" / "scores_train.json"))
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="limit on the time the evaluator waits for each solver (s)")
    args = parser.parse_args()
    run_evaluation(
        train_specs(),
        build_levels,
        split="train",
        scores_path=Path(args.scores),
        workspace=WORKSPACE,
        timeout=args.timeout,
        log_dir=WORKSPACE / "logs" / "solvers_train",
    )


if __name__ == "__main__":
    main()
