#!/usr/bin/env python3
"""Visible train-slice scoring: run every solver on disk on the public training instances."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.core_synthetic.generate import build_levels, train_specs
from data.lm_consensus.load import LM, build_lm_levels, lm_train_specs
from eval.train.core_eval import run_evaluation

WORKSPACE = Path(__file__).resolve().parents[2]


def build(spec: dict):
    return build_lm_levels(spec) if spec["experiment"] == LM else build_levels(spec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring" / "scores_train.json"))
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="limit on the time the evaluator waits for each solver (s)")
    parser.add_argument("--experiments", nargs="*", default=None,
                        help="diagnostic runs only: restrict to these experiments (default: all)")
    args = parser.parse_args()
    run_evaluation(
        train_specs() + lm_train_specs(),
        build,
        split="train",
        scores_path=Path(args.scores),
        workspace=WORKSPACE,
        timeout=args.timeout,
        log_dir=WORKSPACE / "logs" / "solvers_train",
        experiments=args.experiments,
    )


if __name__ == "__main__":
    main()
