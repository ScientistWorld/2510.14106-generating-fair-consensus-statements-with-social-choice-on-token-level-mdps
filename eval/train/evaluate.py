#!/usr/bin/env python3
"""Visible train-slice scoring: write the public training instances, run every solver on disk, score."""

from __future__ import annotations

import argparse
from pathlib import Path

from data.core_synthetic.generate import train_instances
from eval.train.core_eval import run_evaluation

WORKSPACE = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring" / "scores_train.json"))
    parser.add_argument("--timeout", type=float, default=7200.0, help="wall-clock limit per solver (s)")
    args = parser.parse_args()
    run_evaluation(
        train_instances(),
        split="train",
        scores_path=Path(args.scores),
        workspace=WORKSPACE,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
