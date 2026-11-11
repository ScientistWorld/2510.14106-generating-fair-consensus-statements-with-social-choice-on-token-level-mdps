#!/usr/bin/env python3
"""Hidden-test scoring: write held-out instances, run every solver on disk, score its policies."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.test.core_eval import run_evaluation
from eval.test.instances import test_instances

WORKSPACE = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring" / "scores_test.json"))
    parser.add_argument("--timeout", type=float, default=7200.0, help="wall-clock limit per solver (s)")
    args = parser.parse_args()
    run_evaluation(
        test_instances(),
        split="test",
        scores_path=Path(args.scores),
        workspace=WORKSPACE,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    main()
