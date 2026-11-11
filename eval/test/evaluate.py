#!/usr/bin/env python3
"""Hidden-test scoring: run every solver on disk on held-out instances served by the oracle."""

from __future__ import annotations

import argparse
from pathlib import Path

from eval.test.core_eval import run_evaluation
from eval.test.instances import build_levels, test_specs

WORKSPACE = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scores", default=str(WORKSPACE / "scoring" / "scores_test.json"))
    parser.add_argument("--timeout", type=float, default=3600.0,
                        help="limit on the time the evaluator waits for each solver (s)")
    args = parser.parse_args()
    run_evaluation(
        test_specs(),
        build_levels,
        split="test",
        scores_path=Path(args.scores),
        workspace=WORKSPACE,
        timeout=args.timeout,
        log_dir=WORKSPACE / "logs" / "solvers_test",
    )


if __name__ == "__main__":
    main()
