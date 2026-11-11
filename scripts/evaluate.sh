#!/bin/bash
# Standard evaluation entry point. Self-contained in eval/ (never imports method/).
# Runs every solver in scripts/solvers/*.sh on the public training instances and writes
# scoring/scores.json. A solver is any script started as `bash scripts/solvers/<name>.sh` (no
# arguments) that speaks the evaluator's oracle protocol on stdin/stdout: it receives each
# instance, queries the agents' next-token log-probabilities within the instance's budget, and
# returns a stochastic token policy (see data/core_synthetic/oracle_client.py).
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --scores scoring/scores.json
