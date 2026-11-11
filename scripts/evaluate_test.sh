#!/bin/bash
# Held-out test evaluation. Takes no arguments: builds the hidden test instances in memory, runs
# every solver in scripts/solvers/*.sh against them through the query oracle, and writes every
# method's numbers (plus the evaluator-owned baselines) to scoring/scores_test.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.test.evaluate --scores scoring/scores_test.json
