#!/bin/bash
# Held-out test evaluation. Takes no arguments: writes the hidden test instances, runs every
# solver in scripts/solvers/*.sh on them, and writes every method's numbers (plus the
# evaluator-owned baselines) to scoring/scores_test.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.test.evaluate --scores scoring/scores_test.json
