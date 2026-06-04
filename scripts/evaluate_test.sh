#!/bin/bash
# Held-out test-slice evaluation. Mirrors scripts/evaluate.sh's artifact contract
# but writes scoring/scores_test.json and requires methods seen in train scores.

set -e

cd /home/user

mkdir -p scoring

INPUT_DIR="${1:-checkpoints/core_test}"
python -m eval.test.evaluate_core_test \
  --input-dir "$INPUT_DIR" \
  --scores scoring/scores_test.json \
  --train-scores scoring/scores_train.json
