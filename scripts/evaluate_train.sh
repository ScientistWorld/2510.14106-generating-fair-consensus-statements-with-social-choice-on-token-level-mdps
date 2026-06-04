#!/bin/bash
# Visible train-slice evaluation. Mirrors scripts/evaluate.sh's artifact contract
# but writes scoring/scores_train.json.

set -e

cd /home/user

mkdir -p scoring

INPUT_DIR="${1:-checkpoints/core_test}"
python -m eval.train.evaluate_core_test --input-dir "$INPUT_DIR" --scores scoring/scores_train.json
