#!/bin/bash
# Supplemental evaluator for the paper's deterministic search algorithms on a
# scaled synthetic token tree. Writes scoring/deterministic_search_scores.json.

set -e

cd /home/user

INPUT_DIR="${1:-checkpoints/deterministic_search}"
python -m eval.evaluate_deterministic_search \
  --input-dir "$INPUT_DIR" \
  --scores scoring/deterministic_search_scores.json
