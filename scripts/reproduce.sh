#!/bin/bash
# Reproduce the reference end to end: score the paper's method and the baselines on the public
# training slice and on the held-out test slice, then validate the workspace.
set -e
cd /home/user
bash scripts/evaluate.sh
bash scripts/evaluate_train.sh
bash scripts/evaluate_test.sh
python validate.py --compare
