#!/bin/bash
# The paper's baselines (uniform and utilitarian lotteries) and a random-dictatorship lottery are
# computed by the evaluator itself on the same instances, so they cannot be altered by a solver.
# They are reported in every scores file next to each solver.
set -e
cd /home/user
bash scripts/evaluate_train.sh
