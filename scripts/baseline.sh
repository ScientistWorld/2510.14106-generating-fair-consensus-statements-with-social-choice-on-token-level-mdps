#!/bin/bash
# The paper's baselines are computed by the evaluator itself on the same instances, so they cannot
# be altered by a solver: uniform and utilitarian lotteries and a random-dictatorship lottery on
# the core tests; on lm_consensus_search, the base model's greedy statement, Best-of-4 and a
# Habermas-style Schulze vote among 4 base-model samples.
# They are reported in every scores file next to each solver.
set -e
cd /home/user
bash scripts/evaluate_train.sh
