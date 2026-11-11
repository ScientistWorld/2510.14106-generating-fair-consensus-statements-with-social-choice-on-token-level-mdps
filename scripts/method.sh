#!/bin/bash
# Run the paper's method (scripts/solvers/nash_welfare_policy.sh) on the public training
# instances and score it, together with every other solver on disk and the baselines.
set -e
cd /home/user
bash scripts/evaluate.sh
