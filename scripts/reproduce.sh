#!/bin/bash
# Reproduce the paper's results end-to-end.
#
# This script runs inside the compute container and should call
# scripts/baseline.sh and scripts/method.sh to run baselines and
# the paper's method, each of which calls scripts/evaluate.sh.
#
# Assumes:
#   - Container is built (environment/container.sif)
#   - Data and models are downloaded (via scripts/download.sh)
#   - Working directory is /home/user
#
# This is the "gold standard" reproduction — 
# You do not need to run this, but keep this in good shape
# so other agents can easily run a full loop of reproduction and get the paper's numbers.

set -e

cd /home/user

bash scripts/method.sh
python validate.py --compare

# Supplemental deterministic single-statement search smoke. This implements
# the paper's Algorithms 2 and 3 on the same synthetic token-level MDP family,
# but writes a separate report because the paper's EPPL reference numbers use
# Llama/Habermas data rather than this synthetic setup.
python -m method.run_deterministic_search
bash scripts/evaluate_deterministic_search.sh checkpoints/deterministic_search
