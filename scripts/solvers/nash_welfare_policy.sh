#!/bin/bash
# Reference solver (the paper's method): Nash-welfare lottery over complete statements,
# token chunking when the tree is too large to enumerate, induced stochastic token policy.
# Interface (all solvers): bash scripts/solvers/<name>.sh <instances_dir> <outputs_dir>
set -e
cd /home/user
python -m method.solve --instances "$1" --outputs "$2"
