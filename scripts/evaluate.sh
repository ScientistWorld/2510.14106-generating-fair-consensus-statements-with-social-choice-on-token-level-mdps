#!/bin/bash
# Standard evaluation entry point. Self-contained in eval/ (never imports method/).
# Runs every solver in scripts/solvers/*.sh on the public training instances and writes
# scoring/scores.json. A solver is any executable script with the interface
#   bash scripts/solvers/<name>.sh <instances_dir> <outputs_dir>
# that writes <outputs_dir>/<instance_id>.json = {"policy": {"<prefix>": [p_0, ..., p_{B-1}], ...}}.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --scores scoring/scores.json
