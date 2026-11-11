#!/bin/bash
# Visible train-slice evaluation. Takes no arguments: runs every solver in scripts/solvers/*.sh
# against the public training instances through the query oracle and writes
# scoring/scores_train.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --scores scoring/scores_train.json
