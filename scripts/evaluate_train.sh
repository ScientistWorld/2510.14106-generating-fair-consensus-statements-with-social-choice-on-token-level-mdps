#!/bin/bash
# Visible train-slice evaluation. Takes no arguments: writes the public training instances,
# runs every solver in scripts/solvers/*.sh, and writes scoring/scores_train.json.
set -e
cd /home/user
mkdir -p scoring
python -m eval.train.evaluate --scores scoring/scores_train.json
