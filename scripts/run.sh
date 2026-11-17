#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# Reproduces the reference end to end: the paper's methods and the evaluator baselines on the
# public training slice and the held-out test slice (all three experiments), then validation.
# The language-model token trees are committed; rebuilding them needs a GPU and the model weights
# (scripts/download.sh, then python -m data.lm_consensus.build_trees ...). The hardening audit
# that checked variants and shortcuts is hardening/run_job.sh.
set -e
cd /home/user
bash scripts/reproduce.sh
