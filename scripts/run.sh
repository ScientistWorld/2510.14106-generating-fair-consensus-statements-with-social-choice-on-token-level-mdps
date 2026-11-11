#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# Reproduces the reference end to end: the paper's method and the evaluator baselines on the
# public training slice and the held-out test slice, then validation.
# (The hardening audit that checked variants and shortcuts is hardening/run_job.sh.)
set -e
cd /home/user
bash scripts/reproduce.sh
