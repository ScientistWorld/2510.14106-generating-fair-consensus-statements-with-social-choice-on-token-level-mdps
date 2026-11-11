#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# Current job: the hardening audit (reference, variants and shortcuts through the hidden test). Restored to
# `bash scripts/reproduce.sh` once the audit is complete.
set -e
cd /home/user
bash hardening/run_job.sh
