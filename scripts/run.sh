#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# Current job: smoke tests and calibration of the redesigned evaluator. Restored to
# `bash scripts/reproduce.sh` once the audit is complete.
set -e
cd /home/user
bash hardening/smoke_job.sh
