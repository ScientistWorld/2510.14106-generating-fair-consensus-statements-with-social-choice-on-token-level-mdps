#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# This job: the full hardening audit (hardening/run_job.sh): exactness and language-model
# self-tests, the language-model search calibration, the reference alone on train and test (the
# official scores), every audit variant and shortcut through the no-argument hidden test and the
# train evaluation, the workspace-mutation probe, summary, validation.
set -u
cd /home/user
trap 'python -m hardening.install_solvers uninstall >/dev/null 2>&1' EXIT
bash hardening/run_job.sh
