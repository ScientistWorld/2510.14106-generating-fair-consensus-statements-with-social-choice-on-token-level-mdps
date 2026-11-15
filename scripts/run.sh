#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# This job: build the language-model token trees (GPU), then the full hardening audit
# (hardening/run_job.sh: self-tests, calibration, reference-only scores, every variant and
# shortcut through the hidden test, mutation probe, summary, validation).
set -u
cd /home/user
mkdir -p hardening/results hardening/logs
LOG=hardening/results/build_steps.log
step() { echo "=== $(date '+%F %T') $1" | tee -a $LOG; }
: > $LOG
step "environment"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tee -a $LOG
nproc | tee -a $LOG
if ! python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 3)"; then
    echo "NO CUDA DEVICE: tree building needs a GPU; stopping" | tee -a $LOG
    exit 3
fi
step "build train trees"
python -m data.lm_consensus.build_trees --scenarios data/lm_consensus/scenarios_train.json \
    --out data/lm_consensus/trees > hardening/logs/build_train.log 2>&1; echo "rc=$?" | tee -a $LOG
step "build test trees"
python -m data.lm_consensus.build_trees --scenarios eval/test/lm_scenarios.json \
    --out eval/test/lm_trees > hardening/logs/build_test.log 2>&1; echo "rc=$?" | tee -a $LOG
cp hardening/logs/build_*.log hardening/results/ 2>/dev/null
du -sh data/lm_consensus/trees eval/test/lm_trees | tee -a $LOG
step "audit"
bash hardening/run_job.sh
step "done"
