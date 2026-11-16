#!/bin/bash
# Job script (runs inside the compute container, workspace at /home/user, no internet).
# This job (GPU): build the language-model token trees of lm_consensus_search, then the
# language-model self-tests, the search calibration by query budget, and a quick LM-only smoke
# run of the evaluator with the audit variants on the public slice. The full audit
# (hardening/run_job.sh) runs as a separate CPU job once the budget is fixed.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/user
mkdir -p hardening/results hardening/logs
LOG=hardening/results/build_steps.log
step() { echo "=== $(date '+%F %T') $1" | tee -a $LOG; }
trap 'python -m hardening.install_solvers uninstall >/dev/null 2>&1' EXIT
: > $LOG
step "environment"
python -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" 2>&1 | tee -a $LOG
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
n_train=$(ls data/lm_consensus/trees/train_lm_*.npz 2>/dev/null | wc -l)
n_test=$(ls eval/test/lm_trees/test_lm_*.npz 2>/dev/null | wc -l)
echo "trees: train $n_train/10, test $n_test/20" | tee -a $LOG
if [ "$n_train" -ne 10 ] || [ "$n_test" -ne 20 ]; then
    echo "tree building incomplete; stopping" | tee -a $LOG
    exit 4
fi
step "lm selftest"
timeout 1800 python -m hardening.lm_selftest > hardening/results/lm_selftest.jsonl 2> hardening/logs/lm_selftest.err; echo "rc=$?" | tee -a $LOG
step "lm calibration"
timeout 5400 python -m hardening.lm_calibrate > hardening/logs/lm_calibrate.log 2>&1; echo "rc=$?" | tee -a $LOG
step "lm smoke: evaluator, public slice, lm_consensus_search only, reference + LM variants"
python -m hardening.install_solvers install_lm_smoke
timeout 3600 python -m eval.train.evaluate --scores hardening/results/lm_smoke_train.json \
    --experiments lm_consensus_search > hardening/logs/lm_smoke.log 2>&1; echo "rc=$?" | tee -a $LOG
python -m hardening.install_solvers uninstall
cp hardening/logs/lm_*.log hardening/logs/lm_selftest.err hardening/results/ 2>/dev/null
step "done"
