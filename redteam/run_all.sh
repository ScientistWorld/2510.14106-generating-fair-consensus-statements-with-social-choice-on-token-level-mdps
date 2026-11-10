#!/bin/bash
set -uo pipefail
cd /home/user
failed=0
bash redteam/reference/run.sh || failed=1
bash redteam/reference_seed_11/run.sh || failed=1
bash redteam/reference_seed_22/run.sh || failed=1
bash redteam/reference_seed_33/run.sh || failed=1
bash redteam/better_core_fairness/run.sh || failed=1
bash redteam/weaker_uniform/run.sh || failed=1
bash redteam/weaker_one_iteration/run.sh || failed=1
bash redteam/maxmin_unconstrained/run.sh || failed=1
bash redteam/exhaustive_single_leaf/run.sh || failed=1
bash redteam/honest_exhaustive_optimizer/run.sh || failed=1
bash redteam/shortcut_grid_replay/run.sh || failed=1
bash redteam/shortcut_partial_coverage/run.sh || failed=1
bash redteam/shortcut_empty_results/run.sh || failed=1
bash redteam/shortcut_filter_methods/run.sh || failed=1
bash redteam/shortcut_generator_replacement/run.sh || failed=1
bash redteam/shortcut_scaled_mass/run.sh || failed=1
bash redteam/shortcut_fake_artifact_fields/run.sh || failed=1
bash redteam/shortcut_score_prefill/run.sh || failed=1
bash redteam/shortcut_move_baselines/run.sh || failed=1
bash redteam/shortcut_invalid_nan/run.sh || failed=1
bash redteam/shortcut_invalid_null/run.sh || failed=1
bash redteam/shortcut_invalid_zero/run.sh || failed=1
bash redteam/shortcut_invalid_empty/run.sh || failed=1
bash redteam/shortcut_constraint_tradeoff_a/run.sh || failed=1
bash redteam/shortcut_constraint_tradeoff_b/run.sh || failed=1
bash redteam/shortcut_easy_instances/run.sh || failed=1
python -m redteam.summarize
exit "$failed"
