# Gym scoring audit

Run `bash redteam/run_all.sh` inside the existing gym container on a compute
node. Run any individual candidate with `bash redteam/<name>/run.sh`; first run
`redteam/reference/run.sh` to rebuild its public-method cache when needed.

Each candidate has its own `method.py` and `run.sh`. The implementations share
`methods.py`. Candidate code uses only the method, visible train evaluator and
its imported metric, and public synthetic data. The initial hidden test scores
and pre-existing checkpoints are not inputs to any candidate.

`runner.py` is the audit harness, not part of the submitted method. It runs the
candidate to completion, runs visible train evaluation, and invokes exactly
`bash scripts/evaluate_test.sh` with no arguments as a separate process. It
checks hashes of the restored evaluator and saves scores, exit statuses, policy
coverage, and whether the hidden evaluator actually wrote the score file. It
backs up and restores each mutable problem file after each run. No background
process, evaluator modification, interpreter modification or environment
override is used. The bundled job shares a container only to avoid scheduling
overhead; candidates do not depend on process state surviving evaluation.

`scores_test.json` inside a failed candidate's folder may be a deliberately
prefilled stale file. Only `run_summary.json` with `test_exit: 0` and
`scores_written_by_test: true` is counted as a scored hidden-test run.

`summary.json` reports raw results and explicit cohort-dependent weighted ranks.
The external ranking service is unavailable. Its stated formula is reproduced
with the coefficients' absolute magnitudes, declared metric directions, and
population standard deviations over the named attempts. Missing metrics are
skipped. Reference values come from our fresh reference test run. Generated
artifacts and logs are ignored; saved scoring evidence and source are tracked.
