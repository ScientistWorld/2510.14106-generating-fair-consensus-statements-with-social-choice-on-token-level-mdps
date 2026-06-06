# Design Notes

## What I Checked

- Reviewed `scripts/evaluate_train.sh` and `scripts/evaluate_test.sh` side by side for copy-paste mistakes. Train points at `eval.train` and writes `scoring/scores_train.json`; test points at `eval.test` and writes `scoring/scores_test.json`.
- Checked for hardcoded method defaults, train/test cross-imports, forbidden paper-agreement metric names, and premature `scoring/scores_train.json` / `scoring/scores_test.json` artifacts.
- Confirmed the split evaluator reads method-produced policy artifacts rather than importing `method/`.

## What Changed

- Fixed a major artifact-trust shortcut: train/test now reconstruct the fixed benchmark instance from neutral `data.core_synthetic.generate` using each artifact's `(rho, seed)` and expected dimensions, instead of trusting `utilities` embedded in method output.
- Added metadata and leaf checks so a method cannot silently change branching, depth, agent count, embedding dimension, or the leaf set to make the evaluation easier.
- Made split membership depend on the fixed expected rho/seed grid, not on whatever files are present after sorting. Missing, duplicate, or unknown core-test artifacts now fail instead of changing the evaluated slice.
- Added triangulating metrics emitted by both slices: `mean_coalition_improvement`, `coalition_violation_rate`, `distribution_simplex_l1_error`, and `min_agent_expected_utility`.
- Updated `scoring/reference.json` to match the executable empirical core-test benchmark and added descriptions for all active metrics. Removed the stale consensus-generation experiment because the train/test split does not compute those metrics and no corresponding evaluator exists in this gym surface.

## Runtime Concerns For Next Iteration

- I did not run Python evaluators or compute jobs in this design-only pass. The runner should execute both split scripts and verify the new deterministic instance reconstruction matches the existing artifacts.
- `min_agent_expected_utility` is scale-dependent across synthetic instances. Its coefficient is intentionally small; watch for it dominating only if utility scales differ more than expected.
- The test split still uses the same public 25-artifact grid with hidden odd-index scoring logic. This is stronger after fixed membership and utility anchoring, but a larger future hidden grid would further reduce seed memorization risk if the benchmark budget allows.
