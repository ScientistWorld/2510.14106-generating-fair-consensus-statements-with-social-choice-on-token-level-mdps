# Progress

## What Works

Implemented the paper's empirical core-test pipeline for the synthetic token-tree setting in Section 4.2. The method builds normalized vector agent policies, enumerates complete leaves, solves the Nash-welfare lottery, derives the prefix stochastic policy, and writes artifacts for independent evaluation.

Evaluation code reads those artifacts and solves the paper's coalition-blocking LP to compute maximum coalition improvement.

## Results

Job `449b4d8c-419` generated all synthetic core-test method artifacts under `checkpoints/core_test`, but failed during evaluation. The evaluator tried to compute coalition-improvement diagnostics for blockable baseline policies; one coalition LP was correctly unbounded, which crashed the job before scores were written.

The evaluator now scores only `nash_welfare_policy`, matching `scoring/reference.json`. Baseline diagnostics are not required for the current target and should not block the core-claim score.

## Remaining

- Run the submitted CPU job and inspect `scoring/scores.json`
- If the core constraint holds, update milestone to `core_claim`
- If budget allows, add deterministic finite-lookahead or beam-search text-generation experiments

## Issues

- The original Habermas and Llama/Gemma experiments require large model weights and datasets not currently present in the workspace. The first executable reproduction targets the paper's synthetic core experiment because it is self-contained and directly tests the formal fairness claim.
- The first submitted job exposed an evaluator bug for unbounded baseline coalition-improvement LPs. This was fixed by aligning evaluated methods with `reference.json`.

## Deviations from Paper

- The submitted run reproduces the synthetic empirical core test rather than the LLM consensus-generation benchmark. This is a scale/resource choice for the first milestone, not an algorithm substitution: the implemented computation is the paper's Nash-welfare lottery and induced stochastic policy.
