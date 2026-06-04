# Progress

## What Works

Implemented the paper's empirical core-test pipeline for the synthetic token-tree setting in Section 4.2. The method builds normalized vector agent policies, enumerates complete leaves, solves the Nash-welfare lottery, derives the prefix stochastic policy, and writes artifacts for independent evaluation.

Evaluation code reads those artifacts and solves the paper's coalition-blocking LP to compute maximum coalition improvement.

## Results

No scheduler result yet. A CPU job is being submitted to install dependencies in the container and run `scripts/reproduce.sh`.

## Remaining

- Run the submitted CPU job and inspect `scoring/scores.json`
- If the core constraint holds, update milestone to `core_claim`
- If budget allows, add deterministic finite-lookahead or beam-search text-generation experiments

## Issues

The original Habermas and Llama/Gemma experiments require large model weights and datasets not currently present in the workspace. The first executable reproduction targets the paper's synthetic core experiment because it is self-contained and directly tests the formal fairness claim.

## Deviations from Paper

- The submitted run reproduces the synthetic empirical core test rather than the LLM consensus-generation benchmark. This is a scale/resource choice for the first milestone, not an algorithm substitution: the implemented computation is the paper's Nash-welfare lottery and induced stochastic policy.
