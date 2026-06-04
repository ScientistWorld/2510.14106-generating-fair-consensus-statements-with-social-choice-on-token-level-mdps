# Progress

## What Works

Implemented the paper's empirical core-test pipeline for the synthetic token-tree setting in Section 4.2. The method builds normalized vector agent policies, enumerates complete leaves, solves the Nash-welfare lottery, derives the prefix stochastic policy, and writes artifacts for independent evaluation.

Evaluation code reads those artifacts and solves the paper's coalition-blocking LP to compute maximum coalition improvement.

The end-to-end reproduction script now runs successfully in the compute container.

The evaluator now scores every policy distribution saved in the artifacts, including the paper's uniform and utilitarian reference policies, while keeping `eval/` independent from `method/`.

## Results

Job `63da2914-c4d` completed successfully. The reproduced empirical core-test score matches the paper's reported core constraint: `max_coalition_improvement = 1.0` and `core_constraint_satisfied = 1.0` for `nash_welfare_policy`.

The run covered 25 instances: five polarization levels (`rho = 0, 1, 2, 4, 8`) and five seeds.

Pending refresh: a CPU test job is being submitted to regenerate `scoring/scores.json` with uniform and utilitarian baseline coalition-improvement scores included alongside the Nash-welfare policy.

## Remaining

- The deterministic finite-lookahead and beam-search LLM consensus-generation experiments remain for a higher milestone.
- The opinion-length and credit-assignment experiments remain for a higher milestone.

## Issues

- The original Habermas and Llama/Gemma experiments require large model weights and datasets not currently present in the workspace. The first executable reproduction targets the paper's synthetic core experiment because it is self-contained and directly tests the formal fairness claim.
- The first submitted job exposed an evaluator bug for unbounded baseline coalition-improvement LPs. This was fixed by aligning evaluated methods with `reference.json`.

## Deviations from Paper

- The submitted run reproduces the synthetic empirical core test rather than the LLM consensus-generation benchmark. This is a scale/resource choice for the first milestone, not an algorithm substitution: the implemented computation is the paper's Nash-welfare lottery and induced stochastic policy.
