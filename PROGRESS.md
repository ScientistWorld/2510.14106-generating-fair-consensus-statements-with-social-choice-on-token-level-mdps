# Progress

## What Works

Implemented the paper's empirical core-test pipeline for the synthetic token-tree setting in Section 4.2. The method builds normalized vector agent policies, enumerates complete leaves, solves the Nash-welfare lottery, derives the prefix stochastic policy, and writes artifacts for independent evaluation.

Evaluation code reads those artifacts and solves the paper's coalition-blocking LP to compute maximum coalition improvement.

Earlier end-to-end reproduction jobs ran successfully in the compute container. The latest continuation audit found and fixed a numerical conditioning bug in the evaluator; the follow-up smoke job is pending.

The evaluator now scores every policy distribution saved in the artifacts, including the paper's uniform and utilitarian reference policies, while keeping `eval/` independent from `method/`.

The coalition LP now guards against zero-denominator numerical underflow by using the evaluator tolerance in the ratio denominator, and it rescales each blocking constraint so alpha has a stable unit coefficient. The default Section 4.2 softmax utilities are positive, but this keeps the reusable evaluator well-defined for future finite-outcome submissions.

## Results

Job `63da2914-c4d` completed successfully. The reproduced empirical core-test score matches the paper's reported core constraint: `max_coalition_improvement = 1.0` and `core_constraint_satisfied = 1.0` for `nash_welfare_policy`.

The run covered 25 instances: five polarization levels (`rho = 0, 1, 2, 4, 8`) and five seeds.

Job `bf64dcb8-621` completed successfully after the evaluator expansion. It regenerated `scoring/scores.json` with uniform and utilitarian baseline coalition-improvement scores included alongside the Nash-welfare policy.

The scoring schema now includes fixed per-polarization metrics for `rho = 0, 1, 2, 4, 8`. Exact paper reference values are listed only for the Nash-welfare policy because the paper reports the baseline core-test curves qualitatively rather than as a numeric table.

Final reproduced core-test summary:

- `nash_welfare_policy`: max coalition improvement `1.0`; core constraint satisfied at every polarization level.
- `uniform_policy`: max coalition improvement `13.6528`; blockable for polarized settings.
- `utilitarian_policy`: max coalition improvement `14782312.7486`; strongly blockable for polarized settings.

Continuation audit job `449b4d8c-419` failed while rerunning `scripts/reproduce.sh` because HiGHS reported an unbounded coalition LP for a regenerated baseline artifact. The LP was algebraically bounded, but the original formulation used very small expected-utility coefficients on alpha; the evaluator now uses the equivalent scaled form `alpha <= U_i(q) / U_i(p)`.

## Remaining

- The deterministic finite-lookahead and beam-search LLM consensus-generation experiments remain for a higher milestone.
- The opinion-length and credit-assignment experiments remain for a higher milestone.

## Issues

- The original Habermas and Llama/Gemma experiments require large model weights and datasets not currently present in the workspace. The first executable reproduction targets the paper's synthetic core experiment because it is self-contained and directly tests the formal fairness claim.
- The paper's public GitHub repository was checked at commit `27abe654c72e465ea58213c80cfb8416573ac2cc`. Its full text-generation experiment stack requires Together and OpenAI API credentials, so it is not suitable for this no-paid-API reusable benchmark without substantial local-model replacement work.
- The first submitted job exposed an evaluator bug for unbounded baseline coalition-improvement LPs. This was fixed by aligning evaluated methods with `reference.json`.
- The provided external job-log path was not mounted at that absolute location, but the same logs were available under `jobs/bf64dcb8-621/` and were reviewed.
- Continuation job `449b4d8c-419` exposed a second numerical conditioning issue in the coalition LP. The evaluator was patched to validate input distributions, renormalize harmless roundoff, and solve the scaled blocking LP.

## Deviations from Paper

- The submitted run reproduces the synthetic empirical core test rather than the LLM consensus-generation benchmark. This is a scale/resource choice for the first milestone, not an algorithm substitution: the implemented computation is the paper's Nash-welfare lottery and induced stochastic policy.
