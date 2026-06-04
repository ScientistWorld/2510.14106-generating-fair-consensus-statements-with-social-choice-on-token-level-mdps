# Research Direction

## Research Problem

This benchmark studies fair aggregation of multiple text-like preference policies into a collective stochastic outcome. The goal is to improve social-choice fairness in generated consensus outcomes, not to solve a separate language-modeling task.

## Core Contribution

Work should remain within token-level MDP or finite-outcome social-choice aggregation. Improvements should change how participant utilities over generated continuations are combined into a fair outcome.

## Approach Scope

Scientists may change optimization methods, approximations, search procedures, or lottery construction as long as the artifact remains a distribution over generated leaves and is evaluated by the same coalition-blocking test.

## Out of Bounds

Replacing the benchmark with a classifier, a learned surrogate predictor of the score, or a text generator that bypasses the utility-and-lottery structure is out of scope. Reporting a deterministic single leaf as if it were a fair lottery is also out of scope unless the resulting distribution is explicitly evaluated by the same blocking metric.
