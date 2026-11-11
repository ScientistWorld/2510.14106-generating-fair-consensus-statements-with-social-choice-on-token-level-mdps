# Evaluation

## Interface

A method is a solver script `scripts/solvers/<name>.sh`. The evaluator starts it with no
arguments and talks to it with one JSON object per line: evaluator -> solver on the solver's
stdin, solver -> evaluator on its stdout (anything else must go to stderr, which is logged to
`logs/solvers_<split>/`). `data/core_synthetic/oracle_client.py` implements the solver side;
`scripts/solvers/nash_welfare_policy.sh` is an example.

For every instance, in a fixed order:

1. The evaluator sends `{"type": "instance", "instance_id", "experiment", "branching" (B),
   "depth" (L), "num_agents" (n), "rho", "query_budget"}`.
2. The solver may send `{"type": "query", "prefixes": [[a_1, ..., a_t], ...]}` (0 <= t < L,
   tokens in 0..B-1) any number of times. The reply is
   `{"type": "logprobs", "logprobs": [n x B per prefix], "queries_used": k}`:
   log pi_i(a | s) for every participant i and token a at each prefix s. Each distinct prefix
   costs one query; repeating one is free. A batch that is invalid or would exceed the budget
   is refused with `{"type": "error", ...}` and costs nothing.
3. The solver sends `{"type": "policy", "policy": {"": [p_0, ..., p_{B-1}], "0": [...],
   "0,2": [...], ...}}`: next-token probabilities for every prefix (comma-separated token
   indices, "" for the empty prefix) reachable with positive probability. The evaluator replies
   `{"type": "accepted"}` and moves on; after the last instance it sends `{"type": "done"}`.

The evaluator executes the policy exactly to obtain the lottery over complete statements. Each
row must have B finite, non-negative entries summing to 1 within 1e-6. Any other field is
ignored. The evaluator holds the full trees; a solver sees only what it queries.

`scripts/evaluate.sh` / `scripts/evaluate_train.sh` run every solver on disk on the public
training instances (`data/core_synthetic/generate.py`); the hidden test runs every solver on
held-out instances from the same generating process. The time the evaluator spends waiting for
a solver is limited to one hour per evaluation run (the reference needs about ten seconds).

## Experiments and metrics

**empirical_core_test** (paper Section 4.2: B=3, L=4, six participants, d=8, rho on
linspace(0.6, 5, 20); budget 40 = every prefix).
- `max_coalition_improvement` (primary, lower is better): alpha* = the largest factor by which
  any coalition S, using probability mass |S|/n on any statements, can raise all its members'
  expected utility, maximised over instances. Every coalition's LP is solved over all 81
  statements. 1 means the outcome is in the ex-ante core.
- `core_constraint_satisfied`: 1 if alpha* <= 1 + 1e-6 on every instance.
- `nash_welfare_efficiency`: mean of (NW(p) / NW*)^(1/n), where NW* is the highest Nash welfare
  of any lottery on the instance.

**deep_tree_core_test** (B=3, L=12, six participants, d=8, a fresh token vector at every
prefix, rho in {0.4, 0.8, 1.6, 3.2}; budget 1,500 of 265,720 prefixes).
- `nash_welfare_efficiency` (primary, higher is better): mean over instances of
  (NW(p) / NW*)^(1/n), NW* being the highest Nash welfare of any lottery over all 531,441
  statements (computed exactly by the evaluator, certified to 1e-10).
- `mean_log10_coalition_improvement` (lower is better): mean over instances of log10(alpha*),
  alpha* computed exactly for every coalition over every statement of the full tree (0 = in the
  ex-ante core of the full tree).
- `coalition_violation_rate` (lower is better): fraction of instances where alpha* > 1.001.
- `worst_agent_ideal_fraction`: diagnostic, not ranked.

## Failure rule

A run that crashes, exceeds the time limit, breaks the protocol, misses an instance, returns an
invalid policy, or fails the core constraint of empirical_core_test on any instance receives the
worst value of every scored metric in both experiments (alpha* = 1e6, indicators and
efficiencies 0, log10 improvement 12, violation rate 1).

## Baselines

Computed by the evaluator on the same instances and reported next to every solver: the paper's
uniform lottery over all statements and utilitarian statement (maximising the sum of
utilities, exact), and random dictatorship over the participants' greedy statements (each
participant's argmax-token continuation, probability 1/n each). Their numbers are measured, not
masked.
