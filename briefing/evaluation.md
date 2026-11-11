# Evaluation

## Interface

A method is a solver script `scripts/solvers/<name>.sh`, called as

    bash scripts/solvers/<name>.sh <instances_dir> <outputs_dir>

`<instances_dir>/manifest.json` lists instance files. Each instance gives `branching` (B),
`depth` (L), `num_agents` (n), `embedding_dim`, `rho`, `token_vectors` (L x B x d),
`agent_vectors` (n x d) and `agent_token_logprobs` (n x L x B): log pi_i(a | s) for every prefix s
of length t. (pi_i(a|s) = softmax_a(rho * w_i . (z(s) + v_{t,a})) with z(s) the sum of the
prefix's token vectors; z(s) shifts all logits at a prefix equally, so the table is exact for
every prefix.)

For every instance the solver must write `<outputs_dir>/<instance_id>.json` containing

    {"policy": {"": [p_0, ..., p_{B-1}], "0": [...], "0,2": [...], ...}}

a stochastic token policy: next-token probabilities for every prefix (comma-separated token
indices, "" for the empty prefix) reachable with positive probability. The evaluator executes
the policy exactly to obtain the lottery over complete statements. Each row must have B finite,
non-negative entries summing to 1 within 1e-6; at most 100,000 statements may be reachable.
Any other field in the output is ignored.

`scripts/evaluate.sh` / `scripts/evaluate_train.sh` run every solver on disk on the public
training instances (`data/core_synthetic/generate.py`); the hidden test runs every solver on
held-out instances from the same generating process. A solver gets two hours of wall-clock per
evaluation run and must not read evaluator files.

## Experiments and metrics

**empirical_core_test** (B=3, L=4, six participants, d=8, rho on linspace(0.6, 5, 20)).
- `max_coalition_improvement` (primary, lower is better): alpha* = the largest factor by which
  any coalition S, using probability mass |S|/n on any statements, can raise all its members'
  expected utility, maximised over instances. Every coalition's LP is solved over all 81
  statements. 1 means the outcome is in the ex-ante core.
- `core_constraint_satisfied`: 1 if alpha* <= 1 + 1e-6 on every instance.
- `nash_welfare_efficiency`: mean of (NW(p) / NW*)^(1/n), where NW* is the highest Nash welfare
  of any lottery on the instance.

**deep_tree_core_test** (B=3, L=24, six participants, d=8, rho in {0.1, 0.2, 0.4, 0.8}).
- `nash_welfare_ideal_fraction` (primary, higher is better): mean over instances of the
  geometric mean over participants of U_i(p) / max_X U_i(X).
- `mean_log10_coalition_improvement` (lower is better): mean over instances of
  log10(max(1, alpha)), alpha being the largest proportional improvement available to any
  coalition of at most three participants using any statements of the full tree (computed
  exactly).
- `coalition_violation_rate` (lower is better): fraction of instances where that alpha exceeds
  1.001.
- `worst_agent_ideal_fraction`: diagnostic, not ranked.

## Failure rule

A run that crashes, exceeds the time limit, misses an instance, returns an invalid policy, or
fails the core constraint of empirical_core_test on any instance receives the worst value of
every scored metric in both experiments (alpha* = 1e6, indicators and fractions 0,
log10 improvement 12, violation rate 1).

## Baselines

Computed by the evaluator on the same instances and reported next to every solver: the uniform
lottery over all statements, the utilitarian statement (maximising the sum of utilities; found
by beam search on deep trees), and random dictatorship (each participant's favourite statement
with probability 1/n). They are reported with their measured values.
