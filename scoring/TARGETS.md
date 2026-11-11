# Evaluation Targets

## Constraint: empirical_core_test

Every stochastic outcome must lie in the ex-ante core on every small-tree instance: no
coalition can use its proportional share of probability to make all its members better off.
Violating this on any instance fails the run in every experiment.

- **Primary metric:** `max_coalition_improvement` (lower is better; must be 1)
- **Constraint metric:** `core_constraint_satisfied` (must be 1)
- **Benefit metric:** `nash_welfare_efficiency` (higher is better; 1 is the best achievable)
- **Paper result:** the Nash-welfare lottery stays at alpha* = 1 for every polarization level,
  while uniform and utilitarian outcomes become blockable (Figure 2).

## Primary: deep_tree_core_test

Fair outcomes on statement trees too large to enumerate, judged against the full tree.

- **Primary metric:** `nash_welfare_ideal_fraction` (higher is better)
- **Constraint metrics:** `mean_log10_coalition_improvement` and `coalition_violation_rate`
  (lower is better; 0 means no coalition of up to three participants can block)
- **Diagnostic:** `worst_agent_ideal_fraction` (not ranked)

## How to run

- `bash scripts/evaluate_train.sh` scores every solver in `scripts/solvers/` on the public
  instances and writes `scoring/scores_train.json` (`scripts/evaluate.sh` writes the same to
  `scoring/scores.json`).
- `bash scripts/evaluate_test.sh` (no arguments) scores every solver on the held-out instances
  and writes `scoring/scores_test.json`.
