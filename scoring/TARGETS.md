# Evaluation Targets

## Constraint: empirical_core_test

Tests whether the produced stochastic outcome is resistant to proportional coalitional blocking in the paper's synthetic token-tree setting.

- **Primary metric:** `max_coalition_improvement` (lower is better; values above 1 violate the core condition)
- **Constraint metric:** `core_constraint_satisfied` (higher is better; 1 means the core constraint is satisfied)
- **How to run:** `bash scripts/evaluate.sh checkpoints/core_test`
