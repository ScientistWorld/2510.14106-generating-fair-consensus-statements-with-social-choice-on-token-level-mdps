# Evaluation Targets

## Constraint: empirical_core_test

Tests whether each produced stochastic outcome is resistant to proportional coalitional blocking in the paper's synthetic token-tree setting. The Nash-welfare policy should satisfy the core constraint, while the uniform and utilitarian reference policies are included to expose blockable outcomes on the same instances.

- **Primary metric:** `max_coalition_improvement` (lower is better; values above 1 violate the core condition)
- **Constraint metric:** `core_constraint_satisfied` (higher is better; 1 means the core constraint is satisfied)
- **Trend metrics:** `rho_0_max_coalition_improvement`, `rho_1_max_coalition_improvement`, `rho_2_max_coalition_improvement`, `rho_4_max_coalition_improvement`, and `rho_8_max_coalition_improvement` preserve the paper's polarization axis.
- **How to run:** `bash scripts/evaluate.sh checkpoints/core_test`
