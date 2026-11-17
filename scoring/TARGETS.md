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

Fair outcomes on prefix-dependent statement trees seen through a query budget, judged against
the full tree.

- **Primary (benefit) metric:** `nash_welfare_efficiency` (higher is better; 1 = the best
  lottery over the whole tree)
- **Constraint metrics:** `mean_log10_coalition_improvement` and `coalition_violation_rate`
  (lower is better; 0 means no coalition can block using any statement of the tree)
- **Diagnostic:** `worst_agent_ideal_fraction` (not ranked)
- **Reference:** the paper's chunked method reaches 0.772 efficiency, mean log10 alpha* 0.300 and
  violation rate 0.95 on the held-out slice (no paper number exists for this setting).

## Primary: lm_consensus_search

A single consensus statement on language-model statement trees (paper Section 6.3), judged
against the best statement of the full tree.

- **Primary metric:** `log_eppl_regret` (lower is better; 0 = the statement with the lowest
  egalitarian perplexity on every instance)
- **Secondary metric:** `egalitarian_perplexity` (lower is better; the paper's EPPL)
- **Diagnostics:** `optimal_statement_rate`, `base_perplexity` (not ranked)
- **Constraints (enforced, not scored):** one statement per instance (a deterministic policy);
  at most 150 prefix queries per instance.
- **Paper result (Table 10, Llama 3.1 8B, up to 50 tokens):** EPPL beam search 2.87 < finite
  lookahead 4.18 < Best-of-N 6.35 < prompted Habermas Machine 9.67 < Habermas Machine 15.69.
- **Reference (held-out slice, this gym's scale):** the paper's beam search (width 4) reaches
  regret 0.104 / EPPL 5.73; the paper's finite lookahead 0.341 / 7.73; Best-of-4 0.811 / 13.37;
  Habermas-style vote 0.848 / 14.07; the consensus-prompted model's own statement 1.383 / 24.25.
  The best statement of each tree has mean EPPL 5.13.

## How to run

- `bash scripts/evaluate_train.sh` scores every solver in `scripts/solvers/` on the public
  instances and writes `scoring/scores_train.json` (`scripts/evaluate.sh` writes the same to
  `scoring/scores.json`).
- `bash scripts/evaluate_test.sh` (no arguments) scores every solver on the held-out instances
  and writes `scoring/scores_test.json`.
