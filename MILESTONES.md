# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 00:00] - core_claim
- Audited the existing Section 4.2 empirical core-test reproduction.
- Confirmed the implementation contains the paper's actual synthetic token-tree policies, Nash-welfare lottery optimization, induced stochastic prefix policy, and independent coalition-blocking LP evaluation.
- Kept the milestone at `core_claim` while a CPU test job refreshes `scores.json` with the expanded baseline and per-polarization metrics needed to restore `core_claim_plus`.

### [2026-06-04 10:27] - core_claim_plus
- Job `bf64dcb8-621` completed successfully in the container with exit code 0.
- Regenerated `scoring/scores.json` for 25 synthetic token-tree instances: five seeds across `rho = 0, 1, 2, 4, 8`.
- Restored `core_claim_plus`: `nash_welfare_policy` has maximum coalition improvement 1.0 at every polarization level, while `uniform_policy` and `utilitarian_policy` become blockable as polarization increases.

## Stop Justification
- Stopped at `core_claim_plus`. The formal stochastic core claim and its polarization/baseline comparison are reproduced in a reusable, validated CPU environment. Higher milestones would require the LLM Habermas/EPPL experiments, including large instruction-tuned model weights and external datasets not packaged in this workspace.
