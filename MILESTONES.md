# Reproduction Milestones

**Current: core_claim_plus**

## Progress Log

### [2026-06-04 10:31] - method_runs
- Implemented the Section 4.2 synthetic token-tree environment with branching factor 3, depth 4, normalized token vectors, normalized agent vectors, and polarization-controlled softmax agent policies.
- Implemented the paper's Nash-welfare lottery over complete leaves and induced stochastic prefix policy from conditional leaf masses.
- Added a reusable evaluator that reads saved artifacts and computes the coalition-blocking LP without importing from `method/`.

### [2026-06-04 10:31] - core_claim
- Reproduced the empirical ex-ante-core claim on 25 instances across five polarization values and five seeds.
- The Nash-welfare policy satisfies `max_coalition_improvement = 1.0` across the polarization grid, while uniform and utilitarian reference policies become blockable in polarized settings.
- Packaged scripts, scoring metadata, and environment setup for rerunning the core-test reproduction.

### [2026-06-04 10:35] - core_claim
- Audited the continuing workspace and confirmed the method code implements the paper's Section 4.2 Nash-welfare lottery, not a surrogate.
- The latest CPU smoke job (`449b4d8c-419`) exposed a numerical conditioning failure in the independent coalition LP evaluator for a regenerated baseline artifact.
- Rescaled the LP constraints by each agent's current utility so alpha has a stable unit coefficient; a follow-up CPU smoke job is required before restoring `core_claim_plus`.

### [2026-06-04 10:37] - core_claim_plus
- Follow-up CPU smoke job (`910a202e-d02`) reran `scripts/reproduce.sh` successfully in the compute container.
- The reproduced Nash-welfare policy satisfies the empirical core constraint for all 25 generated instances across five polarization values and five seeds.
- Uniform and utilitarian reference policies remain blockable in polarized settings, so the additional rho/seed coverage supports the paper's comparative core-test claim.

## Stop Justification
- The packaged environment reaches `core_claim_plus` for the self-contained stochastic-policy fairness claim.
- Higher milestones would require reproducing the LLM consensus-generation beam/lookahead experiments; the paper's public code path uses Together/OpenAI API clients, which are out of scope for this no-paid-API benchmark.
