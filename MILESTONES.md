# Reproduction Milestones

**Current: secondary_claims**

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

### [2026-06-06 22:53] - core_claim_plus
- Final verification confirmed the reusable scoring pipeline passes `python validate.py --compare` with train/test split checks enabled.
- Removed an unverified secondary experiment from active code, reference scoring, and public target documentation so the gym only exposes completed scored results.
- Confirmed the milestone remains `core_claim_plus` for the verified empirical core-test reproduction.

### [2026-06-06 23:45] - core_claim_plus
- Final verification confirmed the official scored benchmark remains the verified empirical core-test reproduction.
- Removed an unverified supplemental deterministic-search path from active scripts and source files so the gym exposes only completed scored results.
- Confirmed `python validate.py --compare` passes with the core benchmark and train/test split checks enabled.

### [2026-11-11 06:30] - core_claim_plus
- Hardening (final iteration): the deep-tree experiment now uses prefix-dependent 12-token trees (531,441 statements) seen through a query-budgeted oracle (1,500 of 265,720 prefixes); the evaluator holds the full tree and scores Nash welfare against the exact optimum and exact blocking by every coalition.
- The paper's Section 4.2 core test (B=3, L=4, six agents, rho on linspace(0.6, 5, 20)) is served through the same oracle and is a hard gate; the reference stays at alpha* = 1 on all 100 held-out instances while uniform (9.90), utilitarian (2.75e5) and random dictatorship (1.21) are blockable.
- Reference (Algorithm 1 + Section 4.3 chunking) on held-out deep trees: efficiency 0.772, mean log10 alpha* 0.300; better variants +0.22 (40/40 paired wins), coarser chunking -0.058 (39/40 losses), seed spread ~1e-12; every shortcut fails or ties. Details in HARDENING.md.

### [2026-11-17 06:00] - secondary_claims
- Final hardening: added `lm_consensus_search`, the paper's Section 6.3 single-statement experiment on token trees from a real small language model (Qwen2.5-1.5B-Instruct prompted per opinion, B=2, 16-token statements; the paper's three scenarios plus seven authored in train, twenty held-out authored scenarios in test; 150 queries per instance), scored against the exact best statement of each tree.
- The reference runs the paper's beam search (Algorithm 3, width 4): held-out EPPL 5.73 vs finite lookahead 7.73, Best-of-4 13.37, Habermas-style vote 14.07, base-model statement 24.25, the paper's ordering (Table 10). Better searches within the budget gain 0.082 log-EPPL regret (95% CI [0.036, 0.137], 9-10 wins, 0 losses); brute force and Best-of-many stay below the reference.
- Held-out per-instance values are no longer committed (they revealed hidden optima). Weights 0.2 / 0.4 / 0.4. Details in HARDENING.md.

## Stop Justification
- Completed at milestone `secondary_claims`: the core claim (Nash-welfare lottery in the ex-ante core, with the deep-tree tractability study) and the secondary empirical claim (egalitarian-welfare search beats sampling and Habermas-style baselines on worst-case alignment) are reproduced; the opinion-length scaling and credit-assignment studies (Sections 6.1-6.2) are not.
