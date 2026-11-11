# Hardening report (iteration 11)

All "after" numbers come from CPU job `656037ed-838`. It ran the exactness self-tests, then the
reference alone on train and test (these are the committed `scoring/scores_*.json`). It then ran
every audit solver through the no-argument `bash scripts/evaluate_test.sh` and
`bash scripts/evaluate_train.sh`, followed by the workspace-mutation probe, `hardening.summarize`
and `validate.py --compare` (all checks pass). Raw files are in `hardening/results/`:
`all_scores_test.json`, `all_scores_train.json`, `reference_scores_*.json`,
`mutation_scores_test.json`, `summary.json` and `summary.md`, plus solver logs. The audit
solvers are in `hardening/variants.py` and are installed by `hardening/install_solvers.py`.
`bash hardening/run_job.sh` reruns the whole audit.

"Before" means the gym the red team audited (`REDTEAM.md`: 25 fixed public instances, a lottery
read from a file, and 8 of 9 scored metrics saturated). Where useful, it also means the first
hardening attempt of this iteration (a 24-token position-only deep tree whose audit job hit the
6-hour limit).

Ranks use the stated formula, `sum weight * |coef| * directed(value - reference) / std`. Here
`std` is the population standard deviation over the honest-variant cohort in
`hardening/summarize.py` (reference, two reruns of the reference with seeded optimizer starts,
three better variants and three weaker ones).

## What changed

1. **Deep experiment rebuilt where the paper's method falls short.**
   - **Why:** the old deep test (24 tokens, policies depending only on position) let one
     column-generation variant reach the bound of both blocking metrics (log10 alpha 0,
     violations 0). Its Nash welfare (0.4159 vs reference 0.4131) was probably optimal too. The
     test's own exactness came from that position-only structure, which a method could exploit
     just as well. The hidden test also took about 2 hours.
   - **New tree:** 12 tokens (531,441 statements), with a fresh token vector at every prefix, so
     next-token distributions depend on the whole prefix, as in language-model trees.
   - **Oracle access:** participants' policies are only available through an oracle owned by the
     evaluator. Each instance allows 1,500 distinct prefix queries out of 265,720. This is the
     tractability regime of paper Section 4.3.
   - **Exact scoring:** the evaluator holds the full tree. It computes the exact Nash-welfare
     optimum by column generation with a certified Frank-Wolfe gap (at most 1e-10 on every
     instance). It computes exact blocking alpha* for **every** coalition over **every**
     statement; the old test only checked coalitions of up to three agents.
   - **Rho grid:** {0.4, 0.8, 1.6, 3.2}.
   - **Checks:** the self-tests compare these computations with brute force on 6,561-statement
     trees. Nash welfare agrees to 9e-12, game values to 1e-8 and alpha* to 1.3e-10
     (`hardening/results/selftest.jsonl`).
2. **Same protocol for the small trees.** The paper's Section 4.2 trees (B=3, L=4, six agents,
   d=8, rho on linspace(0.6, 5, 20)) are served the same way, with a budget of 40, which covers
   every prefix. The core constraint is a hard gate: a blocked outcome on any instance fails the
   run, and the evaluator stops serving that solver.
3. **Hidden inputs.**
   - Test trees come from 128-bit secret seeds and a private generator copy in
     `eval/test/instances.py`.
   - They exist only in the evaluator's memory; no instance file is written. The previous
     attempt had left the hidden test instances in `tmp/`; they were deleted.
   - Solvers see only the answers to the queries they make.
4. **Everything recomputed.** Only the `policy` field is read. The evaluator executes the
   stochastic token policy exactly, with strict checks: finite, non-negative, each row summing to
   1 within 1e-6.
5. **Failures.** A crash, timeout, protocol break, missing instance, invalid policy or core
   violation gives the worst value in every scored metric of both experiments. Every solver on
   disk is scored. A solver whose name collides with a baseline is renamed, not dropped.
6. **Baselines fixed by `eval/`.**
   - The paper's uniform and utilitarian outcomes are computed exactly on the full tree.
   - Random dictatorship now uses each participant's *greedy* statement. Built from exact
     favourites, it needs information no budgeted method has, and it beat the budgeted
     reference at high polarization (smoke run, `hardening/results/smoke/`).
7. **Light hidden test.** Each tree is built once and serves every solver. Solver time counts
   only the evaluator's waiting time. The reference-only hidden test takes **325 s** (it was about
   2 h before). The limit is one hour of waiting per solver per run; the reference needs 10 s.

The reproduced method is unchanged in substance: the paper's Algorithm 1 (Frank-Wolfe on the
Nash-welfare program, Definition 2's induced policy), with Section 4.3 chunking when the budget
cannot cover the tree.
- **Chunk size:** the finest chunk size whose worst-case query cost fits the budget. That is 4,
  giving three chunks and at most 343 statements.
- **Chunk proposals:** each participant's greedy block plus the participants' mixture's greedy
  block. The paper does not specify how chunks are proposed.

## Do better results score higher?

**Before:**
- The red team's genuinely better variant gained +0.065 and already sat at the bound of every
  scored metric.
- An exhaustive optimizer tied the reference.
- In the first rebuild, column generation reached the bound of the deep blocking metrics.

**After (hidden test, 40 deep instances, paired against the reference on identical instances):**

| variant | deep NW efficiency | mean log10 alpha* | violation rate | rank | paired NW gain, 95% CI | wins/losses (eff.) |
|---|---:|---:|---:|---:|---:|---:|
| better_wide_colgen | 0.9971 | 0.0066 | 0.175 | **+1.104** | +0.225 [0.182, 0.270] | 40/0 |
| better_budgeted_colgen | 0.9907 | 0.0167 | 0.325 | **+0.956** | +0.218 [0.175, 0.264] | 40/0 |
| better_chunked_colgen | 0.9906 | 0.0167 | 0.350 | **+0.933** | +0.218 [0.175, 0.263] | 40/0 |
| **reference** (nash_welfare_policy) | 0.7724 | 0.2999 | 0.950 | 0 | — | — |
| weaker_coarse_chunks (chunk size 6) | 0.7140 | 0.3511 | 1.000 | **−0.146** | −0.058 [−0.073, −0.044] | 0/39 |
| weaker_one_iteration (1 Frank-Wolfe step) | fails core gate (alpha* 1.163) | | | **−5.126** | | |
| weaker_uniform (uniform over the chunked set) | fails core gate (alpha* 1.166) | | | **−5.126** | | |

**Seed noise:**
- The reference was rerun with seeded random Frank-Wolfe starts (`reference_seed_11`,
  `reference_seed_22`).
- Every scored number is identical: paired differences are at most 4e-16 in efficiency and
  4e-12 in log10 alpha*, and rank differences are below 1e-6.
- The instances are fixed, the oracle is deterministic, and the Nash-welfare optimum in utility
  space is unique.
- The smallest effect, the weaker variant's 0.058 loss, is more than 10^13 times this spread and
  well outside its bootstrap interval over instances.

**Train vs. test:**
- The same ordering holds on train (20 deep instances): the better variants gain +0.257 to
  +0.263 in efficiency and win 20/20.
- Reference deep efficiency is 0.732 on train and 0.772 on test.

**Paper direction (small trees):**
- Nash-welfare reference alpha* = 1 on all 100 test instances.
- Uniform alpha* = 9.90, utilitarian 2.75e5, random dictatorship 1.21.
- This matches Figure 2: the Nash-welfare policy stays at alpha* = 1 for every polarization
  level, while uniform and utilitarian outcomes become blockable as polarization grows.

## Shortcuts: fix and score before → after

All after-scores are test ranks against the reference (0). A failed run scores −5.126 because it
gets the worst value in every metric.

| shortcut | before | fix | after |
|---|---|---|---|
| grid replay (public answers by nearest rho) | 0, tie | hidden 128-bit-seeded instances served only through the oracle | core violated on the first instance → **−5.126** |
| partial coverage (omit instances) | +0.065 | every instance is served in order; leaving early fails the run | **−5.126** |
| easy instances only | +0.065 | same | **−5.126** |
| empty results | 0, tie (empty `results`) | empty policy is invalid → worst values | **−5.126** |
| filter methods via editable train scores | 0, tie / weak method hidden | every solver on disk is scored; scores files are never read | mutation probe: hidden weak solver listed with worst values |
| generator replacement (edit public generator) | +0.065 | test trees come from the private copy in `eval/test/` | mutation probe: reference test numbers unchanged |
| scaled probability mass (×1e6) | 0, tie | rows must sum to 1 within 1e-6 | **−5.126** |
| NaN / null / zero / empty policy | exit 1, no worst values written | invalid output → worst value in every metric, file written | **−5.126** each |
| forged fields next to a uniform lottery | −2.10 | only `policy` is read | **−5.126** (core gate) |
| score prefill | −2.10 (overwritten) | file is rewritten by the evaluator | mutation probe: overwritten |
| constraint trade-off (mix in uniform) | +178 in cohort T | core violation is a hard gate with worst values everywhere | **−5.126** |
| crash mid-run | — | — | **−5.126** |
| solver named like a baseline (`uniform_policy`) | — | reported as `solver_uniform_policy`; baseline row untouched | **−5.126** |
| brute force: enumerate the tree (depth-first until the budget refuses) | exhaustive optimizer tied (+6e-13) | 1,500-query budget out of 265,720 prefixes | **−0.945** (efficiency 0.309) |
| seed scan: identify the instance among public seeds, regenerate the whole tree, solve exactly | — | test seeds are 128-bit secrets | train **+1.381** (matched 80/80, exact); test **0.000**, no match, falls back to the reference |

The mutation probe (`hardening/mutation_run.sh`) applied four changes at once, then ran the
no-argument hidden test:
- replaced the public generator so that all participants were identical;
- prefilled a perfect `scores_test.json`;
- stripped `scores_train.json` down to the reference;
- added a weak solver.

The reference's test numbers were byte-for-byte unchanged, and the weak solver appeared with
worst values.

## Shortcuts left open

- **Reading `eval/test/` from a solver.** Solvers run as the same user in the same workspace,
  so a solver could import `eval/test/instances.py` and regenerate the hidden trees. Nothing
  technical stops this; it is closed only by the rules (`scoring/CONSTRAINTS.md`) and the
  separate shortcut grader. The same holds for any attempt to read the evaluator process's
  memory.
- **Seed scanning on train.** The public train trees can be regenerated (seed scan reaches
  efficiency 1.0 on train). That is inherent to a public train split, and it buys nothing on
  test.

## Reference score before and after

- **Before (red-team gym):**
  - Minimum utility 0.010510, at the bound on every other scored metric (alpha* = 1, violation
    rate 0).
  - In the first rebuild: deep Nash welfare relative to ideal 0.4131, log10 alpha 0.0011,
    violation rate 0.125 (blocking by coalitions of at most three agents).
- **After (committed `scoring/scores_test.json`):**
  - Small trees: alpha* = 1, core satisfied, Nash-welfare efficiency 1.
  - Deep trees: Nash-welfare efficiency **0.7724**, mean log10 alpha* **0.2999**, violation rate
    **0.95**, worst-agent ideal fraction 0.121 (diagnostic).
  - Baselines on deep trees: random dictatorship 0.619 / 0.405 / 1.0; uniform 0.188 / 1.410 /
    1.0; utilitarian 0.285 / 3.394 / 0.95.
  - All baselines fail the small-tree core. Their ranks are −1.32, −2.39 and −2.63.

## Headroom check

- **Small trees (paper Section 4.2):** the reference sits at the bound of all three metrics.
  This experiment is the paper's core claim and acts as a hard gate, so headroom comes from the
  deep experiment, which keeps weight 0.6.
- **Deep trees:** the reference is far from every bound (efficiency 0.772, log10 alpha* 0.300,
  violation rate 0.95). No variant reaches a bound.
  - The best, `better_wide_colgen`, uses nearly the whole budget (1,422 queries on average; the
    reference uses 684). It still leaves violations on 17.5% of instances, and on 30% at rho 0.4
    and 0.8.
  - A method exact on every instance would rank about +1.27 against the reference. That is
    about 0.16 above the best variant (+1.10), and that 0.16 is mostly violation rate.
- **Brute force:** enumerating the tree within the budget is far below the reference (−0.945).
- **Copyable baselines:** every simple baseline is clearly below the reference.
- **Weights:** unchanged at 0.4 / 0.6. Coefficients are unchanged except that the deep primary
  is now Nash welfare against the exact optimum (it was against each agent's ideal) and
  blocking is exact for all coalitions.

## What could not be fixed, stated plainly

- **Remaining headroom is modest in Nash welfare.** Budgeted column generation already reaches
  0.99 efficiency. What is left above it is mainly coalitional stability: violation rate and
  log10 alpha*.
  - At rho 3.2 every better variant reaches the optimum. There the optimum is essentially
    random dictatorship over the participants' favourite statements.
  - The 30 instances at rho 0.4 to 1.6 carry the discrimination above column generation.
- **Paper coverage.** The executable gym covers Section 4.2 and the Section 4.3 tractability
  question only. The language-model experiments (EPPL, beam search and finite lookahead for
  single statements, Habermas comparison) are not reproduced, so single-statement search quality
  is untested.
- **Synthetic deep tree.** Its token vectors are independent per prefix, so unqueried parts of
  the tree are unpredictable. Real language-model trees have more structure.
- **Choices the paper leaves open.** The chunk-proposal rule (greedy blocks), the budget (1,500)
  and the deep rho grid are gym choices. The deep reference value in `scoring/reference.json`
  is this gym's reference run, not a paper number.
- **The gate only binds where the full tree is visible.** Full-tree core stability on deep
  trees is not attainable under the budget, so the deep experiment has no hard gate. Its
  blocking metrics are scored, not gated.
