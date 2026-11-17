# Overview

- **Paper ID:** 2510.14106
- **Title:** Generating Fair Consensus Statements with Social Choice on Token-Level MDPs
- **Domain:** Generative social choice / fair text generation
- **TL;DR:** The paper turns consensus statement generation into a token-level social-choice problem and derives fair stochastic and deterministic decoding procedures.

## Short Summary

The paper represents each participant's opinion as a next-token policy and uses that policy to define token-level rewards. It proves that maximizing Nash welfare over complete generated statements yields a stochastic policy in the ex-ante core. It also evaluates finite-lookahead and beam-search procedures that optimize worst-participant alignment for single consensus statements.

## Key Results

- In the synthetic core test, the Nash-welfare policy remains unblocked with maximum coalition improvement equal to 1, while uniform and utilitarian policies become blockable as polarization grows.
- Enumerating every complete statement becomes infeasible for long statements; the paper restricts attention to chunked statement sets, for which the core guarantee holds only relative to the restricted set and no constant-factor Nash-welfare guarantee exists (Theorem 2). This gym measures that gap on deep prefix-dependent trees seen through a query budget.
- On three Habermas Machine scenarios, beam search reports the best overall EPPL in the main experiment (2.87), followed by finite lookahead (4.18), Best-of-N (6.35), a prompted Habermas Machine (9.67) and the original Habermas Machine statement (15.69). This gym reproduces the comparison on token trees from a real small language model (Qwen2.5-1.5B-Instruct, 16-token statements, the paper's three scenarios plus authored ones), scored against the best statement of each full tree: beam search 5.73 < finite lookahead 7.73 < Best-of-4 13.37 < Habermas-style vote 14.07 < the base model's own statement 24.25 on the held-out scenarios.

---

# Problem Definition

## Research Question

Several participants each hold an opinion, represented by a next-token policy that says how
likely the participant is to write each continuation of a partial statement. The task is to
produce a fair *collective* outcome over complete statements. Two forms are studied:

- a lottery over statements (a stochastic generation policy) that every participant, and every
  group of participants, can accept;
- a single consensus statement that is as acceptable as possible to the participant who likes
  it least.

## Why It Matters

Language models can aggregate free-form opinions without a fixed agenda, but a useful consensus
system needs auditable fairness guarantees rather than post-hoc selection. A lottery over
statements is fair in a strong sense when no coalition of participants could take its
proportional share of the probability and use it to make every member better off.

## Setting

Statements are built token by token on a tree with a fixed number of candidate tokens per step
and a fixed length. Participant i's utility for a complete statement X is the probability its
policy assigns to X (the product of its next-token probabilities along X); its utility for a
lottery p is the expected value under p. In the synthetic environments used here (from the
paper), participant policies are softmax functions of random participant and token embeddings,
and a polarization parameter rho controls how strongly participants disagree.

Participants' policies are only available through queries, as with language models: one query
returns every participant's next-token probabilities at one prefix, and each instance has a
query budget.

Three regimes are studied:

- **Small trees** (4 tokens, 81 complete statements, the paper's Section 4.2 setting): the
  budget covers every prefix, so every statement can be listed.
- **Deep trees** (12 tokens, 531,441 complete statements, next-token distributions that depend
  on the whole prefix): the budget covers 1,500 of the 265,720 prefixes, so only a small part of
  the tree can ever be seen, yet fairness is judged against all of it.
- **Language-model trees** (single statements): real consensus scenarios (an issue and four or
  five written opinions, including the paper's three published scenarios). A small instruction
  model (Qwen2.5-1.5B-Instruct) prompted with all the opinions proposes its two most likely next
  tokens at every prefix, giving 65,536 statements of 16 tokens; participant i's policy is the
  same model prompted with opinion i only. A solver sees only log-probabilities (participants'
  and the consensus-prompted model's) of the candidates at the prefixes it queries, 150 per
  instance, never the text, and must return one statement.

## Success Criteria

- No coalition of participants can block the outcome: using its proportional share of
  probability, no coalition can raise the expected utility of all its members (ex-ante core).
- The outcome has high Nash welfare (product of participants' expected utilities).
- On deep trees, as much Nash welfare (relative to the best lottery over the whole tree) and as
  little coalitional blocking (by any coalition, using any statement of the tree) as possible
  within the query budget.
- On language-model trees, a single statement whose least satisfied participant is as satisfied
  as possible: the lowest egalitarian perplexity (the perplexity of the statement under the
  participant who finds it least likely), judged against the best statement of the full tree.

---

# Paper's Method

## Key Contribution

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Approach

For stochastic consensus (Algorithm 1), the method enumerates the complete statements of a finite token tree, computes each participant's probability utility for every statement, solves the Nash-welfare program max_p sum_i log(u_i^T p) over lotteries on statements (Frank-Wolfe with a coordinate linear oracle), and converts the optimal lottery into a stochastic token policy using conditional statement masses, pi*(a|s) = P*(C(s,a)) / P*(C(s)) (Definition 2). Any Nash-welfare maximiser lies in the ex-ante core, and the induced policy generates exactly that lottery (Theorem 1, Corollary 1).

When the tree is too large to enumerate (Section 4.3), positions are grouped into chunks and decoding selects whole chunks from a limited candidate set; Algorithm 1 is run on the resulting chunked statement set, which gives a lottery in the core relative to that set but with no constant-factor guarantee against the full set (Theorem 2). The paper does not specify how chunk candidates are proposed; the reference implementation here proposes, at every chunk-start prefix, each participant's greedy continuation of the chunk and the greedy continuation under the equal-weight mixture of participants (up to n + 1 distinct blocks), and uses the finest chunk size whose worst-case query cost fits the budget (chunk size 4, three chunks, at most 343 statements on the deep trees; the small trees are enumerated).

For single-statement consensus (Section 5), rewards are r_i(s, a) = log pi_i(a | s), a statement's utility for participant i is the sum of these along the statement, and the target is egalitarian welfare, min_i U_i^log(X); with a fixed length this is the same as minimising the egalitarian perplexity EPPL(X) = max_i exp(-U_i^log(X) / L). Two search procedures are proposed. Beam search (Algorithm 3) keeps the w best partial statements, expands each by every candidate token, scores the candidates by their partial egalitarian welfare and keeps the top w; the paper's main experiment uses w = 4. Finite lookahead (Algorithm 2) enumerates, at every step, the paths of depth d below the current prefix (through each node's b most likely tokens under the base model), scores each by the egalitarian welfare of prefix + path and commits the first token of the best one; the paper uses d = 4, b = 2. The reference solver runs beam search with w = 4 on the language-model trees, where every beam is expanded by the tree's two candidates (the authors' code expands each beam with up to w sampled tokens). The paper's brush-up step, which only rewrites a statement's ending with another model, is not part of the gym (statements are fixed-length token sequences).

## Main Claims

The primary theoretical claim is that the Nash-welfare lottery induces a stochastic token policy whose outcome distribution lies in the ex-ante core. The primary empirical claim is that token-level search improves worst-participant alignment for consensus statements compared with sampling and Habermas-style baselines.

---

# Evaluation

## Interface

A method is a solver script `scripts/solvers/<name>.sh`. The evaluator starts it with no
arguments and talks to it with one JSON object per line: evaluator -> solver on the solver's
stdin, solver -> evaluator on its stdout (anything else must go to stderr, which is logged to
`logs/solvers_<split>/`). `data/core_synthetic/oracle_client.py` implements the solver side;
`scripts/solvers/nash_welfare_policy.sh` is an example.

For every instance, in a fixed order:

1. The evaluator sends `{"type": "instance", "instance_id", "experiment", "branching" (B),
   "depth" (L), "num_agents" (n), "rho", "query_budget", "output"}`. `output` is `"lottery"`
   (core-test experiments) or `"statement"` (lm_consensus_search); `rho` is null on
   language-model trees.
2. The solver may send `{"type": "query", "prefixes": [[a_1, ..., a_t], ...]}` (0 <= t < L,
   tokens in 0..B-1) any number of times. The reply is
   `{"type": "logprobs", "logprobs": [n x B per prefix], "queries_used": k}`:
   log pi_i(a | s) for every participant i and token a at each prefix s. On language-model trees
   the reply also carries `"base_logprobs": [B per prefix]`, the consensus-prompted model's
   log-probabilities of the same candidates (full-vocabulary values, not renormalised over the
   candidates). Each distinct prefix costs one query; repeating one is free. A batch that is invalid or would exceed the budget
   is refused with `{"type": "error", ...}` and costs nothing.
3. The solver sends `{"type": "policy", "policy": {"": [p_0, ..., p_{B-1}], "0": [...],
   "0,2": [...], ...}}`: next-token probabilities for every prefix (comma-separated token
   indices, "" for the empty prefix) reachable with positive probability. The evaluator replies
   `{"type": "accepted"}` and moves on; after the last instance it sends `{"type": "done"}`.

The evaluator executes the policy exactly to obtain the lottery over complete statements. Each
row must have B finite, non-negative entries summing to 1 within 1e-6. Any other field is
ignored. On `"statement"` instances the policy must reach exactly one complete statement
(one-hot rows along one path; `path_policy` in the client builds it). The evaluator holds the
full trees; a solver sees only what it queries.

`scripts/evaluate.sh` / `scripts/evaluate_train.sh` run every solver on disk on the public
training instances (`data/core_synthetic/generate.py`; language-model trees in
`data/lm_consensus/trees/`, built from `data/lm_consensus/scenarios_train.json` by
`data/lm_consensus/build_trees.py`); the hidden test runs every solver on held-out instances from
the same generating processes (for language-model trees: held-out scenarios, same model, prompts
and tree shape). The time the evaluator spends waiting for
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

**lm_consensus_search** (paper Section 6.3 on a small model: B=2 candidates per prefix from the
consensus-prompted Qwen2.5-1.5B-Instruct, L=16 tokens, 65,536 statements, 4-5 participants;
10 public scenarios, 20 held-out; budget 150 of 65,535 prefixes). The returned statement X is
scored with the participants' full-vocabulary log-probabilities U_i(X) = sum_t log pi_i(a_t | s_t-1).
- `log_eppl_regret` (primary, lower is better): mean over instances of
  log EPPL(X) - log EPPL(X*), where EPPL(X) = max_i exp(-U_i(X) / L) is the egalitarian
  perplexity and X* the statement of the full tree with the lowest EPPL (exact). 0 means the
  best statement on every instance.
- `egalitarian_perplexity` (lower is better): mean EPPL(X), the paper's metric.
- `optimal_statement_rate`, `base_perplexity` (perplexity under the consensus-prompted model):
  diagnostics, not ranked.

## Failure rule

A run that crashes, exceeds the time limit, breaks the protocol, misses an instance, returns an
invalid policy, or fails the core constraint of empirical_core_test on any instance receives the
worst value of every scored metric in every experiment (alpha* = 1e6, indicators and
efficiencies 0, log10 improvement 12, violation rate 1; on lm_consensus_search the worst
statement of each tree, about 4.9 log-EPPL regret and EPPL about 790 on the held-out slice). On
`"statement"` instances a policy that reaches more than one statement is invalid.

## Baselines

Computed by the evaluator on the same instances and reported next to every solver: the paper's
uniform lottery over all statements and utilitarian statement (maximising the sum of
utilities, exact), and random dictatorship over the participants' greedy statements (each
participant's argmax-token continuation, probability 1/n each). On lm_consensus_search: the
consensus-prompted model's greedy statement (`base_greedy_statement`), Best-of-N with N=4 samples
from the consensus-prompted model (renormalised over the candidates, temperature 1) choosing the
lowest EPPL (`best_of_n_statement`), and a Habermas-style vote among the same 4 samples where each
participant ranks them by its own likelihood and the Schulze winner is chosen
(`habermas_vote_statement`, no critique round); sampled baselines are averaged over 200
fixed-seed draws. Their numbers are measured, not masked.

On the held-out slice, per-instance values are not written to `scoring/scores_test.json` (they
would reveal each hidden instance's optimum); only aggregates and per-polarisation means are.

---

# Reproduction Log

### Iteration 1: gpt-5.5
- **Milestone**: `none` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h

<details>
<summary>Progress Log</summary>

<!-- Write your progress here -->

</details>

### Iteration 1: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 2 total (1 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 10:14] - none
- Read the paper and identified two core algorithmic claims: Nash-welfare stochastic policies for ex-ante-core stability and egalitarian token search for single statements.
- Implemented the synthetic core-test method and independent coalition-blocking evaluator.
- Prepared a CPU job to run the method end-to-end before claiming `method_runs` or `core_claim`.

### [2026-06-04 10:18] - none
- Job `449b4d8c-419` generated method artifacts but failed during evaluation on an unbounded baseline diagnostic LP.
- Patched evaluation to score only the Nash-welfare policy required for the current empirical core-test target.

### [2026-06-04 10:19] - core_claim_plus
- Job `63da2914-c4d` completed successfully.
- Reproduced the empirical core-test result on 25 synthetic token-tree instances across five polarization levels and five seeds.
- The Nash-welfare policy achieved maximum coalition improvement `1.0` and satisfied the core constraint on every instance.

</details>

### Iteration 2: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 3 total (2 completed, 1 failed)

<details>
<summary>Progress Log</summary>

### [2026-06-04 00:00] - core_claim
- Audited the existing Section 4.2 empirical core-test reproduction.
- Confirmed the implementation contains the paper's actual synthetic token-tree policies, Nash-welfare lottery optimization, induced stochastic prefix policy, and independent coalition-blocking LP evaluation.
- Kept the milestone at `core_claim` while a CPU test job refreshes `scores.json` with the expanded baseline and per-polarization metrics needed to restore `core_claim_plus`.

### [2026-06-04 10:27] - core_claim_plus
- Job `bf64dcb8-621` completed successfully in the container with exit code 0.
- Regenerated `scoring/scores.json` for 25 synthetic token-tree instances: five seeds across `rho = 0, 1, 2, 4, 8`.
- Restored `core_claim_plus`: `nash_welfare_policy` has maximum coalition improvement 1.0 at every polarization level, while `uniform_policy` and `utilitarian_policy` become blockable as polarization increases.

</details>

### Iteration 3: gpt-5.5
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 7m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 4: gpt-5.5 (designer_stage_1)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 5: gpt-5.5 (designer_stage_2)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 6m | **GPU**: 0.0h
- **Jobs**: 4 total (3 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 6: gpt-5.5 (designer_stage_3)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 3m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 7: gpt-5.5 (packager)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 2m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 8: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 15m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 9: gpt-5.5 (verifier)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 10m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 10: gpt-6.1-sol (redteam)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 8m | **GPU**: 0.0h
- **Jobs**: 6 total (5 completed, 1 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: error
- **Working time**: 48m | **GPU**: 0.0h
- **Jobs**: 10 total (5 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 28m | **GPU**: 0.0h
- **Jobs**: 12 total (7 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>

### Iteration 11: claude-opus-5-5 (hardener)
- **Milestone**: `core_claim_plus` | **Status**: done
- **Working time**: 50m | **GPU**: 1.2h
- **Jobs**: 16 total (9 completed, 2 failed)

<details>
<summary>Progress Log</summary>

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

</details>


---

# Reproduction Milestones

**Current: secondary_claims**

## Stop Justification
- Completed at milestone `core_claim_plus`.
