# Hardening report (final iteration)

This iteration answers `REVIEW.md`. It adds a scored experiment on token trees from a real small
language model: single-statement search with beam search and finite lookahead, plus the
Habermas-style comparison (paper Section 6.3). It then checks that a better method separates
beyond noise, and closes a leak of held-out optima found along the way.

All "after" numbers come from CPU job `9729d97b-5d6`. That job ran `hardening/run_job.sh`:
- the exactness and language-model self-tests and the search calibration;
- the reference alone on train and test, whose output is the committed `scoring/scores_*.json`;
- every audit solver through the no-argument `bash scripts/evaluate_test.sh` and
  `bash scripts/evaluate_train.sh`;
- the workspace-mutation probe, `hardening.summarize` and `validate.py --compare`.

Every step exited 0 and every check passed. The language-model trees were built and verified on
a GPU in job `e9bae4f3-c6f`.

Raw results are in `hardening/results/`:
- `all_scores_{test,train}.json`, `reference_scores_*.json`, `mutation_scores_test.json`;
- `summary.{json,md}` (cohort as in the previous report) and `ranks.json` (two cohorts, from
  `python -m hardening.ranks`);
- `lm_calibration.{json,md}`, plus `lm_selftest.jsonl` and `selftest.jsonl`.

Ranks use `sum weight * |coef| * directed(value - reference) / std`, with population standard
deviations over a cohort of honest attempts. Two cohorts are reported:
- **all:** every honest variant, including the two that fail the core gate (as before).
- **valid:** only variants that complete every experiment validly.

"Before" means the gym as the previous hardening left it: two experiments, weights 0.4/0.6.

## What changed

1. **New experiment `lm_consensus_search` (paper Section 6.3), weight 0.4.**
   - **Model and prompts.** The base model is Qwen2.5-1.5B-Instruct, prompted with the issue and
     every opinion (paper Figure 6, wording from the authors' released code). Participant i's
     policy is the same model prompted with opinion i only (Figure 7).
   - **Tree shape.** At every prefix the base model proposes its two most likely tokens (B=2,
     the paper's branching factor for lookahead). Special tokens, newlines and the authors'
     biased-against strings are excluded. Statements are 16 tokens long, giving 65,536
     statements per scenario.
   - **Scenarios.** Train has 10 scenarios: the paper's three (Tables 11-13) and seven authored
     ones. Test has 20 authored scenarios on other issues, 4-5 participants each. The test
     texts exist only in `eval/test/lm_scenarios.json`.
   - **Storage.** The trees store every prefix's candidate token ids and the float16
     log-probabilities, under the full-vocabulary softmax, of every participant and of the base
     model. Train trees are in `data/lm_consensus/trees/` (15 MB); held-out trees are in
     `eval/test/lm_trees/` (28 MB), both committed.
   - **Light hidden test.** It only reads the trees; the language model never runs at test
     time. The reference-only hidden test, all three experiments included, took 71 s on the
     audit node.
   - **Protocol.** The same oracle protocol as before. On these trees replies also carry the base
     model's log-probabilities, the header says `"output": "statement"`, and the solver must
     return a deterministic policy (exactly one statement); anything else fails the run. A
     solver sees log-probabilities only, never text, so it cannot run the public model on the
     hidden scenarios itself.
   - **Budget.** 150 queries per instance, 0.23% of the 65,535 prefixes. This is about 2.5x
     what the reference uses (59) and about a third of what exact best-first search needs to
     prove the optimum (445 on average on test).
   - **Scoring, recomputed from the returned statement against the full tree:**
     - `log_eppl_regret` (primary, coef 1.0): log EPPL(X) - log EPPL(X*), where X* is the best
       statement of the tree, computed exactly.
     - `egalitarian_perplexity` (the paper's EPPL, coef 0.5).
     - Diagnostics: `optimal_statement_rate`, and `base_perplexity` (fluency under the
       consensus-prompted model).
     - A failed run receives the tree's worst statement on every instance (log-EPPL regret 4.94,
       EPPL 794 on test).
   - **Baselines fixed by `eval/`:**
     - the base model's greedy statement;
     - Best-of-4 base-model samples, chosen by EPPL (the paper's Best-of-N);
     - a Habermas-style vote: the same 4 samples ranked by each participant's likelihood, with
       the Schulze winner chosen (no critique round).
     - Sampled baselines are averaged over 200 fixed-seed draws, so they are deterministic.
   - **Reference solver.** On statement instances it runs the paper's beam search (Algorithm 3,
     width 4); `--lm-search lookahead` runs Algorithm 2 (d=4, b=2).
   - **Tree builder (`data/lm_consensus/build_trees.py`).**
     - bf16 weights with a float32 output layer.
     - Batches carry ancestor key/value caches.
     - Memory-adaptive for shared GPUs; the first attempt died because another process held 75
       GB of the 80 GB card.
     - Before saving, sampled prefixes on every tree are recomputed with a plain forward pass.
       The largest disagreement over all 30 trees was 0.13 nats, against a tolerance of 0.5.
2. **Deeper trees after a pilot.** The first build used 4 candidates and 8 tokens. On it, a
   width-16 beam and best-first search found the exact optimum on every held-out instance with
   about 50-100 queries, and the reference was already optimal on 14 of 20 instances. The trees
   were rebuilt at 2 candidates and 16 tokens; the pilot calibration is kept in
   `hardening/results/pilot_b4_l8/`.
3. **Budget from the calibration (`hardening/lm_calibration.md`).** At the provisional 300
   queries, a width-19 beam reached the optimum on every held-out instance. At 150, no audited
   method does.
4. **Leak closed: per-instance held-out values.**
   - The committed `scoring/scores_test.json` used to list per-instance values on the held-out
     slice. From a method's value and its gap to the optimum, one can recover each hidden
     instance's optimum: the best statement's EPPL, or a deep tree's Nash-welfare optimum. A
     solver could hard-code those per instance as search bounds.
   - The evaluator now writes held-out per-instance rows only to the git-ignored `logs/`, and
     the scores keep aggregates and per-rho means.
   - The calibration output, held-out solver logs and the held-out tree build log are no longer
     committed outside `eval/test/`.
   - The reference solver no longer logs its statements.
5. **Weights 0.4 / 0.6 → 0.2 / 0.4 / 0.4** (core test / deep trees / language-model search).
   - The core test is a hard gate at its bound: every valid method ties there, so its weight
     only adds to what failing costs.
   - The paper's two contributions are the core-stable lottery (deep trees carry its headroom)
     and single-statement search, its headline empirical result. They now count equally.
6. **Unchanged.** The core-test and deep-tree experiments, their instances and the reference
   method are unchanged. The reference reproduces its previous numbers exactly: deep efficiency
   0.7724, mean log10 alpha* 0.2999, violation rate 0.95 on test; 0.7317 on train.

## Do better results score higher?

**Before:** better variants existed only on the deep trees: +1.104 / +0.956 / +0.933, and the
coarse-chunk weaker variant -0.146. Single-statement search was not tested.

**After, deep trees** (hidden test; identical solvers; only the weights changed):

| variant | rank (all) | rank (valid) | paired NW-efficiency gain, 95% CI | W/L |
|---|---:|---:|---:|---:|
| better_wide_colgen | +0.901 | +2.422 | +0.225 [0.179, 0.271] | 40/0 |
| better_budgeted_colgen | +0.782 | +2.258 | | |
| better_chunked_colgen | +0.764 | +2.240 | | |
| weaker_coarse_chunks | -0.121 | -0.442 | -0.058 [-0.073, -0.044] | 0/39 |

**After, language-model search** (hidden test, 20 held-out scenarios; core and deep parts
identical to the reference):

| variant | log-EPPL regret | EPPL | optimal | rank (all) | rank (valid) | paired regret gain, 95% CI | W/L |
|---|---:|---:|---:|---:|---:|---:|---:|
| better_lm_best_first (best-first within budget) | 0.0217 | 5.242 | 0.80 | +0.021 | +0.213 | +0.082 [0.036, 0.137] | 10/0 |
| better_lm_wide_beam (beam width 9 = budget) | 0.0215 | 5.254 | 0.75 | +0.021 | +0.212 | +0.082 [0.036, 0.132] | 9/0 |
| better_lm_rate_beam (wide beam, rate-adjusted scores) | 0.0215 | 5.255 | 0.75 | +0.021 | +0.212 | +0.082 [0.036, 0.133] | 9/0 |
| **reference** (beam search, width 4) | 0.1040 | 5.734 | 0.50 | 0 | 0 | | |
| paper_finite_lookahead (d=4, b=2) | 0.3405 | 7.730 | 0.20 | -0.060 | -0.675 | -0.237 [-0.427, -0.072] | 4/10 |
| weaker_lm_beam2 (width 2) | 0.3959 | 8.280 | 0.30 | -0.074 | -0.841 | -0.292 [-0.501, -0.107] | 3/11 |
| weaker_lm_greedy (width 1) | 0.7750 | 12.246 | 0.05 | -0.170 | -2.003 | -0.671 [-0.907, -0.448] | 2/18 |

- **Train:** the same picture. The reference has regret 0.115; the better variants 0.011-0.017
  (6-7 wins, at most 1 loss in 10). Lookahead scores 0.261 and width 2 scores 0.215.
- **Comparable difficulty:** the reference scores 0.115 on train and 0.104 on test.
- **Seed noise:** the reference rerun with seeded Frank-Wolfe starts (`reference_seed_11`, `_22`)
  is identical on every scored number of every experiment (paired differences 0). The
  instances are fixed, the oracle is deterministic, and beam search is deterministic.
- **The smallest effect is far outside noise:** the better language-model variants' gain of
  0.082 has a bootstrap CI that excludes 0, with no losing instance.
- **Paper direction on the held-out scenarios.** The paper reports EPPL: beam 2.87 < lookahead
  4.18 < Best-of-N 6.35 < prompted Habermas Machine 9.67 < Habermas Machine statement 15.69. The
  gym gives the same ordering:

  | method | EPPL | log-EPPL regret |
  |---|---:|---:|
  | beam search | 5.73 | 0.104 |
  | finite lookahead | 7.73 | 0.341 |
  | Best-of-4 | 13.37 | 0.811 |
  | Habermas-style vote | 14.07 | 0.848 |
  | base model's own statement | 24.25 | 1.383 |

  The best statement of each tree averages EPPL 5.13. The core test still reproduces Figure 2:
  the Nash-welfare policy stays at alpha* = 1, while uniform reaches 9.90, utilitarian 2.75e5 and
  random dictatorship 1.21.

## Shortcuts: fix and score before → after

Test ranks against the reference, in the "all" cohort; a failed run scores -5.509. The probes of
the previous report were rerun on the final code.

| shortcut | before | fix / why it fails now | after |
|---|---:|---|---:|
| grid replay, partial coverage, easy instances, empty results, scaled mass, NaN/null/zero/empty policy, forged fields, constraint trade-off, crash, solver named like a baseline | -5.126 each | unchanged rules: hidden instances served in order, strict policy checks, core gate, worst values in every experiment | -5.509 each |
| solver named like a language-model baseline (`best_of_n_statement`) | — | reported as `solver_best_of_n_statement`; the baseline row is untouched | scored as its own solver (-0.170) |
| brute force on deep trees (enumerate until refused) | -0.945 | 1,500-query budget | -0.794 |
| seed scan (regenerate public trees) | test 0.000, train +1.381 | 128-bit secret test seeds | test 0.000, train +1.118 |
| **brute force on language-model trees** (list the tree level by level until the budget, then complete greedily) | — (new) | 150 of 65,535 prefixes; listing gets 7 levels deep | -0.087 (valid -0.978); regret 0.451 |
| **Best-of-many** (as many base-model samples as the budget allows) | — (new) | budget | -0.097 (valid -1.089); regret 0.489 |
| **copy the base model's statement** (zero-shot consensus) | — (new) | it is what the experiment measures against | -0.328 (valid -4.463); regret 1.383 |
| **lottery instead of one statement** | — (new) | a policy reaching more than one statement fails the run | -5.509 |
| **read held-out optima from committed per-instance scores** | possible (per-instance test rows were committed) | per-instance held-out rows only in git-ignored `logs/` | closed |
| **run the public model on the hidden scenarios** | — (new) | solvers never see text; held-out scenarios are authored and exist only in `eval/test/` | not possible without reading `eval/test/` |

**Mutation probe** (`hardening/mutation_run.sh`, four changes applied at once, then the
no-argument hidden test):
1. the public synthetic generator was rewritten so that all agents are identical;
2. the public language-model loader was rewritten to give every participant the base model's
   preferences;
3. a perfect `scores_test.json` was prefilled and train scores were filtered;
4. a weak solver was added.

The reference's numbers were unchanged in all three experiments, and the weak solver appeared
with worst values (LM regret 4.938).

## Shortcuts left open

- **Reading `eval/test/` from a solver.** A solver runs as the same user and could open
  `eval/test/lm_scenarios.json`, the held-out trees or the test generator. Only the rules
  (`scoring/CONSTRAINTS.md`) and the separate shortcut grader stop this.
- **Train is public.** Anyone can regenerate the train trees (synthetic seeds, public scenarios
  and model), so seed scanning pays on train (+1.118). It buys nothing on test.

## Reference score before and after

| | before | after (committed `scoring/scores_test.json`) |
|---|---|---|
| small trees | alpha* 1, core satisfied, NW efficiency 1 | identical |
| deep trees | efficiency 0.7724, log10 alpha* 0.2999, violation rate 0.95 | identical |
| language-model search | — | log-EPPL regret 0.1040, EPPL 5.734, optimal on 10/20, 59 queries |

- **Language-model baselines, test:**
  - Best-of-4: 0.811 / 13.37.
  - Habermas-style vote: 0.848 / 14.07.
  - Greedy base statement: 1.383 / 24.25.
- **Language-model, train:** the reference scores 0.115 / 4.44; the baselines score 0.593 /
  0.630 / 0.894 regret.

## Headroom check

- **Small trees:** the reference sits at the bound of all three metrics. This is the paper's
  core claim, kept as a gate.
- **Deep trees:** the reference is far from every bound (0.772 / 0.300 / 0.95). The best variant
  reaches 0.997 / 0.0066 / 0.175, so headroom above it remains mostly in violation rate.
- **Language-model search (added):** the reference is at regret 0.104, optimal on half the
  instances.
  - The best audited variants reach 0.0215 (optimal on 75-80%). No variant reaches the bound.
  - Exact search would need about three times the budget.
  - Brute force (0.451), Best-of-many (0.489) and the paper's baselines (0.81-1.38) all sit
    clearly below the reference.
  - The baselines are on a scale that leaves room above the reference.

## What could not be fixed, stated plainly

- **Thin language-model headroom.**
  - The reference already finds the best statement on 10 of the 20 held-out scenarios, so the
    separation of better methods rests on the other 10. It is consistent there: 9-10 wins and
    no losses.
  - Above the best audited variant, the remaining gap is small: 0.02 regret, on 20-25% of
    instances.
- **The rank depends on the cohort.** A failed run receives the worst statement of every tree
  (regret about 4.9, EPPL about 790). Attempts that fail therefore inflate the spread of the
  language-model metrics. In a cohort with failures, honest language-model gains count for
  little (+0.02); among valid attempts they count +0.21. The deep-tree headroom dominates the
  rank in both cohorts.
- **Scaled setting, not the paper's numbers.**
  - Qwen2.5-1.5B-Instruct instead of Llama 3.1 8B.
  - 16-token statements from the base model's top two tokens, instead of sampled tokens up to
    50 tokens with a brush-up rewrite.
  - The paper's three published scenarios are in train; the other 27 (7 train, all 20 test)
    were written for the gym in the same style.
  - The paper's EPPL values are kept under `_paper_reported` and are not comparable. The paper's
    ordering is reproduced; its magnitudes are not.
  - Beam width 4 expands each beam by the tree's two candidates (the authors sample up to four
    tokens per beam).
- **Habermas comparison.** It is a voting-only analogue (samples, then Schulze); the critique
  round and the original Chinchilla statement are not reproducible in an index-only tree.
- **Not reproduced:** the paper's Sections 6.1 (opinion-length scaling against human ratings)
  and 6.2 (credit assignment), and the 41-agent appendix experiment.
- **Synthetic deep trees.** Their token vectors are still independent per prefix. Real
  language-model structure is now covered by the new experiment instead.
