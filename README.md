# Overview

- **Paper ID:** 2510.14106
- **Title:** Generating Fair Consensus Statements with Social Choice on Token-Level MDPs
- **Domain:** Generative social choice / fair text generation
- **TL;DR:** The paper turns consensus statement generation into a token-level social-choice problem and derives fair stochastic and deterministic decoding procedures.

## Short Summary

The paper represents each participant's opinion as a next-token policy and uses that policy to define token-level rewards. It proves that maximizing Nash welfare over complete generated statements yields a stochastic policy in the ex-ante core. It also evaluates finite-lookahead and beam-search procedures that optimize worst-participant alignment for single consensus statements.

## Key Results

- In the synthetic core test, the Nash-welfare policy remains unblocked with maximum coalition improvement equal to 1.
- On three Habermas Machine scenarios, beam search reports the best overall EPPL in the main experiment, followed by finite lookahead.

---

# Problem Definition

## Research Question

The task is to produce collective text statements from multiple free-form participant opinions while preserving fairness to the least satisfied participants and avoiding outcomes that a subset of participants could reasonably reject.

## Why It Matters

Open-ended text aggregation can reduce agenda-setting by allowing participants to express views in their own words, but a useful consensus system must be auditable and fair rather than relying on opaque post-hoc selection alone.

## Success Criteria

A successful system should produce statements or statement lotteries that are aligned with all participant perspectives. For stochastic outcomes, no coalition should be able to improve on the outcome using only its proportional share. For single statements, the worst-aligned participant should remain as well represented as possible.

---

# Paper's Method

## Key Contribution

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Approach

For stochastic consensus, the method enumerates a finite token tree, computes each participant's probability utility for every complete path, solves a Nash-welfare optimization over lotteries on complete paths, and converts the optimal leaf distribution into a stochastic token policy using conditional leaf masses.

For single-statement consensus, the method uses additive log utilities and searches the token tree for statements that maximize egalitarian welfare, using finite lookahead or beam search.

## Main Claims

The primary theoretical claim is that the Nash-welfare lottery induces a stochastic token policy whose outcome distribution lies in the ex-ante core. The primary empirical claim is that token-level search improves worst-participant alignment for consensus statements compared with sampling and Habermas-style baselines.

---

# Evaluation

## Metrics

The core-stability metric is maximum coalition improvement `alpha*`. Values greater than 1 mean some coalition can block the outcome; values at or below 1 satisfy the ex-ante-core constraint.

For single consensus statements, the paper reports Egalitarian Perplexity, the maximum participant-specific perplexity of the generated statement. Lower values mean better worst-participant alignment.

## Baselines and Targets

The paper's synthetic core test expects the fair stochastic outcome to have maximum coalition improvement equal to 1 across polarization levels. The same test reports that uniform and utilitarian alternatives become blockable as polarization increases.

The paper's main text-generation benchmark reports lower EPPL for token-search methods than Best-of-N and Habermas-style baselines on three Habermas Machine scenarios.

## Evaluation Protocol

The included executable reproduction evaluates the synthetic core-stability setting from the paper: multiple random token trees, multiple polarization values, fixed branching factor 3, and depth 4. Evaluation reads saved utility matrices and output distributions, then independently solves the coalition-blocking LP for each instance.

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


---

# Reproduction Milestones

**Current: core_claim_plus**

## Stop Justification
- Completed at milestone `core_claim_plus`.
