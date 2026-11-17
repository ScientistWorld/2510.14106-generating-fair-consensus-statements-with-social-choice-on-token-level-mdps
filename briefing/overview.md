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
