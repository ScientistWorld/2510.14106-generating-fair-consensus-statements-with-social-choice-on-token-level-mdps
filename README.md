# Generating Fair Consensus Statements with Social Choice on Token-Level MDPs

<!-- scope-note -->
> [!NOTE]
> **Not a full reproduction of the paper.** This environment keeps the paper's core problem at a reduced scale so it is easy to run. It measures an AI scientist's idea and how well it is carried out, not the full research loop.

| This paper | Scientist World, the framework this environment belongs to |
|:---:|:---:|
| <a href="https://arxiv.org/abs/2510.14106"><img alt="This paper on arXiv" src="https://img.shields.io/badge/arXiv-2510.14106-b31b1b?logo=arxiv&logoColor=white"></a> | <a href="https://arxiv.org/abs/2611.08478"><img alt="Scientist World on arXiv" src="https://img.shields.io/badge/arXiv-2611.08478-b31b1b?logo=arxiv&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world"><img alt="Scientist World code" src="https://img.shields.io/badge/Code-Scientist_World-181717?logo=github&logoColor=white"></a> <a href="https://github.com/zlab-princeton/scientist_world_cli"><img alt="Command line code" src="https://img.shields.io/badge/Code-Command_line-181717?logo=github&logoColor=white"></a> |

| **Paper ID** | [2510.14106](https://arxiv.org/abs/2510.14106) |
|:---|:---|
| **Domain** | Generative social choice / fair text generation |
| **TL;DR** | The paper turns consensus statement generation into a token-level social-choice problem and derives fair stochastic and deterministic decoding procedures. |

## Summary

The paper represents each participant's opinion as a next-token policy and uses that policy to define token-level rewards. It proves that maximizing Nash welfare over complete generated statements yields a stochastic policy in the ex-ante core. It also evaluates finite-lookahead and beam-search procedures that optimize worst-participant alignment for single consensus statements.

## Problem

The task is to produce collective text statements from multiple free-form participant opinions while preserving fairness to the least satisfied participants and avoiding outcomes that a subset of participants could reasonably reject.

## The paper's method

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Key results of the paper

- In the synthetic core test, the Nash-welfare policy remains unblocked with maximum coalition improvement equal to 1.
- On three Habermas Machine scenarios, beam search reports the best overall EPPL in the main experiment, followed by finite lookahead.

## Run it

```bash
pip install sciworld
sw research 2510.14106 --mode innovate --agent claude-code
sw evaluate 2510.14106
```

## More

The full problem statement, method notes, evaluation protocol and reproduction log are in [DETAILS.md](DETAILS.md).

