# Paper's Method

## Key Contribution

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Approach

For stochastic consensus, the method enumerates a finite token tree, computes each participant's probability utility for every complete path, solves a Nash-welfare optimization over lotteries on complete paths, and converts the optimal leaf distribution into a stochastic token policy using conditional leaf masses.

For single-statement consensus, the method uses additive log utilities and searches the token tree for statements that maximize egalitarian welfare, using finite lookahead or beam search.

## Main Claims

The primary theoretical claim is that the Nash-welfare lottery induces a stochastic token policy whose outcome distribution lies in the ex-ante core. The primary empirical claim is that token-level search improves worst-participant alignment for consensus statements compared with sampling and Habermas-style baselines.
