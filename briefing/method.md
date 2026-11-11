# Paper's Method

## Key Contribution

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Approach

For stochastic consensus (Algorithm 1), the method enumerates the complete statements of a finite token tree, computes each participant's probability utility for every statement, solves the Nash-welfare program max_p sum_i log(u_i^T p) over lotteries on statements (Frank-Wolfe with a coordinate linear oracle), and converts the optimal lottery into a stochastic token policy using conditional statement masses, pi*(a|s) = P*(C(s,a)) / P*(C(s)) (Definition 2). Any Nash-welfare maximiser lies in the ex-ante core, and the induced policy generates exactly that lottery (Theorem 1, Corollary 1).

When the tree is too large to enumerate (Section 4.3), positions are grouped into chunks and decoding selects whole chunks from a limited candidate set; Algorithm 1 is run on the resulting chunked statement set, which gives a lottery in the core relative to that set but with no constant-factor guarantee against the full set (Theorem 2). The paper does not specify how chunk candidates are proposed; the reference implementation here proposes, for each chunk, every participant's most likely chunk plus the most likely chunks under the equal-weight mixture of participants (chunk size 6, 12 candidates).

For single-statement consensus, the method uses additive log utilities and searches the token tree for statements that maximize egalitarian welfare, using finite lookahead or beam search.

## Main Claims

The primary theoretical claim is that the Nash-welfare lottery induces a stochastic token policy whose outcome distribution lies in the ex-ante core. The primary empirical claim is that token-level search improves worst-participant alignment for consensus statements compared with sampling and Habermas-style baselines.
