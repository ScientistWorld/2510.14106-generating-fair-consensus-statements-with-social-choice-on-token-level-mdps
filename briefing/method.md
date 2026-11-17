# Paper's Method

## Key Contribution

The paper models free-form consensus generation as a multi-objective token-level Markov Decision Process, where each participant is represented by a policy over next tokens and each policy induces participant-specific token rewards.

## Approach

For stochastic consensus (Algorithm 1), the method enumerates the complete statements of a finite token tree, computes each participant's probability utility for every statement, solves the Nash-welfare program max_p sum_i log(u_i^T p) over lotteries on statements (Frank-Wolfe with a coordinate linear oracle), and converts the optimal lottery into a stochastic token policy using conditional statement masses, pi*(a|s) = P*(C(s,a)) / P*(C(s)) (Definition 2). Any Nash-welfare maximiser lies in the ex-ante core, and the induced policy generates exactly that lottery (Theorem 1, Corollary 1).

When the tree is too large to enumerate (Section 4.3), positions are grouped into chunks and decoding selects whole chunks from a limited candidate set; Algorithm 1 is run on the resulting chunked statement set, which gives a lottery in the core relative to that set but with no constant-factor guarantee against the full set (Theorem 2). The paper does not specify how chunk candidates are proposed; the reference implementation here proposes, at every chunk-start prefix, each participant's greedy continuation of the chunk and the greedy continuation under the equal-weight mixture of participants (up to n + 1 distinct blocks), and uses the finest chunk size whose worst-case query cost fits the budget (chunk size 4, three chunks, at most 343 statements on the deep trees; the small trees are enumerated).

For single-statement consensus (Section 5), rewards are r_i(s, a) = log pi_i(a | s), a statement's utility for participant i is the sum of these along the statement, and the target is egalitarian welfare, min_i U_i^log(X); with a fixed length this is the same as minimising the egalitarian perplexity EPPL(X) = max_i exp(-U_i^log(X) / L). Two search procedures are proposed. Beam search (Algorithm 3) keeps the w best partial statements, expands each by every candidate token, scores the candidates by their partial egalitarian welfare and keeps the top w; the paper's main experiment uses w = 4. Finite lookahead (Algorithm 2) enumerates, at every step, the paths of depth d below the current prefix (through each node's b most likely tokens under the base model), scores each by the egalitarian welfare of prefix + path and commits the first token of the best one; the paper uses d = 4, b = 2. The reference solver runs beam search with w = 4 on the language-model trees, where every beam is expanded by the tree's two candidates (the authors' code expands each beam with up to w sampled tokens). The paper's brush-up step, which only rewrites a statement's ending with another model, is not part of the gym (statements are fixed-length token sequences).

## Main Claims

The primary theoretical claim is that the Nash-welfare lottery induces a stochastic token policy whose outcome distribution lies in the ex-ante core. The primary empirical claim is that token-level search improves worst-participant alignment for consensus statements compared with sampling and Habermas-style baselines.
