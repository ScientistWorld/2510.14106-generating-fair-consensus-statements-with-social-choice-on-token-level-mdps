# Research Direction

## Research Problem

Fair aggregation of several participants' token-level preferences into a collective outcome over
complete statements: a stochastic outcome judged by coalitional stability (ex-ante core) and Nash
welfare, and a single statement judged by the satisfaction of its least satisfied participant
(egalitarian perplexity).

## Core Contribution

Improvements should change how fair outcomes are found when only a small part of the statement
tree can be queried: which prefixes to query, which statements to consider, how to search the
tree, how to optimise a lottery and keep it stable against coalitions that can use any statement
of the tree, and how to find a single statement close to the best one for the worst-off
participant on real language-model trees.

## Approach Scope

Any optimisation, search, approximation or decoding procedure is in scope, as long as the output
is a token policy over the given tree (stochastic on "lottery" instances, one statement on
"statement" instances), computed from the participants' policies on each instance within its
query budget and the time limit.

## Out of Bounds

Reading or reconstructing the held-out instances (including regenerating trees with the public
generator or running the language model on guessed scenarios), obtaining participants'
probabilities other than through the oracle, hard-coding answers for known instances, editing
the evaluator, generator or tree data, or bypassing the failure rule. Trading coalitional
stability on small trees for welfare is not an improvement: such runs fail.
