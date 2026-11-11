# Research Direction

## Research Problem

Fair aggregation of several participants' token-level preferences into a collective stochastic
outcome over complete statements, with fairness judged by coalitional stability (ex-ante core)
and Nash welfare.

## Core Contribution

Improvements should change how a fair lottery over statements is found, especially when the
statement tree is far too large to list: which statements to consider, how to search the tree,
how to optimise the lottery, and how to keep it stable against coalitions that can use any
statement of the tree.

## Approach Scope

Any optimisation, search, approximation or decoding procedure is in scope, as long as the output
is a stochastic token policy over the given tree, computed from the participants' policies on
each instance within the time limit.

## Out of Bounds

Reading or reconstructing the held-out instances, hard-coding answers for known instances,
editing the evaluator or generator, or bypassing the failure rule. Trading coalitional stability
on small trees for welfare is not an improvement: such runs fail.
