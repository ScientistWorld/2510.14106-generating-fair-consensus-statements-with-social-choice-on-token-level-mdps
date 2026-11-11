# Problem Definition

## Research Question

Several participants each hold an opinion, represented by a next-token policy that says how
likely the participant is to write each continuation of a partial statement. The task is to
produce a fair *collective* outcome over complete statements: a lottery over statements (a
stochastic generation policy) that every participant, and every group of participants, can
accept.

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

Two regimes are studied:

- **Small trees** (4 tokens, 81 complete statements, the paper's Section 4.2 setting): the
  budget covers every prefix, so every statement can be listed.
- **Deep trees** (12 tokens, 531,441 complete statements, next-token distributions that depend
  on the whole prefix): the budget covers 1,500 of the 265,720 prefixes, so only a small part of
  the tree can ever be seen, yet fairness is judged against all of it.

## Success Criteria

- No coalition of participants can block the outcome: using its proportional share of
  probability, no coalition can raise the expected utility of all its members (ex-ante core).
- The outcome has high Nash welfare (product of participants' expected utilities).
- On deep trees, as much Nash welfare (relative to the best lottery over the whole tree) and as
  little coalitional blocking (by any coalition, using any statement of the tree) as possible
  within the query budget.
