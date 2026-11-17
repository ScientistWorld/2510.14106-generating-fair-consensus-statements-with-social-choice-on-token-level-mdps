# Experiment Overview

## empirical_core_test

Small synthetic statement trees (81 complete statements) with six participants whose
preferences range from mild to strongly polarized; every prefix can be queried. The experiment
establishes whether a stochastic consensus outcome is stable against every possible coalition of
participants (no coalition can do better for all its members with its proportional share of
probability), and how much collective welfare it achieves among outcomes on the same
statements. Outcomes that some coalition can block fail the whole run. The uniform, utilitarian
and random-dictatorship outcomes are reported for contrast: they become blockable as
polarization grows.

## deep_tree_core_test

The same kind of preferences on statements 12 tokens long whose next-token distributions depend
on the whole prefix, so the 531,441 complete statements form an unstructured tree; participants'
policies can only be queried at 1,500 of its 265,720 prefixes, about as a language model would be
queried. The experiment asks how close a method can get, with that limited view, to a fair
outcome judged against the entire tree: how much Nash welfare it delivers relative to the best
lottery over all statements, and how far any coalition of participants could still improve on it
using statements anywhere in the tree. Seeing more of the tree is not possible; spending the
queries well is the problem.

## lm_consensus_search

Real consensus scenarios (an issue and four or five written opinions, including the three
published in the paper) turned into statement trees by a small instruction-tuned language
model: at every prefix the model, prompted with all opinions, proposes its two most likely next
tokens, and each participant's preferences are the same model's probabilities when prompted with
that participant's opinion alone. Statements are 16 tokens long (65,536 per scenario), only 150
prefixes per scenario can be queried, and no text is ever shown. The experiment asks how close a
single returned statement comes to the one its least satisfied participant likes best (lowest
egalitarian perplexity among all statements of the tree), and whether it beats statements
obtained by sampling from the consensus-prompted model and choosing among the samples, by a
post-hoc vote among sampled candidates, or by the model's own most likely statement. Held-out
scenarios are distinct issues written in the same style as the public ones.
