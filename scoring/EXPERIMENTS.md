# Experiment Overview

## empirical_core_test

Small synthetic statement trees (81 complete statements) with six participants whose
preferences range from mild to strongly polarized. The experiment establishes whether a
stochastic consensus outcome is stable against every possible coalition of participants (no
coalition can do better for all its members with its proportional share of probability), and
how much collective welfare it achieves among outcomes on the same statements. Outcomes that
some coalition can block fail the whole run. The uniform, utilitarian and random-dictatorship
outcomes are reported for contrast: they become blockable as polarization grows.

## deep_tree_core_test

The same kind of preferences on statements 24 tokens long, so that the set of complete
statements (about 2.8 * 10^11) is far too large to list. The experiment asks how close a
method can get, within a fixed time budget, to a fair outcome judged against the entire
statement tree: how much welfare it delivers relative to what each participant could ideally
get, and how often small coalitions (up to three participants) could still block it using
statements anywhere in the tree.
