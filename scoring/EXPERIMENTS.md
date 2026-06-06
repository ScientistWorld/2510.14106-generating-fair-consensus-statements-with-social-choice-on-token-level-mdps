# Experiment Overview

## empirical_core_test

Tests whether a stochastic consensus outcome is stable against every possible participant coalition under the paper's small synthetic preference environment. The experiment measures whether the outcome stays within the fairness constraint as participant preferences become increasingly polarized.

The evaluation also reports the same blocking metric separately for each polarization level and for the uniform and utilitarian reference outcomes, so future agents can see whether improvements preserve the paper's contrast between the fair lottery and blockable alternatives.

## synthetic_single_statement_search

Tests whether single-statement search improves the least-participant alignment score on the same finite token-tree instances. The experiment compares search-based outputs with simple deterministic reference paths and reports both mean and worst-case egalitarian perplexity.

The reported paper reference values come from the paper's Table 10 LLM consensus-generation EPPL summary. The reproduced values are a synthetic adaptation intended to exercise the same single-statement objective without requiring paid APIs or large model downloads.
