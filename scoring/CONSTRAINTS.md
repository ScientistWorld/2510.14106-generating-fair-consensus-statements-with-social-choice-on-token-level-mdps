# Scientist Constraints

## Problem Setting

A method is a solver in `scripts/solvers/<name>.sh` speaking the oracle protocol described in
`briefing/evaluation.md`. For each instance it may query participants' next-token
probabilities within the instance's budget and returns a token policy: a stochastic policy on
"lottery" instances (the core-test experiments), and a deterministic policy that spells one
statement on "statement" instances (language-model consensus search). It must handle all
instance families with the same code; it must not read anything under `eval/` (including the
held-out scenarios and trees in `eval/test/`), must not try to reconstruct the hidden instances,
and must not depend on which instances are served.

## Evaluation

The evaluator (`eval/`) is fixed: the held-out instances, the query oracle and its budgets,
policy execution, the exact optima and coalition-blocking computations, the baselines, the
failure rule and the aggregation. Do not edit `eval/`, `data/core_synthetic/generate.py`,
`data/lm_consensus/` (scenarios, trees, builder, loader), or the scores files by hand; scores are
only what `scripts/evaluate_*.sh` writes.

## Resources

CPU only at evaluation time. Query budgets: every prefix on small trees, 1,500 prefixes per
deep-tree instance, 150 prefixes per language-model instance. The evaluator waits at most one
hour per evaluation run for each solver (the reference needs a few seconds). A solver must not
run a language model to obtain participants' or the base model's probabilities: they come only
from the oracle. The model weights (`scripts/download.sh`) are needed only to rebuild the public
trees.

## Compliance Checklist

- [ ] Solver returns a valid policy for every instance served (one statement on "statement"
      instances)
- [ ] Outcome lies in the ex-ante core on every small-tree instance
- [ ] Solver gets participants' policies only through the oracle; it does not read `eval/`,
      regenerate instances, run the language model, or hard-code answers
- [ ] Evaluation code does not import from `method/`
