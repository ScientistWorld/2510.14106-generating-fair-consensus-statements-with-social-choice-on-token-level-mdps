# Scientist Constraints

## Problem Setting

A method is a solver in `scripts/solvers/<name>.sh` speaking the oracle protocol described in
`briefing/evaluation.md`. For each instance it may query participants' next-token
probabilities within the instance's budget and returns a stochastic token policy. It must handle
both instance families with the same code; it must not read anything under `eval/`, must not try
to reconstruct the hidden instances, and must not depend on which instances are served.

## Evaluation

The evaluator (`eval/`) is fixed: the held-out instances, the query oracle and its budget,
policy execution, the exact optimum and coalition-blocking computations, the baselines, the
failure rule and the aggregation. Do not edit `eval/`, `data/core_synthetic/generate.py`, or
the scores files by hand; scores are only what `scripts/evaluate_*.sh` writes.

## Resources

CPU only. Query budgets: every prefix on small trees, 1,500 prefixes per deep-tree instance.
The evaluator waits at most one hour per evaluation run for each solver (the reference needs
about ten seconds). No external data or model weights.

## Compliance Checklist

- [ ] Solver returns a valid policy for every instance served
- [ ] Outcome lies in the ex-ante core on every small-tree instance
- [ ] Solver gets participants' policies only through the oracle; it does not read `eval/`,
      regenerate instances, or hard-code answers
- [ ] Evaluation code does not import from `method/`
