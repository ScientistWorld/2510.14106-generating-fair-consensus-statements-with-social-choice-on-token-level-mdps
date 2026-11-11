# Scientist Constraints

## Problem Setting

A method is a solver in `scripts/solvers/<name>.sh` with the interface described in
`briefing/evaluation.md`. It receives each instance (the participants' token policies) and
returns a stochastic token policy. It must handle both instance families with the same code;
it must not read anything under `eval/`, and must not depend on which instances are passed.

## Evaluation

The evaluator (`eval/`) is fixed: instance generation for the held-out slice, policy execution,
the coalition-blocking computations, the baselines, the failure rule and the aggregation. Do not
edit `eval/`, `data/core_synthetic/generate.py`, or the scores files by hand; scores are only
what `scripts/evaluate_*.sh` writes.

## Resources

CPU only. Each solver run has a two-hour wall-clock limit for all instances of a slice (the
reference uses a few minutes). No external data or model weights.

## Compliance Checklist

- [ ] Solver writes a valid policy for every instance in the manifest
- [ ] Outcome lies in the ex-ante core on every small-tree instance
- [ ] Solver does not read `eval/`, test files, or hard-code instance answers
- [ ] Evaluation code does not import from `method/`
