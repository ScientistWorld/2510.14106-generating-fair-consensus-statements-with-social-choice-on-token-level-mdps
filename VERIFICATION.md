# Verification Report

## Initial Assessment

The workspace started at milestone `core_claim_plus`. The core empirical test artifacts and split evaluation scripts had been run successfully in the compute container, and `validate.py --compare` passed after the final cleanup.

## Issues Found

- **Minor:** A secondary deterministic single-statement search experiment had been added to code and reference documentation, but the completed job did not produce matching `scores.json` entries. Leaving it active would make the gym advertise an unverified target.
- **Minor:** Public scoring docs named the paper's specific policy instead of describing the target outcome in method-agnostic terms.

## Fixes Applied

- Removed the unverified secondary experiment from active method/eval code, `scoring/reference.json`, `scoring/TARGETS.md`, and `scoring/EXPERIMENTS.md`.
- Recomputed `scoring/scores.json` from saved empirical core-test artifacts so it reports every declared metric for the verified policies.
- Clarified method-agnostic scoring language in `briefing/evaluation.md`, `scoring/TARGETS.md`, and `scoring/EXPERIMENTS.md`.
- Ran `python validate.py --compare` and Python compile checks for the active method/eval/data modules.

## Final Judgment

The workspace is ready as a gym for the verified synthetic empirical core-stability task. It supports improve mode through the provided method, baselines, artifacts, and scores, and from-scratch mode through method-independent data and evaluation code. The main limitation is scope: it reproduces the paper's formal synthetic core-stability claim, not the full LLM text-generation benchmark.

- Milestone: core_claim_plus
- Ready for gym use: yes
- Confidence: high
- Key limitation: synthetic core-stability benchmark only
