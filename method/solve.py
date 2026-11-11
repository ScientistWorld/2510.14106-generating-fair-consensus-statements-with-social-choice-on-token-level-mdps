#!/usr/bin/env python3
"""Solver entry point: core-stochastic token policies for every instance in a directory.

Usage: python -m method.solve --instances DIR --outputs DIR

For a tree small enough to enumerate (B^L <= --enumeration-limit) Algorithm 1 runs on all
complete statements; otherwise it runs on the chunked statement set of Section 4.3.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from method.chunking import chunk_candidates, chunked_leaves
from method.nash_lottery import all_leaves, induce_prefix_policy, leaf_logutils, solve_nash_welfare


def candidate_leaves(logp: np.ndarray, args) -> tuple[np.ndarray, str]:
    n, depth, branching = logp.shape
    if branching ** depth <= args.enumeration_limit:
        return all_leaves(branching, depth), "full_tree"
    blocks = chunk_candidates(logp, args.chunk_size, args.num_candidates)
    return chunked_leaves(blocks), f"chunked(c={args.chunk_size},K={args.num_candidates})"


def solve_instance(instance: dict, args) -> dict:
    logp = np.asarray(instance["agent_token_logprobs"], dtype=np.float64)
    branching = int(instance["branching"])
    leaves, mode = candidate_leaves(logp, args)
    logu = leaf_logutils(logp, leaves)
    utilities = np.exp(logu - logu.max(axis=1, keepdims=True))
    full = mode == "full_tree"
    init = None
    if args.init_seed is not None:
        rng = np.random.default_rng(args.init_seed)
        if full:
            init = rng.dirichlet(np.ones(leaves.shape[0]))
        else:
            init = np.zeros(leaves.shape[0])
            init[int(rng.integers(leaves.shape[0]))] = 1.0
    result = solve_nash_welfare(
        utilities,
        tol=args.tol_full if full else args.tol_chunked,
        max_iter=args.max_iter_full if full else args.max_iter_chunked,
        init=init,
    )
    p = result.distribution
    keep = p > 1e-14
    leaves, p = leaves[keep], p[keep] / p[keep].sum()
    return {
        "policy": induce_prefix_policy(leaves, p, branching),
        "info": {"mode": mode, "candidates": int(keep.size), "support": int(keep.sum()),
                 "fw_gap": result.gap, "fw_iterations": result.iterations},
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instances", required=True)
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--enumeration-limit", type=int, default=200_000)
    parser.add_argument("--chunk-size", type=int, default=6)
    parser.add_argument("--num-candidates", type=int, default=12)
    parser.add_argument("--tol-full", type=float, default=1e-10)
    parser.add_argument("--tol-chunked", type=float, default=1e-9)
    parser.add_argument("--max-iter-full", type=int, default=50_000)
    parser.add_argument("--max-iter-chunked", type=int, default=20_000)
    parser.add_argument("--init-seed", type=int, default=None)
    return parser


def main(argv=None, solve_fn=solve_instance) -> None:
    args = build_parser().parse_args(argv)
    in_dir, out_dir = Path(args.instances), Path(args.outputs)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((in_dir / "manifest.json").read_text(encoding="utf-8"))
    start = time.time()
    for entry in manifest["instances"]:
        instance = json.loads((in_dir / entry["file"]).read_text(encoding="utf-8"))
        t0 = time.time()
        out = solve_fn(instance, args)
        (out_dir / f"{instance['instance_id']}.json").write_text(json.dumps(out), encoding="utf-8")
        print(f"{instance['instance_id']} {out.get('info', {})} {time.time() - t0:.2f}s", flush=True)
    print(f"solved {len(manifest['instances'])} instances in {time.time() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
