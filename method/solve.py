#!/usr/bin/env python3
"""Reference solver: the paper's methods for both kinds of instance.

Usage: python -m method.solve   (speaks the evaluator's oracle protocol on stdin/stdout)

* "lottery" instances (core tests): core-stochastic token policies (paper Algorithm 1). When the
  instance's query budget covers every prefix of the tree, Algorithm 1 runs on all complete
  statements; otherwise it runs on the chunked statement set of Section 4.3 with the finest
  chunk size that fits the budget.
* "statement" instances (language-model consensus search): one statement by egalitarian-welfare
  beam search (paper Algorithm 3, width 4, as in the paper's main experiment); ``--lm-search
  lookahead`` runs finite lookahead (Algorithm 2) instead.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

from data.core_synthetic.oracle_client import OracleSession, path_policy
from method.chunking import choose_chunk_size, chunked_leaves, full_tree_leaves
from method.consensus_search import beam_search, finite_lookahead
from method.nash_lottery import induce_prefix_policy, leaf_logutils, solve_nash_welfare


def num_internal_prefixes(branching: int, depth: int) -> int:
    return (branching ** depth - 1) // (branching - 1)


def candidate_leaves(inst, args) -> tuple[list[tuple], str]:
    if num_internal_prefixes(inst.branching, inst.depth) <= inst.query_budget:
        return full_tree_leaves(inst), "full_tree"
    c = args.chunk_size or choose_chunk_size(inst.num_agents, inst.depth, inst.query_budget)
    return chunked_leaves(inst, c), f"chunked(c={c})"


def nash_policy(inst, leaves, args, full: bool):
    """Algorithm 1 on a set of fully queried statements: Nash-welfare lottery, induced policy."""

    leaves = list(dict.fromkeys(tuple(x) for x in leaves))
    logu = leaf_logutils(inst, leaves)
    utilities = np.exp(logu - logu.max(axis=1, keepdims=True))
    init = None
    if args.init_seed is not None:
        init = np.random.default_rng(args.init_seed).dirichlet(np.ones(len(leaves)))
    result = solve_nash_welfare(utilities, tol=args.tol, max_iter=args.max_iter, init=init)
    p = result.distribution
    arr = np.asarray(leaves, dtype=np.int64)
    keep = p > 1e-14
    policy = induce_prefix_policy(arr[keep], p[keep] / p[keep].sum(), inst.branching)
    info = {"candidates": len(leaves), "support": int(keep.sum()), "fw_gap": result.gap,
            "fw_iterations": result.iterations}
    return policy, info


def solve_statement(inst, args):
    if args.lm_search == "lookahead":
        tokens = finite_lookahead(inst, depth=args.lookahead_depth, branching=args.lookahead_branching)
    else:
        tokens = beam_search(inst, width=args.beam_width)
    return path_policy(tokens, inst.branching), {"mode": args.lm_search}


def solve_instance(inst, args):
    if getattr(inst, "output", "lottery") == "statement":
        return solve_statement(inst, args)
    leaves, mode = candidate_leaves(inst, args)
    policy, info = nash_policy(inst, leaves, args, mode == "full_tree")
    info["mode"] = mode
    return policy, info


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk-size", type=int, default=0, help="0: finest size that fits the budget")
    parser.add_argument("--tol", type=float, default=1e-10)
    parser.add_argument("--max-iter", type=int, default=50_000)
    parser.add_argument("--init-seed", type=int, default=None)
    parser.add_argument("--lm-search", choices=("beam", "lookahead"), default="beam")
    parser.add_argument("--beam-width", type=int, default=4)
    parser.add_argument("--lookahead-depth", type=int, default=4)
    parser.add_argument("--lookahead-branching", type=int, default=2)
    return parser


def main(argv=None, solve_fn=solve_instance) -> None:
    args = build_parser().parse_args(argv)
    session = OracleSession()
    start = time.time()
    count = 0
    for inst in session.instances():
        t0 = time.time()
        policy, info = solve_fn(inst, args)
        inst.submit(policy)
        count += 1
        print(f"{inst.instance_id} queries={inst.queries_used}/{inst.query_budget} {info} "
              f"{time.time() - t0:.2f}s", file=sys.stderr, flush=True)
    print(f"solved {count} instances in {time.time() - start:.1f}s", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
