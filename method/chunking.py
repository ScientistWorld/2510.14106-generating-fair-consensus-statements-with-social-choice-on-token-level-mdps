"""Token chunking (paper Section 4.3) under the oracle's query budget.

Positions are partitioned into contiguous blocks of ``chunk_size`` tokens and decoding selects
whole blocks. The paper leaves open how the candidate macro-actions are proposed; here each
block's candidates at a chunk-start prefix are every participant's most likely continuation of
the block (greedy decoding under that participant's policy, so each viewpoint is represented in
the chunked statement set) and the greedy continuation under the base model, the equal-weight
mixture of the participants' policies (the analogue of a base model prompted with every
opinion). The chunked statement set C_K is the tree of block choices; Algorithm 1 run on C_K
yields a lottery in the ex-ante core relative to C_K.

The finest chunk size whose worst-case query cost fits the instance's budget is used; a tree
small enough to query completely is enumerated instead.
"""

from __future__ import annotations

import math

import numpy as np


def _mixture_logprobs(lp: np.ndarray) -> np.ndarray:
    m = lp.max(axis=0)
    return np.log(np.exp(lp - m).sum(axis=0)) + m - math.log(lp.shape[0])


def block_candidates(inst, start: tuple, length: int) -> list[tuple]:
    """Greedy blocks of ``length`` tokens from prefix ``start``: one per agent, one for the mixture."""

    n = inst.num_agents
    paths = [list(start) for _ in range(n + 1)]  # rows 0..n-1: agents, row n: mixture
    for _ in range(length):
        lps = inst.logprobs_many([tuple(p) for p in paths])
        for k, (path, lp) in enumerate(zip(paths, lps)):
            row = lp[k] if k < n else _mixture_logprobs(lp)
            path.append(int(np.argmax(row)))
    blocks = []
    for path in [paths[n]] + paths[:n]:
        block = tuple(path[len(start):])
        if block not in blocks:
            blocks.append(block)
    return blocks


def chunk_query_cost(num_agents: int, depth: int, chunk_size: int) -> int:
    """Worst-case number of distinct prefixes queried to build the chunked statement set."""

    k = num_agents + 1
    total, starts = 0, 1
    for begin in range(0, depth, chunk_size):
        length = min(chunk_size, depth - begin)
        total += starts * (1 + k * (length - 1))
        starts *= k
    return total


def choose_chunk_size(num_agents: int, depth: int, budget: int) -> int:
    for c in range(1, depth + 1):
        if chunk_query_cost(num_agents, depth, c) <= budget:
            return c
    return depth


def chunked_leaves(inst, chunk_size: int) -> list[tuple]:
    frontier = [()]
    for begin in range(0, inst.depth, chunk_size):
        length = min(chunk_size, inst.depth - begin)
        frontier = [prefix + block for prefix in frontier for block in block_candidates(inst, prefix, length)]
    return frontier


def full_tree_leaves(inst) -> list[tuple]:
    """Query every prefix (level by level) and return every complete statement."""

    level = [()]
    for _ in range(inst.depth):
        inst.logprobs_many(level)
        level = [prefix + (a,) for prefix in level for a in range(inst.branching)]
    return level
