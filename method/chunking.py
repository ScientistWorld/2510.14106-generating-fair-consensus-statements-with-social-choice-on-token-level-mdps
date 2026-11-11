"""Token chunking (paper Section 4.3).

Positions are partitioned into contiguous blocks of ``chunk_size`` tokens and decoding selects
whole blocks. At every block the candidate macro-actions are the ``num_candidates`` most likely
blocks under the base model, here the equal-weight mixture of the agents' block distributions
(the analogue of a base model prompted with every opinion; agent policies in this environment
do not depend on the prefix, so neither do the proposals). The chunked statement set C_K is the
product of the per-block candidates; Algorithm 1 is then run on C_K, which yields a lottery in
the ex-ante core relative to C_K.
"""

from __future__ import annotations

import math
from itertools import product

import numpy as np

from method.nash_lottery import leaf_logutils


def _logsumexp0(x: np.ndarray) -> np.ndarray:
    m = x.max(axis=0)
    return np.log(np.exp(x - m).sum(axis=0)) + m


def chunk_candidates(logp: np.ndarray, chunk_size: int, num_candidates: int) -> list[np.ndarray]:
    n, depth, branching = logp.shape
    blocks_per_chunk = []
    for start in range(0, depth, chunk_size):
        stop = min(start + chunk_size, depth)
        blocks = np.asarray(list(product(range(branching), repeat=stop - start)), dtype=np.int64)
        block_logu = leaf_logutils(logp[:, start:stop, :], blocks)
        mixture = _logsumexp0(block_logu) - math.log(n)
        order = np.argsort(-mixture, kind="stable")[:num_candidates]
        blocks_per_chunk.append(blocks[order])
    return blocks_per_chunk


def chunked_leaves(blocks_per_chunk: list[np.ndarray]) -> np.ndarray:
    leaves = blocks_per_chunk[0]
    for blocks in blocks_per_chunk[1:]:
        leaves = np.concatenate(
            [np.repeat(leaves, len(blocks), axis=0), np.tile(blocks, (len(leaves), 1))], axis=1
        )
    return leaves
