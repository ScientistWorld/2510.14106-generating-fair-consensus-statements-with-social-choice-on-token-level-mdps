"""Single consensus statements by egalitarian-welfare search in the token-level MDP (paper Section 5).

Rewards are r_i(s, a) = log pi_i(a | s) (beta = 1), so a statement's utility for participant i is
U_i^log(X) = sum_t log pi_i(s_{t-1}, a_t) and its egalitarian welfare is min_i U_i^log(X)
(paper Equation 1). With a fixed statement length L, maximising it minimises the egalitarian
perplexity EPPL(X) = max_i exp(-U_i^log(X) / L).

* ``beam_search``: paper Algorithm 3. Keep the ``width`` best partial statements; expand every
  beam by all B candidate tokens, score each candidate by min_i U_i^log, keep the top ``width``;
  return the best statement of the final beam. The paper's main experiment uses width 4 (the
  authors' code expands each beam with up to ``width`` sampled tokens; here each beam is expanded
  by the tree's B candidates).
* ``finite_lookahead``: paper Algorithm 2 (depth d=4, branching 2 in the paper). At every step,
  enumerate the paths of length min(d, remaining) below the current state through each node's
  ``branching`` most likely tokens under the base model, score each path P by
  min_i U_i^log(X_prefix || P), and commit the first token of the best one.

Both query the oracle only at prefixes whose next-token log-probabilities they use.
"""

from __future__ import annotations

import numpy as np


def beam_search(inst, width: int = 4) -> tuple:
    n, b, depth = inst.num_agents, inst.branching, inst.depth
    beam = [((), np.zeros(n))]
    for _ in range(depth):
        lps = inst.logprobs_many([prefix for prefix, _ in beam])
        candidates = []
        for (prefix, u), lp in zip(beam, lps):
            for a in range(b):
                nu = u + lp[:, a]
                candidates.append((float(nu.min()), prefix + (a,), nu))
        candidates.sort(key=lambda c: -c[0])  # stable: ties keep beam order, then token order
        beam = [(c[1], c[2]) for c in candidates[:width]]
    return beam[0][0]


def finite_lookahead(inst, depth: int = 4, branching: int = 2) -> tuple:
    n, length = inst.num_agents, inst.depth
    prefix: tuple = ()
    u = np.zeros(n)
    for t in range(length):
        horizon = min(depth, length - t)
        frontier = [((), u)]
        for _ in range(horizon):
            nodes = [prefix + rel for rel, _ in frontier]
            lps = inst.logprobs_many(nodes)
            bases = inst.base_logprobs_many(nodes)
            nxt = []
            for (rel, uu), lp, base in zip(frontier, lps, bases):
                for a in np.argsort(-base, kind="stable")[:branching]:
                    nxt.append((rel + (int(a),), uu + lp[:, int(a)]))
            frontier = nxt
        best_rel = max(frontier, key=lambda item: float(item[1].min()))[0]
        a = best_rel[0]
        u = u + inst.logprobs(prefix)[:, a]
        prefix = prefix + (a,)
    return prefix
