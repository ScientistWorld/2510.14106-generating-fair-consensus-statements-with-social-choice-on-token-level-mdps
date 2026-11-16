"""Client side of the solver protocol (method-agnostic helper; any solver may use it or not).

The evaluator starts a solver as ``bash scripts/solvers/<name>.sh`` with no arguments and talks to
it with one JSON object per line: evaluator -> solver on the solver's stdin, solver -> evaluator on
the solver's stdout (anything else a solver prints must go to stderr, which is logged).

    evaluator: {"type": "instance", "instance_id", "experiment", "branching", "depth",
                "num_agents", "rho", "query_budget", "output"}
    solver:    {"type": "query", "prefixes": [[a_1, ..., a_t], ...]}      (0 <= t < depth)
    evaluator: {"type": "logprobs", "logprobs": [ n x B per prefix ], "queries_used": k}
               log pi_i(a | s) of every agent i and token a at each prefix s (language-model
               trees also carry "base_logprobs": [ B per prefix ], the base model's
               log-probabilities of the same candidates);
               or {"type": "error", "message": ...} if a prefix is invalid or the batch would
               exceed the budget (nothing is charged then).
    solver:    {"type": "policy", "policy": {"": [p_0, ..., p_{B-1}], "0,2": [...], ...}}
    evaluator: {"type": "accepted"}, then the next instance, or {"type": "done"}.

Each distinct prefix costs one query; repeating a prefix is free. The policy maps every prefix
reachable with positive probability (comma-separated token indices, "" for the empty prefix) to
its next-token probabilities. ``output`` is "lottery" (a stochastic policy is scored as the
lottery it induces) or "statement" (the policy must be deterministic: it reaches exactly one
complete statement; ``path_policy`` builds one). ``rho`` is null on language-model trees.

Usage::

    session = OracleSession()
    for inst in session.instances():
        lp = inst.logprobs([])            # (num_agents, branching) at the empty prefix
        ...
        inst.submit(policy)
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np


class OracleError(RuntimeError):
    """The evaluator refused a query (invalid prefix or query budget exceeded)."""


class Instance:
    def __init__(self, session: "OracleSession", header: dict):
        self._session = session
        self.header = header
        self.instance_id = header["instance_id"]
        self.experiment = header["experiment"]
        self.branching = int(header["branching"])
        self.depth = int(header["depth"])
        self.num_agents = int(header["num_agents"])
        self.rho = None if header.get("rho") is None else float(header["rho"])
        self.query_budget = int(header["query_budget"])
        self.output = header.get("output", "lottery")
        self.queries_used = 0
        self.submitted = False
        self._cache: dict[tuple, np.ndarray] = {}
        self._base: dict[tuple, np.ndarray] = {}

    @property
    def queries_left(self) -> int:
        return self.query_budget - self.queries_used

    def known(self, prefix) -> bool:
        return tuple(int(a) for a in prefix) in self._cache

    def cached(self, prefix) -> np.ndarray:
        return self._cache[tuple(int(a) for a in prefix)]

    def cost(self, prefixes) -> int:
        """Number of queries a batch would be charged."""

        return len({tuple(int(a) for a in p) for p in prefixes} - self._cache.keys())

    def logprobs_many(self, prefixes) -> list[np.ndarray]:
        keys = [tuple(int(a) for a in p) for p in prefixes]
        missing = list(dict.fromkeys(k for k in keys if k not in self._cache))
        if missing:
            reply = self._session.request({"type": "query", "prefixes": [list(k) for k in missing]})
            if reply.get("type") != "logprobs":
                raise OracleError(reply.get("message", f"unexpected reply {reply.get('type')!r}"))
            for k, lp in zip(missing, reply["logprobs"]):
                self._cache[k] = np.asarray(lp, dtype=np.float64)
            for k, lp in zip(missing, reply.get("base_logprobs") or []):
                self._base[k] = np.asarray(lp, dtype=np.float64)
            self.queries_used = int(reply["queries_used"])
        return [self._cache[k] for k in keys]

    def logprobs(self, prefix) -> np.ndarray:
        return self.logprobs_many([prefix])[0]

    def base_logprobs_many(self, prefixes) -> list[np.ndarray]:
        """Base-model log-probabilities of the candidates (language-model trees; queries if needed)."""

        self.logprobs_many(prefixes)
        return [self._base[tuple(int(a) for a in p)] for p in prefixes]

    def base_logprobs(self, prefix) -> np.ndarray:
        return self.base_logprobs_many([prefix])[0]

    def submit(self, policy: dict) -> None:
        if self.submitted:
            raise RuntimeError("policy already submitted for this instance")
        reply = self._session.request({"type": "policy", "policy": policy})
        if reply.get("type") != "accepted":
            raise RuntimeError(f"policy not accepted: {reply}")
        self.submitted = True


def path_policy(tokens, branching: int) -> dict:
    """Deterministic policy that spells one complete statement (for "statement" instances)."""

    policy, key = {}, ""
    for a in tokens:
        row = [0.0] * branching
        row[int(a)] = 1.0
        policy[key] = row
        key = f"{key},{int(a)}" if key else f"{int(a)}"
    return policy


class OracleSession:
    def __init__(self):
        sys.stdout.flush()
        # Keep the protocol on a private copy of stdout; stray prints now go to stderr.
        self._out = os.fdopen(os.dup(1), "w", encoding="utf-8")
        os.dup2(2, 1)
        self._in = sys.stdin.buffer

    def _send(self, message: dict) -> None:
        self._out.write(json.dumps(message) + "\n")
        self._out.flush()

    def _recv(self) -> dict:
        line = self._in.readline()
        if not line:
            raise EOFError("evaluator closed the connection")
        return json.loads(line)

    def request(self, message: dict) -> dict:
        self._send(message)
        return self._recv()

    def instances(self):
        while True:
            message = self._recv()
            if message.get("type") == "done":
                return
            if message.get("type") != "instance":
                raise RuntimeError(f"unexpected message {message.get('type')!r}")
            inst = Instance(self, message)
            yield inst
            if not inst.submitted:
                raise RuntimeError(f"no policy submitted for {inst.instance_id}")
