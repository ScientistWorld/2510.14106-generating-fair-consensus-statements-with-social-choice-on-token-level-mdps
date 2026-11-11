"""Method-agnostic scoring for the token-tree ex-ante-core experiments.

An identical copy of this file lives in eval/train/ and eval/test/ (the two slices share no
code). It builds every instance's token tree in memory, starts each solver found in
scripts/solvers/, serves the agents' next-token log-probabilities to it through a query oracle
with a per-instance budget, executes the stochastic token policy the solver returns, and
recomputes every scored number from the full tree. Nothing a solver writes besides its policy
is read.

Experiments
-----------
empirical_core_test   Paper Section 4.2: B=3, L=4 trees (81 complete statements); the budget
                      covers every prefix. Failing the ex-ante core (alpha* > 1 + 1e-6) on any
                      instance fails the run.
deep_tree_core_test   B=3, L=12 trees (531,441 complete statements) with prefix-dependent
                      policies; a solver may query 1,500 of the 265,720 prefixes. Nash welfare is
                      compared with the exact optimum over the full tree, and blocking is
                      computed exactly for every coalition over every statement.

A run that crashes, exceeds the time limit, breaks the protocol, omits an instance, returns an
invalid policy, or fails the core constraint of empirical_core_test receives the worst value of
every scored metric in every experiment.
"""

from __future__ import annotations

import json
import math
import os
import re
import select
import signal
import subprocess
import time
import traceback
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

EMPIRICAL = "empirical_core_test"
DEEP = "deep_tree_core_test"
EXPERIMENTS = (EMPIRICAL, DEEP)

CORE_TOLERANCE = 1e-6
SNAP_TOLERANCE = 1e-6
DEEP_VIOLATION_TOLERANCE = 1e-3
ALPHA_CAP = 1e6
LOG10_ALPHA_CAP = 12.0
SIMPLEX_TOLERANCE = 1e-6
MAX_MESSAGE_BYTES = 256 * 1024 * 1024
MAX_PREFIXES_PER_QUERY = 100_000
CLOSE_GRACE_SECONDS = 10.0
BASELINES = ("uniform_policy", "utilitarian_policy", "random_dictatorship_policy")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
HEADER_KEYS = ("instance_id", "experiment", "branching", "depth", "num_agents", "rho", "query_budget")

WORST = {
    EMPIRICAL: {
        "max_coalition_improvement": ALPHA_CAP,
        "core_constraint_satisfied": 0.0,
        "nash_welfare_efficiency": 0.0,
    },
    DEEP: {
        "nash_welfare_efficiency": 0.0,
        "mean_log10_coalition_improvement": LOG10_ALPHA_CAP,
        "coalition_violation_rate": 1.0,
        "worst_agent_ideal_fraction": 0.0,
    },
}


class SolverFailure(Exception):
    """Raised when a solver's run cannot be scored."""


# ---------------------------------------------------------------------------
# Numerical helpers
# ---------------------------------------------------------------------------


def _logsumexp(x: np.ndarray, axis=None) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True)) + m
    if axis is None:
        return out.reshape(())
    return np.squeeze(out, axis=axis)


def _is_number(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _line_search(a: np.ndarray, d: np.ndarray, gmax: float) -> float:
    """argmax_{gamma in [0, gmax]} sum_i log(a_i + gamma d_i) (concave)."""

    if float(np.sum(d / (a + gmax * (1.0 - 1e-15) * d))) >= 0.0:
        return gmax
    lo, hi, g = 0.0, gmax, 0.0
    for _ in range(80):
        r = d / (a + g * d)
        f1 = float(r.sum())
        if f1 > 0:
            lo = g
        else:
            hi = g
        f2 = -float((r * r).sum())
        newton = g - f1 / f2 if f2 < 0 else 0.5 * (lo + hi)
        g_new = newton if lo < newton < hi else 0.5 * (lo + hi)
        if abs(f1) <= 1e-12 * float(np.abs(r).sum()) or abs(g_new - g) <= 1e-15 + 1e-12 * abs(g_new) \
                or hi - lo <= 1e-15 * max(1.0, gmax):
            return g_new
        g = g_new
    return g


def nash_fw(v: np.ndarray, *, tol: float, max_iter: int, init: np.ndarray | None = None):
    """Away-step Frank-Wolfe for max_p sum_i log(v_i^T p) on the simplex; returns (p, gap)."""

    v = np.asarray(v, dtype=np.float64)
    n, m = v.shape
    vt = np.ascontiguousarray(v.T)
    if init is None:
        p = np.full(m, 1.0 / m)
    else:
        p = np.maximum(np.asarray(init, dtype=np.float64), 0.0)
        p /= p.sum()
    a = v @ p
    gap = math.inf
    for it in range(max_iter):
        if it % 200 == 0:
            a = v @ p
        g = vt @ (1.0 / a)
        gp = float(g @ p)
        j_fw = int(np.argmax(g))
        gap = float(g[j_fw] - gp)
        if gap <= tol:
            break
        active = np.flatnonzero(p > 0)
        j_aw = int(active[np.argmin(g[active])])
        if gap >= gp - float(g[j_aw]):
            d = v[:, j_fw] - a
            gamma = _line_search(a, d, 1.0)
            p *= 1.0 - gamma
            p[j_fw] += gamma
        else:
            pa = float(p[j_aw])
            gmax = pa / (1.0 - pa)
            d = a - v[:, j_aw]
            gamma = _line_search(a, d, gmax)
            p *= 1.0 + gamma
            p[j_aw] -= gamma
            if gamma >= gmax * (1.0 - 1e-12):
                p[j_aw] = 0.0
        np.maximum(p, 0.0, out=p)
        p /= p.sum()
        a = a + gamma * d
    return p, gap


def nash_optimum(leaf_logu: np.ndarray, tol: float = 1e-10) -> tuple[float, float]:
    """Upper bound on max_p sum_i log U_i(p) over all statements, and its certified gap.

    Column generation: Frank-Wolfe on a growing set of statements, priced against every
    statement of the tree. The Frank-Wolfe gap over all statements bounds the sub-optimality.
    """

    n, m = leaf_logu.shape
    shift = leaf_logu.max(axis=1)
    v_all = np.exp(leaf_logu - shift[:, None])
    if m <= 4096:
        p, gap = nash_fw(v_all, tol=tol, max_iter=200_000)
        a = v_all @ p
        return float(np.log(a).sum() + shift.sum() + max(gap, 0.0)), gap
    vt_all = np.ascontiguousarray(v_all.T)
    cols = np.unique(np.r_[np.argmax(leaf_logu, axis=1), int(np.argmax(leaf_logu.sum(axis=0))),
                           int(np.argmax(v_all.sum(axis=0)))])
    p = None
    gap = math.inf
    a = None
    for _ in range(2000):
        sub = v_all[:, cols]
        p, _ = nash_fw(sub, tol=tol * 0.1, max_iter=200_000, init=p)
        a = sub @ p
        g = vt_all @ (1.0 / a)
        gp = float(g[cols] @ p)
        gap = float(g.max() - gp)
        if gap <= tol:
            break
        top = np.argpartition(-g, 64)[:64]
        known = set(cols.tolist())
        new = [int(j) for j in top[np.argsort(-g[top])] if g[j] > gp + 1e-13 and int(j) not in known]
        if not new:
            break
        keep = p > 0
        cols = np.r_[cols[keep], np.asarray(new, dtype=np.int64)]
        p = np.r_[p[keep], np.zeros(len(new))]
    return float(np.log(a).sum() + shift.sum() + max(gap, 0.0)), gap


def _game_lp(e: np.ndarray):
    """max_{q in simplex} min_i (e q)_i for a small nonnegative matrix; returns (value, duals)."""

    k, c = e.shape
    for method in ("highs", "highs-ipm", "highs-ds"):
        res = linprog(
            np.r_[np.zeros(c), -1.0],
            A_ub=np.hstack([-e, np.ones((k, 1))]),
            b_ub=np.zeros(k),
            A_eq=np.r_[np.ones(c), 0.0][None, :],
            b_eq=np.array([1.0]),
            bounds=[(0.0, None)] * (c + 1),
            method=method,
        )
        if res.success:
            break
    else:
        raise RuntimeError(f"coalition LP failed: {res.message}")
    lam = None
    marg = getattr(getattr(res, "ineqlin", None), "marginals", None)
    if marg is not None:
        lam = np.maximum(-np.asarray(marg, dtype=np.float64), 0.0)
        lam = lam / lam.sum() if lam.sum() > 0 else None
    return float(res.x[-1]), lam


def log_game_value(log_ratios: np.ndarray) -> float:
    """log of max_{q in simplex} min_i sum_x q_x exp(R[i, x]) over all columns x (exact).

    With ub = min_i max_x R[i, x], the value lies in [lower, ub] where lower = max(best single
    column, ub - log k) (mixing every member's best column). Columns whose every entry is below
    the lower bound are never in an optimal mixture and are dropped; entries are scaled by
    exp(-ub), so the LP value lies in [exp(lower - ub), 1] >= 1/k. The LP is solved directly when
    small and by column generation (pricing every remaining column) otherwise.
    """

    r = np.asarray(log_ratios, dtype=np.float64)
    k = r.shape[0]
    if k == 1:
        return float(r.max())
    ub = float(r.max(axis=1).min())
    lower = max(float(r.min(axis=0).max()), ub - math.log(k))
    if ub - lower <= 1e-12:
        return lower
    keep = np.flatnonzero(r.max(axis=0) >= lower - 1e-12)
    e = np.exp(np.minimum(r[:, keep] - ub, math.log(1e8)))
    e[e < 1e-14] = 0.0
    t_min = math.exp(lower - ub)
    if e.shape[1] <= 3000:
        t, _ = _game_lp(e)
        return ub + math.log(min(max(t, t_min), 1.0))
    cols = np.unique(np.r_[np.argmax(e, axis=1), int(np.argmax(e.min(axis=0)))])
    t = t_min
    for _ in range(1000):
        t, lam = _game_lp(e[:, cols])
        if lam is None:
            t, _ = _game_lp(e)
            break
        price = lam @ e
        if float(price.max()) <= t * (1.0 + 1e-10) + 1e-15:
            break
        top = np.argpartition(-price, 64)[:64]
        known = set(cols.tolist())
        new = [int(j) for j in top if price[j] > t * (1.0 + 1e-12) and int(j) not in known]
        if not new:
            break
        cols = np.r_[cols, np.asarray(new, dtype=np.int64)]
    return ub + math.log(min(max(t, t_min), 1.0))


def all_coalitions(n: int) -> list[list[int]]:
    """Every coalition of two or more agents, largest first (singletons are handled directly)."""

    return [list(s) for r in range(n, 1, -1) for s in combinations(range(n), r)]


def log_max_coalition_improvement(leaf_logu: np.ndarray, log_u: np.ndarray, coalitions) -> float:
    """log alpha*: the largest factor by which a coalition S, with probability mass |S|/n on any
    statements, can raise every member's expected utility (paper Section 4.2), exact.

    Statements that cannot be in any coalition's optimal mixture (every agent's ratio below the
    lowest per-coalition lower bound, see ``log_game_value``) are dropped once, up front.
    """

    n = log_u.shape[0]
    r_all = leaf_logu - log_u[:, None]
    rmax = r_all.max(axis=1)
    best = float((rmax + math.log(1.0 / n)).max())
    threshold = max(float(r_all.min(axis=0).max()), float(rmax.min()) - math.log(n))
    r_all = r_all[:, r_all.max(axis=0) >= threshold - 1e-12]
    for members in coalitions:
        size = len(members)
        if math.log(size / n) + float(rmax[members].min()) <= best + 1e-12:
            continue
        best = max(best, math.log(size / n) + log_game_value(r_all[members]))
    return best


# ---------------------------------------------------------------------------
# Token trees and policy execution
# ---------------------------------------------------------------------------


class Tree:
    """All agents' next-token log-probabilities at every prefix, and every statement's utility."""

    def __init__(self, levels: list):
        self.levels = [np.asarray(lp, dtype=np.float64) for lp in levels]
        self.num_agents, _, self.branching = self.levels[0].shape
        self.depth = len(self.levels)
        lu = np.zeros((self.num_agents, 1))
        for lp in self.levels:
            lu = (lu[:, :, None] + lp).reshape(self.num_agents, -1)
        self.leaf_logu = lu

    def prefix_logprobs(self, prefix: tuple) -> np.ndarray:
        lp = self.levels[len(prefix)]
        if lp.shape[1] == 1:
            return lp[:, 0, :]
        idx = 0
        for a in prefix:
            idx = idx * self.branching + a
        return lp[:, idx, :]


def _policy_rows(policy: dict, keys: list) -> list:
    rows = []
    for key in keys:
        row = policy.get(key)
        if row is None:
            raise SolverFailure(f"no next-token probabilities at reachable prefix '{key}'")
        rows.append(row)
    return rows


def execute_policy(policy, branching: int, depth: int):
    """Exactly execute a stochastic token policy; returns reachable statement indices and probabilities.

    A statement's index is the base-B number spelled by its tokens.
    """

    if not isinstance(policy, dict) or not policy:
        raise SolverFailure("policy must be a non-empty object mapping prefixes to next-token probabilities")
    keys = [""]
    ids = np.zeros(1, dtype=np.int64)
    masses = np.ones(1)
    for t in range(depth):
        rows = _policy_rows(policy, keys)
        for key, row in zip(keys, rows):
            if not isinstance(row, list) or len(row) != branching:
                raise SolverFailure(f"prefix '{key}' must map to a list of {branching} probabilities")
            for x in row:
                if not _is_number(x):
                    raise SolverFailure(f"non-numeric probability at prefix '{key}'")
        arr = np.asarray(rows, dtype=np.float64)
        bad = ~np.isfinite(arr).all(axis=1)
        if bad.any():
            raise SolverFailure(f"non-finite probability at prefix '{keys[int(np.argmax(bad))]}'")
        neg = (arr < -1e-12).any(axis=1)
        if neg.any():
            raise SolverFailure(f"negative probability at prefix '{keys[int(np.argmax(neg))]}'")
        arr = np.maximum(arr, 0.0)
        totals = arr.sum(axis=1)
        off = np.abs(totals - 1.0) > SIMPLEX_TOLERANCE
        if off.any():
            j = int(np.argmax(off))
            raise SolverFailure(f"next-token probabilities at prefix '{keys[j]}' sum to {float(totals[j])!r}, not 1")
        child = masses[:, None] * (arr / totals[:, None])
        parent, action = np.nonzero(child > 0.0)
        masses = child[parent, action]
        ids = ids[parent] * branching + action
        if t < depth - 1:
            keys = [f"{keys[pi]},{a}" if keys[pi] else f"{a}" for pi, a in zip(parent.tolist(), action.tolist())]
        if ids.size == 0:
            break
    if ids.size == 0:
        raise SolverFailure("policy reaches no complete statement")
    total = float(masses.sum())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise SolverFailure(f"executed policy has total probability {total!r}")
    return ids, masses / total


def log_expected_utilities(leaf_logu: np.ndarray, ids: np.ndarray, probs: np.ndarray) -> np.ndarray:
    return _logsumexp(leaf_logu[:, ids] + np.log(probs)[None, :], axis=1)


# ---------------------------------------------------------------------------
# Per-instance reference data, baselines and metrics
# ---------------------------------------------------------------------------


def reference_data(tree: Tree) -> dict:
    log_nw_upper, gap = nash_optimum(tree.leaf_logu)
    return {"log_nw_upper": log_nw_upper, "nw_gap": gap, "log_umax": tree.leaf_logu.max(axis=1)}


def greedy_statement(tree: Tree, agent: int) -> int:
    """Index of the statement agent ``agent`` decodes greedily (argmax token at every prefix)."""

    prefix: tuple = ()
    for _ in range(tree.depth):
        prefix = prefix + (int(np.argmax(tree.prefix_logprobs(prefix)[agent])),)
    idx = 0
    for a in prefix:
        idx = idx * tree.branching + a
    return idx


def baseline_utilities(tree: Tree) -> dict:
    """Evaluator-owned baselines: the paper's uniform policy (uniform over all statements) and
    utilitarian policy (the statement maximising the sum of utilities, exact), and random
    dictatorship over the participants' greedy statements (each with probability 1/n)."""

    lu = tree.leaf_logu
    n, m = lu.shape
    out = {"uniform_policy": _logsumexp(lu, axis=1) - math.log(m)}
    j = int(np.argmax(_logsumexp(lu, axis=0)))
    out["utilitarian_policy"] = lu[:, j].copy()
    fav, counts = np.unique([greedy_statement(tree, i) for i in range(n)], return_counts=True)
    out["random_dictatorship_policy"] = _logsumexp(lu[:, fav] + np.log(counts / n)[None, :], axis=1)
    return out


def instance_metrics(experiment: str, tree: Tree, ref: dict, log_u: np.ndarray, coalitions) -> dict:
    n = log_u.shape[0]
    log_alpha = log_max_coalition_improvement(tree.leaf_logu, log_u, coalitions)
    efficiency = math.exp(min(0.0, (float(log_u.sum()) - ref["log_nw_upper"]) / n))
    if efficiency >= 1.0 - SNAP_TOLERANCE:
        efficiency = 1.0
    row = {
        "nash_welfare_efficiency": efficiency,
        "worst_agent_ideal_fraction": float(np.exp(np.min(log_u - ref["log_umax"]))),
    }
    if experiment == EMPIRICAL:
        alpha = math.exp(min(log_alpha, math.log(ALPHA_CAP)))
        row["max_coalition_improvement"] = alpha
        row["blocked"] = 1.0 if alpha > 1.0 + CORE_TOLERANCE else 0.0
    else:
        log10_alpha = log_alpha / math.log(10.0)
        row["log10_coalition_improvement"] = float(min(max(log10_alpha, 0.0), LOG10_ALPHA_CAP))
        row["blocked"] = 1.0 if log10_alpha > math.log10(1.0 + DEEP_VIOLATION_TOLERANCE) else 0.0
    return row


def aggregate(experiment: str, rows: list[dict]) -> dict:
    eff = float(np.mean([r["nash_welfare_efficiency"] for r in rows]))
    if experiment == EMPIRICAL:
        alpha_max = float(min(max(r["max_coalition_improvement"] for r in rows), ALPHA_CAP))
        if alpha_max <= 1.0 + CORE_TOLERANCE:
            alpha_max = 1.0
        if eff >= 1.0 - SNAP_TOLERANCE:
            eff = 1.0
        return {
            "max_coalition_improvement": alpha_max,
            "core_constraint_satisfied": 1.0 if alpha_max <= 1.0 + CORE_TOLERANCE else 0.0,
            "nash_welfare_efficiency": eff,
        }
    return {
        "nash_welfare_efficiency": eff,
        "mean_log10_coalition_improvement": float(np.mean([r["log10_coalition_improvement"] for r in rows])),
        "coalition_violation_rate": float(np.mean([r["blocked"] for r in rows])),
        "worst_agent_ideal_fraction": float(np.mean([r["worst_agent_ideal_fraction"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# Solver sessions (JSON lines over the solver's stdin / stdout)
# ---------------------------------------------------------------------------


class SolverSession:
    """A running solver. Only the time spent waiting for the solver counts against its limit."""

    def __init__(self, script: Path, workspace: Path, log_path: Path, timeout: float):
        self.log = open(log_path, "wb")
        self.proc = subprocess.Popen(
            ["bash", str(script)],
            cwd=str(workspace),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            start_new_session=True,
        )
        self.timeout = timeout
        self.solver_time = 0.0
        self.buf = bytearray()
        self.in_fd = self.proc.stdin.fileno()
        self.out_fd = self.proc.stdout.fileno()
        os.set_blocking(self.in_fd, False)

    def _remaining(self) -> float:
        remaining = self.timeout - self.solver_time
        if remaining <= 0:
            raise SolverFailure(f"solver exceeded its {self.timeout:.0f}s time limit")
        return remaining

    def send(self, message: dict) -> None:
        data = memoryview((json.dumps(message) + "\n").encode())
        while len(data):
            start = time.time()
            _, writable, _ = select.select([], [self.in_fd], [], self._remaining())
            self.solver_time += time.time() - start
            if not writable:
                continue
            try:
                k = os.write(self.in_fd, data)
            except BlockingIOError:
                continue
            except (BrokenPipeError, OSError):
                raise SolverFailure(f"solver stopped reading (exit code {self.proc.poll()})") from None
            data = data[k:]

    def recv(self) -> dict:
        while True:
            nl = self.buf.find(b"\n")
            if nl >= 0:
                line = bytes(self.buf[:nl])
                del self.buf[: nl + 1]
                break
            if len(self.buf) > MAX_MESSAGE_BYTES:
                raise SolverFailure("solver message exceeds the size limit")
            start = time.time()
            readable, _, _ = select.select([self.out_fd], [], [], self._remaining())
            if readable:
                chunk = os.read(self.out_fd, 1 << 20)
            self.solver_time += time.time() - start
            if not readable:
                continue
            if not chunk:
                rc = self.proc.poll()
                raise SolverFailure(f"solver closed its output before finishing (exit code {rc})")
            self.buf += chunk
        try:
            message = json.loads(line)
        except Exception:
            raise SolverFailure(f"malformed message from solver: {line[:200]!r}") from None
        if not isinstance(message, dict):
            raise SolverFailure("solver messages must be JSON objects")
        return message

    def close(self) -> int | None:
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        deadline = time.time() + CLOSE_GRACE_SECONDS
        while self.proc.poll() is None and time.time() < deadline:
            time.sleep(0.05)
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        rc = self.proc.wait()
        try:
            self.proc.stdout.close()
        except Exception:
            pass
        self.log.close()
        return rc


def _valid_prefix(p, branching: int, depth: int) -> bool:
    return (
        isinstance(p, list)
        and len(p) < depth
        and all(isinstance(a, int) and not isinstance(a, bool) and 0 <= a < branching for a in p)
    )


def serve_instance(session: SolverSession, spec: dict, tree: Tree):
    """Send one instance, answer its queries within budget, return (policy, queries used)."""

    session.send({"type": "instance", **{k: spec[k] for k in HEADER_KEYS}})
    budget = int(spec["query_budget"])
    seen: set = set()
    while True:
        message = session.recv()
        kind = message.get("type")
        if kind == "query":
            prefixes = message.get("prefixes")
            if not isinstance(prefixes, list) or not prefixes or len(prefixes) > MAX_PREFIXES_PER_QUERY:
                session.send({"type": "error", "message": "'prefixes' must be a non-empty list"})
                continue
            if not all(_valid_prefix(p, tree.branching, tree.depth) for p in prefixes):
                session.send({"type": "error", "message": f"prefixes must be token lists of length < {tree.depth} "
                                                          f"with entries in 0..{tree.branching - 1}"})
                continue
            keys = [tuple(p) for p in prefixes]
            new = set(keys) - seen
            if len(seen) + len(new) > budget:
                session.send({"type": "error", "message": f"query budget exceeded: {len(seen)} of {budget} used, "
                                                          f"batch needs {len(new)} more"})
                continue
            seen |= new
            session.send({
                "type": "logprobs",
                "logprobs": [tree.prefix_logprobs(k).tolist() for k in keys],
                "queries_used": len(seen),
            })
        elif kind == "policy":
            session.send({"type": "accepted"})
            return message.get("policy"), len(seen)
        else:
            raise SolverFailure(f"unexpected message type {kind!r} from solver")


def discover_solvers(workspace: Path) -> list[Path]:
    solver_dir = workspace / "scripts" / "solvers"
    if not solver_dir.is_dir():
        return []
    return sorted(p for p in solver_dir.glob("*.sh") if p.is_file())


# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def _by_rho(specs: list[dict], rows: list[dict], key: str, reducer) -> dict:
    groups: dict[str, list[float]] = {}
    for spec, row in zip(specs, rows):
        groups.setdefault(f"{spec['rho']:.4g}", []).append(row[key])
    return {rho: float(reducer(vals)) for rho, vals in groups.items()}


def _solver_name(script: Path, used: set) -> str:
    name = script.stem
    if not NAME_RE.match(name) or name in used:
        # Every solver on disk is scored; names that are invalid or collide with an evaluator
        # baseline are reported under a distinct key.
        base = "solver_" + re.sub(r"[^A-Za-z0-9_]", "_", name)[:50]
        name, k = base, 1
        while name in used:
            name, k = f"{base}_{k}", k + 1
    used.add(name)
    return name


def run_evaluation(specs: list[dict], build_levels, *, split: str, scores_path: Path, workspace: Path,
                   timeout: float, log_dir: Path) -> dict:
    """Instance by instance: build the tree once, score the evaluator baselines, then serve the
    instance to every solver still running (all solvers run side by side, each waiting for its
    next instance) and score its policy."""

    t0 = time.time()
    by_exp = {exp: [s for s in specs if s["experiment"] == exp] for exp in EXPERIMENTS}
    coalitions = {n: all_coalitions(n) for n in {int(s["num_agents"]) for s in specs}}
    log_dir.mkdir(parents=True, exist_ok=True)

    used_names: set[str] = set(BASELINES)
    sessions: dict[str, SolverSession] = {}
    failures: dict[str, list[str]] = {}
    for script in discover_solvers(workspace):
        name = _solver_name(script, used_names)
        try:
            sessions[name] = SolverSession(script, workspace, log_dir / f"{split}_{name}.log", timeout)
        except Exception as exc:
            failures[name] = [f"could not start solver: {exc!r}"]
    alive = sorted(sessions)
    print(f"[{split}] {len(specs)} instances, {len(sessions)} solvers: {', '.join(alive)}", flush=True)

    raw: dict[str, dict[str, list[dict]]] = {exp: {} for exp in EXPERIMENTS}
    used_queries: dict[str, dict[str, list[int]]] = {name: {exp: [] for exp in EXPERIMENTS} for name in sessions}
    refs: dict[str, dict] = {}

    def fail(name: str, reason: str) -> None:
        failures.setdefault(name, []).append(reason)
        if name in alive:
            alive.remove(name)
        sessions[name].close()

    for count, spec in enumerate(specs, 1):
        exp = spec["experiment"]
        tree = Tree(build_levels(spec))
        ref = refs[spec["instance_id"]] = reference_data(tree)
        for name, log_u in baseline_utilities(tree).items():
            raw[exp].setdefault(name, []).append(
                instance_metrics(exp, tree, ref, log_u, coalitions[tree.num_agents]))
        for name in list(alive):
            try:
                policy, n_queries = serve_instance(sessions[name], spec, tree)
                ids, probs = execute_policy(policy, tree.branching, tree.depth)
                log_u = log_expected_utilities(tree.leaf_logu, ids, probs)
                row = instance_metrics(exp, tree, ref, log_u, coalitions[tree.num_agents])
                raw[exp].setdefault(name, []).append(row)
                used_queries[name][exp].append(n_queries)
                if exp == EMPIRICAL and row["blocked"] > 0:
                    # The core constraint fails the whole run; stop serving this solver.
                    fail(name, f"ex-ante core violated on {spec['instance_id']} "
                               f"(alpha* {row['max_coalition_improvement']:.6g} > 1 + {CORE_TOLERANCE:g})")
            except SolverFailure as exc:
                fail(name, f"instance {count} of {len(specs)} ({spec['instance_id']}): {exc}")
            except Exception as exc:  # evaluator-side error on a malformed output
                traceback.print_exc()
                fail(name, f"could not score output for {spec['instance_id']}: {exc!r}")
        if count % 20 == 0 or count == len(specs):
            print(f"[{split}] {count}/{len(specs)} instances scored ({time.time() - t0:.1f}s); "
                  f"{len(alive)} solvers running", flush=True)

    for name in list(alive):
        try:
            sessions[name].send({"type": "done"})
        except SolverFailure as exc:
            failures.setdefault(name, []).append(f"after the last instance: {exc}")
        rc = sessions[name].close()
        if name not in failures and rc not in (0, -signal.SIGKILL):
            failures[name] = [f"solver exited with code {rc}"]

    solver_time = {name: round(sess.solver_time, 1) for name, sess in sessions.items()}
    queries = {name: {exp: float(np.mean(v)) for exp, v in per.items() if v} for name, per in used_queries.items()}
    for name in sessions:
        if name in failures:
            continue
        bad = [r for r in raw[EMPIRICAL].get(name, []) if r["blocked"] > 0]
        if bad:
            failures[name] = [
                f"ex-ante core violated on {len(bad)} of {len(raw[EMPIRICAL][name])} {EMPIRICAL} instances "
                f"(max alpha* {max(r['max_coalition_improvement'] for r in bad):.6g})"
            ]
    for exp in EXPERIMENTS:
        for name in list(raw[exp]):
            if name not in BASELINES and len(raw[exp][name]) != len(by_exp[exp]):
                del raw[exp][name]  # incomplete: the run failed before this experiment finished
    for name in sessions:
        status = "FAILED: " + failures[name][0][:300] if name in failures else "ok"
        print(f"[{split}] solver {name}: {status} (solver time {solver_time[name]:.1f}s)", flush=True)

    method_names = list(BASELINES) + sorted(n for n in set(sessions) | set(failures) if n not in BASELINES)
    scores = {"experiments": {}, "_meta": {
        "split": split,
        "solvers_failed": failures,
        "solver_time_seconds": solver_time,
        "mean_queries_used": queries,
        "evaluation_seconds": None,
    }}
    for exp in EXPERIMENTS:
        results, measured, per_rho = {}, {}, {}
        for name in method_names:
            rows = raw[exp].get(name)
            if rows:
                measured[name] = aggregate(exp, rows)
                if exp == EMPIRICAL:
                    per_rho[name] = _by_rho(by_exp[exp], rows, "max_coalition_improvement", np.max)
                else:
                    per_rho[name] = {
                        key: _by_rho(by_exp[exp], rows, key, np.mean)
                        for key in ("nash_welfare_efficiency", "log10_coalition_improvement", "blocked")
                    }
            results[name] = dict(WORST[exp]) if (name in failures or not rows) else measured[name]
        scores["experiments"][exp] = {
            "results": results,
            "_details": {
                "split": split,
                "num_instances": len(by_exp[exp]),
                "query_budget": sorted({int(s["query_budget"]) for s in by_exp[exp]}),
                "rhos": sorted({float(s["rho"]) for s in by_exp[exp]}),
                "measured_before_failure_masking": measured,
                "per_rho": per_rho,
                "per_instance": {
                    name: [{"instance_id": s["instance_id"], "rho": s["rho"], **row}
                           for s, row in zip(by_exp[exp], rows)]
                    for name, rows in raw[exp].items()
                },
                "exact_optimum_gap_max": float(max(refs[s["instance_id"]]["nw_gap"] for s in by_exp[exp]))
                if by_exp[exp] else None,
            },
        }
    scores["_meta"]["evaluation_seconds"] = round(time.time() - t0, 1)
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    summary = {exp: scores["experiments"][exp]["results"] for exp in EXPERIMENTS}
    print(json.dumps(summary, indent=1))
    if failures:
        print(json.dumps({"failed": {k: v[:1] for k, v in failures.items()}}, indent=1))
    print(f"[{split}] wrote {scores_path} ({time.time() - t0:.1f}s)", flush=True)
    return scores
