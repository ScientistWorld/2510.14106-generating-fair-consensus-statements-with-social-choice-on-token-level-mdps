"""Method-agnostic scoring for the token-tree ex-ante-core experiments.

An identical copy of this file lives in eval/train/ and eval/test/ (the two slices share no
code). It runs every solver found in scripts/solvers/ on freshly written instance files,
executes the stochastic token policy each solver returns, and recomputes every scored number
from the agents' policies held in memory by the evaluator. Nothing a solver writes besides its
policy is read.

Experiments
-----------
empirical_core_test   B=3, L=4 trees (81 complete statements). Every coalition's blocking LP is
                      solved over all statements. Failing the ex-ante core (alpha* > 1 + 1e-6)
                      on any instance fails the run.
deep_tree_core_test   B=3, L=24 trees (3^24 statements). Blocking by coalitions of up to three
                      agents is computed exactly (see ``hull_vertex_leaves``), Nash welfare is
                      computed exactly for the executed policy.

A run that crashes, times out, omits an instance, returns an invalid policy, or fails the
core constraint of empirical_core_test receives the worst value of every scored metric in
every experiment.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import traceback
from itertools import combinations, product
from pathlib import Path

import numpy as np
from scipy.optimize import linprog
from scipy.spatial import ConvexHull

EMPIRICAL = "empirical_core_test"
DEEP = "deep_tree_core_test"
EXPERIMENTS = (EMPIRICAL, DEEP)

CORE_TOLERANCE = 1e-6
SNAP_TOLERANCE = 1e-6
DEEP_VIOLATION_TOLERANCE = 1e-3
ALPHA_CAP = 1e6
LOG10_ALPHA_CAP = 12.0
MAX_SUPPORT = 100_000
MAX_OUTPUT_BYTES = 200 * 1024 * 1024
SIMPLEX_TOLERANCE = 1e-6
UTILITARIAN_BEAM_WIDTH = 4096
BASELINES = ("uniform_policy", "utilitarian_policy", "random_dictatorship_policy")
NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]{0,63}$")

WORST = {
    EMPIRICAL: {
        "max_coalition_improvement": ALPHA_CAP,
        "core_constraint_satisfied": 0.0,
        "nash_welfare_efficiency": 0.0,
    },
    DEEP: {
        "nash_welfare_ideal_fraction": 0.0,
        "mean_log10_coalition_improvement": LOG10_ALPHA_CAP,
        "coalition_violation_rate": 1.0,
        "worst_agent_ideal_fraction": 0.0,
    },
}


class SolverFailure(Exception):
    """Raised when a solver's output cannot be scored."""


# ---------------------------------------------------------------------------
# Small numerical helpers
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


def leaf_logutils(logp: np.ndarray, leaves: np.ndarray) -> np.ndarray:
    """log U_i^prob(X) for each agent row of ``logp`` (k, L, B) and leaf row, shape (k, m)."""

    leaves = np.asarray(leaves, dtype=np.int64)
    k, depth, _ = logp.shape
    out = np.zeros((k, leaves.shape[0]), dtype=np.float64)
    for t in range(depth):
        out += logp[:, t, leaves[:, t]]
    return out


def log_game_value(log_ratios: np.ndarray) -> float:
    """log of max_{q in simplex} min_i sum_x q_x exp(R[i, x]) (value of a small matrix game)."""

    r = np.asarray(log_ratios, dtype=np.float64)
    k = r.shape[0]
    if k == 1:
        return float(r.max())
    v0 = float(r.min(axis=0).max())
    keep = r.max(axis=0) >= v0 - 1e-12
    s = r[:, keep] - v0
    cap = float(s.max(axis=1).min()) + math.log(1e6)
    mat = np.exp(np.minimum(s, cap))
    cols = mat.shape[1]
    if cols == 1:
        return v0
    c = np.zeros(cols + 1)
    c[-1] = -1.0
    a_ub = np.hstack([-mat, np.ones((k, 1))])
    b_ub = np.zeros(k)
    a_eq = np.zeros((1, cols + 1))
    a_eq[0, :cols] = 1.0
    res = linprog(
        c,
        A_ub=a_ub,
        b_ub=b_ub,
        A_eq=a_eq,
        b_eq=np.array([1.0]),
        bounds=[(0.0, None)] * (cols + 1),
        method="highs",
    )
    if not res.success:
        raise RuntimeError(f"coalition LP failed: {res.message}")
    return v0 + math.log(max(float(res.x[-1]), 1.0))


def nash_welfare_fw(v: np.ndarray, *, tol: float = 1e-10, max_iter: int = 50_000):
    """Evaluator-side Nash-welfare optimum on a utility matrix (agents x outcomes).

    Away-step Frank-Wolfe; returns (distribution, FW duality gap). The gap upper-bounds the
    sub-optimality of sum_i log(v_i^T p).
    """

    v = np.asarray(v, dtype=np.float64)
    n, m = v.shape
    vt = np.ascontiguousarray(v.T)
    p = np.full(m, 1.0 / m)
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
        gap_aw = gp - float(g[j_aw])
        if gap >= gap_aw:
            d = v[:, j_fw] - a
            gmax = 1.0
            away = False
        else:
            pa = float(p[j_aw])
            d = a - v[:, j_aw]
            gmax = pa / (1.0 - pa)
            away = True
        gamma = _line_search(a, d, gmax)
        if away:
            p *= 1.0 + gamma
            p[j_aw] -= gamma
            if gamma >= gmax * (1.0 - 1e-12):
                p[j_aw] = 0.0
        else:
            p *= 1.0 - gamma
            p[j_fw] += gamma
        np.maximum(p, 0.0, out=p)
        p /= p.sum()
        a = a + gamma * d
    return p, gap


def _line_search(a: np.ndarray, d: np.ndarray, gmax: float) -> float:
    """argmax_{gamma in [0, gmax]} sum_i log(a_i + gamma d_i) (concave)."""

    def deriv(g):
        return float(np.sum(d / (a + g * d)))

    if deriv(gmax * (1.0 - 1e-15)) >= 0.0:
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
        if abs(g_new - g) <= 1e-13 * max(g_new, 1e-300) or hi - lo <= 1e-15 * max(1.0, gmax):
            return g_new
        g = g_new
    return g


# ---------------------------------------------------------------------------
# Policy execution
# ---------------------------------------------------------------------------


def execute_policy(policy, branching: int, depth: int, max_support: int = MAX_SUPPORT):
    """Exactly execute a stochastic token policy; returns reachable leaves and their probabilities."""

    if not isinstance(policy, dict) or not policy:
        raise SolverFailure("policy must be a non-empty object mapping prefixes to next-token probabilities")
    prefixes: list[tuple] = [()]
    masses: list[float] = [1.0]
    for _ in range(depth):
        new_prefixes: list[tuple] = []
        new_masses: list[float] = []
        for prefix, mass in zip(prefixes, masses):
            key = ",".join(str(a) for a in prefix)
            row = policy.get(key)
            if row is None:
                raise SolverFailure(f"no next-token probabilities at reachable prefix '{key}'")
            if not isinstance(row, list) or len(row) != branching:
                raise SolverFailure(f"prefix '{key}' must map to a list of {branching} probabilities")
            vals = []
            for x in row:
                if not _is_number(x) or not math.isfinite(float(x)):
                    raise SolverFailure(f"non-numeric or non-finite probability at prefix '{key}'")
                if x < -1e-12:
                    raise SolverFailure(f"negative probability at prefix '{key}'")
                vals.append(max(float(x), 0.0))
            total = sum(vals)
            if abs(total - 1.0) > SIMPLEX_TOLERANCE:
                raise SolverFailure(f"next-token probabilities at prefix '{key}' sum to {total!r}, not 1")
            for action, x in enumerate(vals):
                if x > 0.0:
                    child = mass * x / total
                    if child > 0.0:
                        new_prefixes.append(prefix + (action,))
                        new_masses.append(child)
            if len(new_prefixes) > max_support:
                raise SolverFailure(f"policy reaches more than {max_support} prefixes")
        prefixes, masses = new_prefixes, new_masses
    if not prefixes:
        raise SolverFailure("policy reaches no complete statement")
    leaves = np.asarray(prefixes, dtype=np.int64).reshape(len(prefixes), depth)
    probs = np.asarray(masses, dtype=np.float64)
    total = float(probs.sum())
    if not math.isfinite(total) or abs(total - 1.0) > 1e-6:
        raise SolverFailure(f"executed policy has total probability {total!r}")
    return leaves, probs / total


def log_expected_utilities(logp: np.ndarray, leaves: np.ndarray, probs: np.ndarray) -> np.ndarray:
    lu = leaf_logutils(logp, leaves)
    return _logsumexp(lu + np.log(probs)[None, :], axis=1)


# ---------------------------------------------------------------------------
# Instance contexts
# ---------------------------------------------------------------------------


def _hull_vertices(points: np.ndarray) -> np.ndarray:
    m, k = points.shape
    if m <= k + 1:
        return np.arange(m)
    if k == 1:
        return np.unique([int(np.argmin(points[:, 0])), int(np.argmax(points[:, 0]))])
    centered = points - points.mean(axis=0)
    scale = float(np.abs(centered).max()) or 1.0
    for options in (None, "QJ"):
        try:
            hull = ConvexHull(centered / scale) if options is None else ConvexHull(
                centered / scale, qhull_options=options
            )
            return np.sort(hull.vertices)
        except Exception:  # degenerate input; try joggled hull, then keep everything
            continue
    return np.arange(m)


def hull_vertex_leaves(y: np.ndarray) -> np.ndarray:
    """Complete statements whose coalition log-utility vector is a vertex of the hull of all of them.

    ``y`` has shape (L, B, k): log-probability of token a at depth t for each of k agents. The
    log-utility vector of a statement is a sum over depths, so the set of all such vectors is a
    Minkowski sum and its hull is built one depth at a time. For any non-negative weights
    gamma, sum_i gamma_i exp(y_i) is convex in y, so its maximum over all B^L statements is
    attained at one of the returned statements; the coalition blocking LP restricted to them
    therefore equals the LP over the full tree.
    """

    depth, branching, _ = y.shape
    m0 = 1
    while branching ** m0 < 64 and m0 < depth:
        m0 += 1
    leaves = np.asarray(list(product(range(branching), repeat=m0)), dtype=np.int64)
    points = np.zeros((leaves.shape[0], y.shape[2]))
    for t in range(m0):
        points += y[t, leaves[:, t]]
    keep = _hull_vertices(points)
    leaves, points = leaves[keep], points[keep]
    for t in range(m0, depth):
        count = leaves.shape[0]
        leaves = np.concatenate(
            [np.repeat(leaves, branching, axis=0), np.tile(np.arange(branching), count)[:, None]], axis=1
        )
        points = np.repeat(points, branching, axis=0) + np.tile(y[t], (count, 1))
        keep = _hull_vertices(points)
        leaves, points = leaves[keep], points[keep]
    return leaves


def _instance_logp(instance: dict) -> np.ndarray:
    logp = np.asarray(instance["agent_token_logprobs"], dtype=np.float64)
    n, depth, branching = logp.shape
    assert (n, depth, branching) == (instance["num_agents"], instance["depth"], instance["branching"])
    return logp


def empirical_context(instance: dict) -> dict:
    logp = _instance_logp(instance)
    n, depth, branching = logp.shape
    leaves = np.asarray(list(product(range(branching), repeat=depth)), dtype=np.int64)
    logu = leaf_logutils(logp, leaves)
    shift = logu.max(axis=1, keepdims=True)
    p_star, gap = nash_welfare_fw(np.exp(logu - shift))
    log_u_star = np.log(np.exp(logu - shift) @ p_star) + shift[:, 0]
    return {
        "logp": logp,
        "leaves": leaves,
        "logu": logu,
        "nw_upper": float(log_u_star.sum() + gap),
        "coalitions": [list(s) for r in range(1, n + 1) for s in combinations(range(n), r)],
        "radix": branching ** np.arange(depth - 1, -1, -1),
    }


def deep_context(instance: dict) -> dict:
    logp = _instance_logp(instance)
    n = logp.shape[0]
    favorites = logp.argmax(axis=2)
    candidates = {}
    for size in (2, 3):
        for coalition in combinations(range(n), size):
            members = list(coalition)
            y = np.transpose(logp[members], (1, 2, 0))
            verts = np.unique(np.concatenate([hull_vertex_leaves(y), favorites], axis=0), axis=0)
            candidates[coalition] = leaf_logutils(logp[members], verts)
    return {
        "logp": logp,
        "favorites": favorites,
        "logu_max": logp.max(axis=2).sum(axis=1),
        "candidates": candidates,
        "num_candidates": int(sum(v.shape[1] for v in candidates.values())),
    }


# ---------------------------------------------------------------------------
# Per-instance metrics
# ---------------------------------------------------------------------------


def empirical_metrics(ctx: dict, log_u: np.ndarray) -> dict:
    n = log_u.shape[0]
    alpha_log = -math.inf
    for members in ctx["coalitions"]:
        value = log_game_value(ctx["logu"][members] - log_u[members][:, None])
        alpha_log = max(alpha_log, math.log(len(members) / n) + value)
    alpha = math.exp(min(alpha_log, math.log(ALPHA_CAP)))
    efficiency = math.exp(min(0.0, (float(log_u.sum()) - ctx["nw_upper"]) / n))
    if efficiency >= 1.0 - SNAP_TOLERANCE:
        efficiency = 1.0
    return {
        "max_coalition_improvement": alpha,
        "core_constraint_satisfied": 1.0 if alpha <= 1.0 + CORE_TOLERANCE else 0.0,
        "nash_welfare_efficiency": efficiency,
    }


def deep_metrics(ctx: dict, log_u: np.ndarray) -> dict:
    n = log_u.shape[0]
    frac = log_u - ctx["logu_max"]
    best = float(np.max(math.log(1.0 / n) - frac))
    for coalition, logu_c in ctx["candidates"].items():
        members = list(coalition)
        value = log_game_value(logu_c - log_u[members][:, None])
        best = max(best, math.log(len(members) / n) + value)
    log10_alpha = best / math.log(10.0)
    return {
        "nash_welfare_ideal_fraction": float(np.exp(frac.mean())),
        "worst_agent_ideal_fraction": float(np.exp(frac.min())),
        "log10_coalition_improvement": float(min(max(log10_alpha, 0.0), LOG10_ALPHA_CAP)),
        "blocked": 1.0 if log10_alpha > math.log10(1.0 + DEEP_VIOLATION_TOLERANCE) else 0.0,
    }


# ---------------------------------------------------------------------------
# Evaluator-owned baselines
# ---------------------------------------------------------------------------


def beam_search_max(logp: np.ndarray, log_weights: np.ndarray, width: int):
    """Beam search for the statement maximising sum_i exp(log_weights_i + log U_i(X))."""

    n, depth, branching = logp.shape
    best = logp.max(axis=2)
    suffix = np.zeros((n, depth + 1))
    suffix[:, :depth] = np.cumsum(best[:, ::-1], axis=1)[:, ::-1]
    prefixes = np.zeros((1, 0), dtype=np.int64)
    partial = np.zeros((1, n))
    for t in range(depth):
        cand = partial[:, None, :] + logp[:, t, :].T[None, :, :]
        score = _logsumexp(cand + (log_weights + suffix[:, t + 1])[None, None, :], axis=2)
        flat = score.ravel()
        order = np.argsort(-flat, kind="stable")[: min(width, flat.size)]
        rows, acts = np.divmod(order, branching)
        prefixes = np.concatenate([prefixes[rows], acts[:, None]], axis=1)
        partial = cand[rows, acts]
    values = _logsumexp(partial + log_weights[None, :], axis=1)
    order = np.argsort(-values, kind="stable")
    return prefixes[order], values[order]


def empirical_baselines(ctx: dict) -> dict:
    logu = ctx["logu"]
    n, m = logu.shape
    out = {}
    out["uniform_policy"] = _logsumexp(logu, axis=1) - math.log(m)
    j = int(np.argmax(_logsumexp(logu, axis=0)))
    out["utilitarian_policy"] = logu[:, j].copy()
    favorites = np.argmax(logu, axis=1)
    probs = np.zeros(m)
    np.add.at(probs, favorites, 1.0 / n)
    sup = probs > 0
    out["random_dictatorship_policy"] = _logsumexp(logu[:, sup] + np.log(probs[sup])[None, :], axis=1)
    return out


def deep_baselines(ctx: dict) -> dict:
    logp = ctx["logp"]
    n = logp.shape[0]
    out = {}
    out["uniform_policy"] = (_logsumexp(logp, axis=2) - math.log(logp.shape[2])).sum(axis=1)
    leaves, _ = beam_search_max(logp, np.zeros(n), UTILITARIAN_BEAM_WIDTH)
    out["utilitarian_policy"] = leaf_logutils(logp, leaves[:1])[:, 0]
    fav, counts = np.unique(ctx["favorites"], axis=0, return_counts=True)
    out["random_dictatorship_policy"] = log_expected_utilities(logp, fav, counts / counts.sum())
    return out


# ---------------------------------------------------------------------------
# Solver execution
# ---------------------------------------------------------------------------


def discover_solvers(workspace: Path) -> list[Path]:
    solver_dir = workspace / "scripts" / "solvers"
    if not solver_dir.is_dir():
        return []
    return sorted(p for p in solver_dir.glob("*.sh") if p.is_file())


def run_solver(solver: Path, instance_dir: Path, workspace: Path, timeout: float, scratch: Path):
    out_dir = scratch / f"outputs_{solver.stem}"
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True)
    log_path = scratch / f"{solver.stem}.log"
    start = time.time()
    status = "ok"
    with open(log_path, "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            ["bash", str(solver), str(instance_dir), str(out_dir)],
            cwd=str(workspace),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            rc = proc.wait(timeout=timeout)
            if rc != 0:
                status = f"solver exited with code {rc}"
        except subprocess.TimeoutExpired:
            status = f"solver exceeded the {timeout:.0f}s time limit"
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            proc.wait()
    log_tail = ""
    try:
        log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-1500:]
    except OSError:
        pass
    return out_dir, status, time.time() - start, log_tail


def read_policy(out_dir: Path, instance_id: str):
    path = out_dir / f"{instance_id}.json"
    if not path.is_file():
        raise SolverFailure(f"missing output for instance {instance_id}")
    if path.stat().st_size > MAX_OUTPUT_BYTES:
        raise SolverFailure(f"output for {instance_id} exceeds {MAX_OUTPUT_BYTES} bytes")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SolverFailure(f"unreadable output for {instance_id}: {exc}") from None
    if not isinstance(payload, dict) or "policy" not in payload:
        raise SolverFailure(f"output for {instance_id} has no 'policy'")
    return payload["policy"]


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_empirical(per_instance: list[dict]) -> dict:
    alphas = np.array([r["max_coalition_improvement"] for r in per_instance])
    alpha_max = float(min(alphas.max(), ALPHA_CAP))
    if alpha_max <= 1.0 + CORE_TOLERANCE:
        alpha_max = 1.0
    eff = float(np.mean([r["nash_welfare_efficiency"] for r in per_instance]))
    if eff >= 1.0 - SNAP_TOLERANCE:
        eff = 1.0
    return {
        "max_coalition_improvement": alpha_max,
        "core_constraint_satisfied": 1.0 if alpha_max <= 1.0 + CORE_TOLERANCE else 0.0,
        "nash_welfare_efficiency": eff,
    }


def aggregate_deep(per_instance: list[dict]) -> dict:
    return {
        "nash_welfare_ideal_fraction": float(np.mean([r["nash_welfare_ideal_fraction"] for r in per_instance])),
        "mean_log10_coalition_improvement": float(np.mean([r["log10_coalition_improvement"] for r in per_instance])),
        "coalition_violation_rate": float(np.mean([r["blocked"] for r in per_instance])),
        "worst_agent_ideal_fraction": float(np.mean([r["worst_agent_ideal_fraction"] for r in per_instance])),
    }


def _by_rho(instances: list[dict], per_instance: list[dict], key: str, reducer) -> dict:
    groups: dict[str, list[float]] = {}
    for inst, row in zip(instances, per_instance):
        groups.setdefault(f"{inst['rho']:.4g}", []).append(row[key])
    return {rho: float(reducer(vals)) for rho, vals in groups.items()}


def write_instances(instances: list[dict], directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for inst in instances:
        name = f"{inst['instance_id']}.json"
        (directory / name).write_text(json.dumps(inst), encoding="utf-8")
        entries.append({"instance_id": inst["instance_id"], "experiment": inst["experiment"], "file": name})
    (directory / "manifest.json").write_text(json.dumps({"instances": entries}, indent=1), encoding="utf-8")


def run_evaluation(instances: list[dict], *, split: str, scores_path: Path, workspace: Path, timeout: float) -> dict:
    by_exp = {exp: [inst for inst in instances if inst["experiment"] == exp] for exp in EXPERIMENTS}
    t0 = time.time()
    contexts = {
        EMPIRICAL: [empirical_context(inst) for inst in by_exp[EMPIRICAL]],
        DEEP: [deep_context(inst) for inst in by_exp[DEEP]],
    }
    print(f"[{split}] built {len(instances)} instance contexts in {time.time() - t0:.1f}s", flush=True)

    raw: dict[str, dict[str, list[dict]]] = {exp: {} for exp in EXPERIMENTS}
    failures: dict[str, list[str]] = {}
    runtimes: dict[str, float] = {}

    for name, fn in (("empirical", empirical_baselines), ("deep", deep_baselines)):
        exp = EMPIRICAL if name == "empirical" else DEEP
        metric_fn = empirical_metrics if exp == EMPIRICAL else deep_metrics
        for ctx in contexts[exp]:
            for baseline, log_u in fn(ctx).items():
                raw[exp].setdefault(baseline, []).append(metric_fn(ctx, log_u))
    print(f"[{split}] scored evaluator baselines ({time.time() - t0:.1f}s)", flush=True)

    scratch = Path(tempfile.mkdtemp(prefix=f"core_eval_{split}_"))
    instance_dir = scratch / "instances"
    try:
        solvers = discover_solvers(workspace)
        used: set[str] = set(BASELINES)
        for solver in solvers:
            name = solver.stem
            if not NAME_RE.match(name) or name in used:
                # Every solver on disk is scored; names that are invalid or collide with an
                # evaluator baseline are reported under a distinct key.
                base = "solver_" + re.sub(r"[^A-Za-z0-9_]", "_", name)[:50]
                name, k = base, 1
                while name in used:
                    name, k = f"{base}_{k}", k + 1
            used.add(name)
            reasons: list[str] = []
            shutil.rmtree(instance_dir, ignore_errors=True)
            write_instances(instances, instance_dir)
            out_dir, status, elapsed, log_tail = run_solver(solver, instance_dir, workspace, timeout, scratch)
            runtimes[name] = round(elapsed, 1)
            print(f"[{split}] solver {name}: {status} after {elapsed:.1f}s", flush=True)
            if status != "ok":
                reasons.append(status)
                if log_tail:
                    reasons.append("log tail: " + log_tail)
            per_exp: dict[str, list[dict]] = {exp: [] for exp in EXPERIMENTS}
            if not reasons:
                for exp in EXPERIMENTS:
                    metric_fn = empirical_metrics if exp == EMPIRICAL else deep_metrics
                    for inst, ctx in zip(by_exp[exp], contexts[exp]):
                        try:
                            policy = read_policy(out_dir, inst["instance_id"])
                            leaves, probs = execute_policy(policy, inst["branching"], inst["depth"])
                            log_u = log_expected_utilities(ctx["logp"], leaves, probs)
                            per_exp[exp].append(metric_fn(ctx, log_u))
                        except SolverFailure as exc:
                            reasons.append(str(exc))
                            break
                        except Exception as exc:  # evaluator-side error on a malformed output
                            reasons.append(f"could not score {inst['instance_id']}: {exc!r}")
                            traceback.print_exc()
                            break
                    if reasons:
                        break
            if not reasons:
                bad = [r for r in per_exp[EMPIRICAL] if r["core_constraint_satisfied"] < 1.0]
                if bad:
                    reasons.append(
                        f"ex-ante core violated on {len(bad)} of {len(per_exp[EMPIRICAL])} "
                        f"{EMPIRICAL} instances (max alpha* {max(r['max_coalition_improvement'] for r in bad):.6g})"
                    )
            for exp in EXPERIMENTS:
                if per_exp[exp] and len(per_exp[exp]) == len(by_exp[exp]):
                    raw[exp][name] = per_exp[exp]
            if reasons:
                failures[name] = reasons
            shutil.rmtree(out_dir, ignore_errors=True)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)

    method_names = list(BASELINES) + sorted(n for n in set(raw[EMPIRICAL]) | set(raw[DEEP]) | set(failures) if n not in BASELINES)
    scores = {"experiments": {}, "_meta": {"split": split, "solvers_failed": failures, "solver_runtime_seconds": runtimes}}
    for exp in EXPERIMENTS:
        aggregate = aggregate_empirical if exp == EMPIRICAL else aggregate_deep
        results, raw_summary, per_rho = {}, {}, {}
        for name in method_names:
            rows = raw[exp].get(name)
            measured = aggregate(rows) if rows else None
            if measured is not None:
                raw_summary[name] = measured
                if exp == EMPIRICAL:
                    per_rho[name] = _by_rho(by_exp[exp], rows, "max_coalition_improvement", np.max)
                else:
                    per_rho[name] = {
                        "nash_welfare_ideal_fraction": _by_rho(by_exp[exp], rows, "nash_welfare_ideal_fraction", np.mean),
                        "log10_coalition_improvement": _by_rho(by_exp[exp], rows, "log10_coalition_improvement", np.mean),
                    }
            if name in failures or measured is None:
                results[name] = dict(WORST[exp])
            else:
                results[name] = measured
        scores["experiments"][exp] = {
            "results": results,
            "_details": {
                "split": split,
                "num_instances": len(by_exp[exp]),
                "rhos": sorted({float(inst["rho"]) for inst in by_exp[exp]}),
                "measured_before_failure_masking": raw_summary,
                "per_rho": per_rho,
                "per_instance": {
                    name: [{"rho": inst["rho"], **row} for inst, row in zip(by_exp[exp], rows)]
                    for name, rows in raw[exp].items()
                },
            },
        }
    scores_path = Path(scores_path)
    scores_path.parent.mkdir(parents=True, exist_ok=True)
    scores_path.write_text(json.dumps(scores, indent=1), encoding="utf-8")
    summary = {exp: scores["experiments"][exp]["results"] for exp in EXPERIMENTS}
    print(json.dumps(summary, indent=1))
    if failures:
        print(json.dumps({"failed": {k: v[:1] for k, v in failures.items()}}, indent=1))
    print(f"[{split}] wrote {scores_path} ({time.time() - t0:.1f}s)", flush=True)
    return scores
