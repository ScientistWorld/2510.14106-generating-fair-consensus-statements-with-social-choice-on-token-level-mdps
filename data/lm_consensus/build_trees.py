#!/usr/bin/env python3
"""Build the language-model token trees of the consensus-search experiment (run on a GPU node).

    python -m data.lm_consensus.build_trees --scenarios data/lm_consensus/scenarios_train.json \
        --out data/lm_consensus/trees

For every scenario (an issue and n participants' opinions) the tree has branching B=2 and depth
L=16 (paper Section 3: at each prefix the base model proposes its B most likely next tokens;
the paper's finite-lookahead experiment uses B=2). The base model is Qwen2.5-1.5B-Instruct
prompted with the issue and all opinions (paper Figure 6); participant i's policy is the same
model prompted with the issue and opinion i only (Figure 7). For each of the
(B^L - 1)/(B - 1) = 65,535 internal prefixes the file stores

    ids_t     (B^t, B)     int32    token ids of the B candidate next tokens
    base_t    (B^t, B)     float16  log pi_base(a | s) of those tokens (full-vocabulary softmax)
    agents_t  (n, B^t, B)  float16  log pi_i(a | s) of those tokens (full-vocabulary softmax)

where row k of level t is the prefix whose candidate indices spell k in base B.

The base prompt is traversed first (it fixes the candidates), then each participant's prompt is
evaluated on the same tree. Prefixes of one level are run in batches on top of the prompt's
key/value cache plus the cached keys/values of their ancestor at depth 4, 8 or 12, so each
prefix only feeds its last 1-4 tokens. Before a tree is saved, sampled prefixes are recomputed
with a plain forward pass of prompt + prefix and must agree.

The model runs with bfloat16 weights and a float32 output layer. GPUs may be shared, so the
builder waits for free memory, sizes its batches from the memory actually free, and retries
smaller batches on OOM. Files are written one scenario at a time; existing files are skipped.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np

from data.lm_consensus.prompts import (BANNED_SUBSTRINGS, BRANCHING, DEPTH, MODEL_NAME, agent_messages,
                                       reference_messages)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "downloads" / "Qwen2.5-1.5B-Instruct"
ANCHOR_EVERY = 4
VERIFY_TOLERANCE = 0.5  # nats; a wrong position or cache would be off by far more


def _cache_layers(cache) -> list:
    if isinstance(cache, (tuple, list)):
        return [(k, v) for k, v in cache]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(layer.keys, layer.values) for layer in cache.layers]


def prefix_tokens(ids_levels: list, t: int, row: int, b: int) -> list[int]:
    digits, x = [], row
    for _ in range(t):
        digits.append(x % b)
        x //= b
    toks, idx = [], 0
    for j, d in enumerate(reversed(digits)):
        toks.append(int(ids_levels[j][idx, d]))
        idx = idx * b + d
    return toks


class TreeBuilder:
    def __init__(self, model_path: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            self._wait_for_memory(6.0e9)
        self.tok = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=self.dtype, attn_implementation="sdpa").to(self.device).eval()
        self.head = self.model.get_output_embeddings().weight.detach().float()  # float32 output layer
        cfg = self.model.config
        self.vocab = int(self.head.shape[0])
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        elt = 2 if self.dtype == torch.bfloat16 else 4
        # batched past keys/values + per-layer repeated keys/values (GQA), per sequence and token
        self.bytes_per_token = (cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * elt
                                + 2 * cfg.num_attention_heads * head_dim * elt * 2)
        self.bytes_per_row = 4 * self.vocab * 4  # float32 logits, log-softmax and masked copies
        self.chunk_cap = 1024
        self.banned = self._banned_mask()
        print(f"[build] device={self.device} dtype={self.dtype} vocab={self.vocab} "
              f"banned={int(self.banned.sum())} free={self._free() / 1e9:.1f}GB", flush=True)

    # ------------------------------------------------------------------ utilities

    def _free(self) -> float:
        if self.device != "cuda":
            return 4.0e9
        free, _ = self.torch.cuda.mem_get_info()
        return float(free)

    def _wait_for_memory(self, need: float, max_wait: float = 3600.0) -> None:
        start = time.time()
        while self._free() < need and time.time() - start < max_wait:
            print(f"[build] waiting for GPU memory: {self._free() / 1e9:.2f}GB free, need {need / 1e9:.1f}GB",
                  flush=True)
            time.sleep(60)

    def _banned_mask(self):
        torch = self.torch
        mask = torch.zeros(self.vocab, dtype=torch.bool)
        n_tok = len(self.tok)
        mask[n_tok:] = True
        for tid in set(self.tok.all_special_ids) | set(self.tok.get_added_vocab().values()):
            if 0 <= tid < self.vocab:
                mask[tid] = True
        for tid in range(min(n_tok, self.vocab)):
            text = self.tok.decode([tid])
            if text == "" or any(b in text for b in BANNED_SUBSTRINGS):
                mask[tid] = True
        return mask.to(self.device)

    def _logprobs_from_hidden(self, h):
        return self.torch.log_softmax(h.float() @ self.head.T, dim=-1)

    def _prompt_ids(self, messages) -> list[int]:
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return [int(x) for x in ids]

    def prompt_state(self, messages) -> dict:
        torch = self.torch
        ids = self._prompt_ids(messages)
        with torch.no_grad():
            out = self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
            root = self._logprobs_from_hidden(out.last_hidden_state[:, -1])
        return {"ids": ids, "root": root, "layers": _cache_layers(out.past_key_values), "plen": len(ids)}

    def _forward(self, state: dict, anchor_layers, anc_idx, tokens):
        """Next-token log-probabilities after prompt + ancestor (cached) + ``tokens`` for a batch."""

        torch = self.torch
        from transformers import DynamicCache

        c = tokens.shape[0]
        cache = DynamicCache()
        for layer, (pk, pv) in enumerate(state["layers"]):
            if anchor_layers is None:
                k = pk.expand(c, -1, -1, -1).contiguous()
                v = pv.expand(c, -1, -1, -1).contiguous()
            else:
                ak, av = anchor_layers[layer]
                k = torch.cat([pk.expand(c, -1, -1, -1), ak[anc_idx]], dim=2)
                v = torch.cat([pv.expand(c, -1, -1, -1), av[anc_idx]], dim=2)
            cache.update(k, v, layer)
        with torch.no_grad():
            out = self.model.model(input_ids=tokens, past_key_values=cache, use_cache=True)
            lp = self._logprobs_from_hidden(out.last_hidden_state[:, -1])
        return lp, cache

    # ------------------------------------------------------------------ traversal

    def traverse(self, state: dict, ids_levels=None):
        """Visit every prefix of the tree level by level. Without ``ids_levels`` (the base prompt)
        the B candidates of each prefix are chosen (top-B allowed tokens); otherwise this prompt's
        log-probabilities of the given candidates are gathered. Returns (ids_levels, lp_levels)."""

        torch = self.torch
        b, depth, plen = BRANCHING, DEPTH, state["plen"]
        prefixes = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        anchor = None  # (depth a, [(K, V) per layer] with rows = prefixes of length a)
        ids_out, lp_out = [], []
        for t in range(depth):
            rows = b ** t
            if ids_levels is None:
                ids_t = torch.zeros((rows, b), dtype=torch.long, device=self.device)
            else:
                ids_t = torch.as_tensor(np.asarray(ids_levels[t]), dtype=torch.long, device=self.device)
            lp_t = torch.zeros((rows, b), dtype=torch.float32, device=self.device)

            def take(sl, lp):
                if ids_levels is None:
                    ids_t[sl] = torch.topk(lp.masked_fill(self.banned[None, :], -math.inf), b, dim=-1).indices
                lp_t[sl] = lp.gather(1, ids_t[sl])

            if t == 0:
                take(slice(0, 1), state["root"])
            else:
                a = ANCHOR_EVERY * ((t - 1) // ANCHOR_EVERY)
                if a > 0:
                    assert anchor is not None and anchor[0] == a, (t, a, None if anchor is None else anchor[0])
                make_anchor = t % ANCHOR_EVERY == 0 and t < depth - 1
                new_anchor = None
                per_seq = self.bytes_per_token * (plen + t) + self.bytes_per_row
                start, failures = 0, 0
                while start < rows:
                    chunk = int(max(1, min(self.chunk_cap, 0.6 * self._free() // per_seq)))
                    stop = min(rows, start + chunk)
                    try:
                        anc_idx = None
                        if a > 0:
                            anc_idx = torch.arange(start, stop, device=self.device) // (b ** (t - a))
                        lp, cache = self._forward(state, anchor[1] if a > 0 else None, anc_idx,
                                                  prefixes[start:stop, a:t])
                        take(slice(start, stop), lp)
                        if make_anchor:
                            layers = _cache_layers(cache)
                            if new_anchor is None:
                                new_anchor = [(torch.empty((rows, k.shape[1], t, k.shape[3]), dtype=k.dtype,
                                                           device=self.device),
                                               torch.empty((rows, v.shape[1], t, v.shape[3]), dtype=v.dtype,
                                                           device=self.device)) for k, v in layers]
                            for layer, (k, v) in enumerate(layers):
                                new_anchor[layer][0][start:stop] = k[:, :, plen:plen + t]
                                new_anchor[layer][1][start:stop] = v[:, :, plen:plen + t]
                        del lp, cache
                    except (torch.OutOfMemoryError, RuntimeError) as exc:
                        if "out of memory" not in str(exc).lower():
                            raise
                        torch.cuda.empty_cache()
                        failures += 1
                        self.chunk_cap = max(1, chunk // 2)
                        if chunk == 1:
                            print(f"[build] OOM at batch 1 ({self._free() / 1e9:.2f}GB free); waiting", flush=True)
                            time.sleep(30)
                        if failures > 200:
                            raise
                        continue
                    self.chunk_cap = min(1024, self.chunk_cap + max(1, self.chunk_cap // 8))
                    start = stop
                if make_anchor:
                    anchor = (t, new_anchor)
            ids_out.append(ids_t)
            lp_out.append(lp_t)
            if t + 1 < depth:
                prefixes = torch.cat([prefixes.repeat_interleave(b, dim=0), ids_t.reshape(-1, 1)], dim=1)
        del anchor
        return [x.cpu().numpy().astype(np.int32) for x in ids_out], [x.cpu().numpy() for x in lp_out]

    def verify(self, state: dict, ids_levels: list, lp_levels: list, rng, levels=(3, 5, 9, 13, 15)) -> float:
        """Max |stored - direct| over sampled prefixes, recomputing prompt + prefix in one plain pass."""

        torch = self.torch
        worst = 0.0
        for t in levels:
            if t >= DEPTH:
                continue
            row = int(rng.integers(0, BRANCHING ** t))
            toks = prefix_tokens(ids_levels, t, row, BRANCHING)
            with torch.no_grad():
                out = self.model.model(input_ids=torch.tensor([state["ids"] + toks], device=self.device))
                lp = self._logprobs_from_hidden(out.last_hidden_state[:, -1])[0]
            direct = lp[torch.as_tensor(ids_levels[t][row], dtype=torch.long, device=self.device)].cpu().numpy()
            worst = max(worst, float(np.abs(direct - lp_levels[t][row]).max()))
        return worst

    def build(self, issue: str, opinions: list[str], seed: int = 0):
        rng = np.random.default_rng(seed)
        out, checks = {}, []
        base_state = self.prompt_state(reference_messages(issue, opinions))
        ids_levels, base_lp = self.traverse(base_state)
        checks.append(self.verify(base_state, ids_levels, base_lp, rng))
        del base_state
        agents = []
        for i, op in enumerate(opinions):
            state = self.prompt_state(agent_messages(issue, op))
            _, lp = self.traverse(state, ids_levels)
            if i == 0:
                checks.append(self.verify(state, ids_levels, lp, rng))
            agents.append(lp)
            del state
        for t in range(DEPTH):
            out[f"ids_{t}"] = ids_levels[t]
            out[f"base_{t}"] = base_lp[t].astype(np.float16)
            out[f"agents_{t}"] = np.stack([lp[t] for lp in agents]).astype(np.float16)
        return out, max(checks)

    def decode_path(self, arrays: dict, path) -> str:
        idx, toks = 0, []
        for t, a in enumerate(path):
            toks.append(int(arrays[f"ids_{t}"][idx, a]))
            idx = idx * BRANCHING + a
        return self.tok.decode(toks)


def _examples(builder: TreeBuilder, arrays: dict, n: int) -> dict:
    """Statements for the log: the base model's greedy path and the egalitarian optimum."""

    b, depth = BRANCHING, DEPTH
    lu = np.zeros((n, 1))
    for t in range(depth):
        lu = (lu[:, :, None] + arrays[f"agents_{t}"].astype(np.float64)).reshape(n, -1)
    ew = lu.min(axis=0)

    def digits(k):
        out = []
        for _ in range(depth):
            out.append(int(k % b))
            k //= b
        return out[::-1]

    greedy, idx = [], 0
    for t in range(depth):
        a = int(np.argmax(arrays[f"base_{t}"][idx]))
        greedy.append(a)
        idx = idx * b + a
    best = int(np.argmax(ew))
    return {
        "base_greedy": builder.decode_path(arrays, greedy),
        "base_greedy_eppl": float(np.exp(-ew[idx] / depth)),
        "egalitarian_optimum": builder.decode_path(arrays, digits(best)),
        "optimum_eppl": float(np.exp(-ew[best] / depth)),
        "worst_eppl": float(np.exp(-ew.min() / depth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenarios", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    args = parser.parse_args()
    scenarios = json.loads(Path(args.scenarios).read_text(encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "build_log.json"
    log = json.loads(log_path.read_text()) if log_path.exists() else []
    builder = None
    for k, sc in enumerate(scenarios):
        path = out_dir / f"{sc['id']}.npz"
        if path.exists():
            print(f"[build] {sc['id']}: exists, skipped", flush=True)
            continue
        if builder is None:
            builder = TreeBuilder(Path(args.model))
        t0 = time.time()
        arrays, max_diff = builder.build(sc["issue"], sc["opinions"], seed=k)
        if not max_diff <= VERIFY_TOLERANCE:
            raise RuntimeError(f"{sc['id']}: batched log-probabilities differ from a direct pass by {max_diff}")
        meta = {"id": sc["id"], "model": MODEL_NAME, "branching": BRANCHING, "depth": DEPTH,
                "num_agents": len(sc["opinions"]), "stored_dtype": "float16",
                "compute": "bfloat16 weights, float32 output layer" if builder.device == "cuda" else "float32",
                "verify_max_abs_diff": max_diff}
        tmp = out_dir / f".{sc['id']}.tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.rename(path)
        ex = _examples(builder, arrays, len(sc["opinions"]))
        ex.update({"id": sc["id"], "seconds": round(time.time() - t0, 1), "bytes": path.stat().st_size,
                   "verify_max_abs_diff": round(max_diff, 4)})
        log = [e for e in log if e.get("id") != sc["id"]] + [ex]
        log_path.write_text(json.dumps(log, indent=1), encoding="utf-8")
        print(f"[build] {json.dumps(ex)}", flush=True)


if __name__ == "__main__":
    main()
