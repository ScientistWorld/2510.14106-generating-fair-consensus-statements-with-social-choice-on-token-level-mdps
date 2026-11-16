#!/usr/bin/env python3
"""Build the language-model token trees of the consensus-search experiment (run on a GPU node).

    python -m data.lm_consensus.build_trees --scenarios data/lm_consensus/scenarios_train.json \
        --out data/lm_consensus/trees

For every scenario (an issue and n participants' opinions) the tree has branching B=4 and depth
L=8 (paper Section 3: at each prefix the base model proposes its B most likely next tokens).
The base model is Qwen2.5-1.5B-Instruct prompted with the issue and all opinions (paper
Figure 6); participant i's policy is the same model prompted with the issue and opinion i only
(Figure 7). For every one of the (B^L - 1)/(B - 1) = 21,845 internal prefixes the file stores

    ids_t     (B^t, B)     int32    token ids of the B candidate next tokens
    base_t    (B^t, B)     float16  log pi_base(a | s) of those tokens (full-vocabulary softmax)
    agents_t  (n, B^t, B)  float16  log pi_i(a | s) of those tokens (full-vocabulary softmax)

where row k of level t is the prefix whose candidate indices spell k in base B. Files are
written one scenario at a time and existing files are skipped, so the job can resume.

The model runs with bfloat16 weights and a float32 output layer (log-probabilities are computed
in float32 from the last hidden state). GPUs may be shared, so the builder waits for enough free
memory, sizes its batches from the memory actually free, and retries smaller batches on OOM.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np

from data.lm_consensus.prompts import (BANNED_SUBSTRINGS, BRANCHING, DEPTH, MODEL_NAME, agent_messages,
                                       reference_messages)

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = ROOT / "data" / "downloads" / "Qwen2.5-1.5B-Instruct"


def _legacy(cache):
    if isinstance(cache, (tuple, list)):
        return tuple((k, v) for k, v in cache)
    if hasattr(cache, "to_legacy_cache"):
        return cache.to_legacy_cache()
    return tuple((layer.keys, layer.values) for layer in cache.layers)


class TreeBuilder:
    def __init__(self, model_path: Path, device: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        if self.device == "cuda":
            self._wait_for_memory(6.0e9)
        self.tok = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        self.head = self.model.get_output_embeddings().weight.detach().float()  # float32 output layer
        cfg = self.model.config
        self.vocab = int(self.head.shape[0])
        head_dim = cfg.hidden_size // cfg.num_attention_heads
        elt = 2 if dtype == torch.bfloat16 else 4
        # expanded prompt cache + per-layer repeated keys/values (GQA), per sequence and token
        self.bytes_per_token = (cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * head_dim * elt
                                + 2 * cfg.num_attention_heads * head_dim * elt * 2)
        self.bytes_per_row = 4 * self.vocab * 4  # float32 logits, log-softmax and masked copies
        self.chunk_cap = 1024
        self.banned = self._banned_mask()
        print(f"[build] device={self.device} dtype={dtype} vocab={self.vocab} banned={int(self.banned.sum())} "
              f"free={self._free() / 1e9:.1f}GB", flush=True)

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

    def _logprobs_from_hidden(self, h):
        return self.torch.log_softmax(h.float() @ self.head.T, dim=-1)

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

    def _prompt_ids(self, messages) -> list[int]:
        ids = self.tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        return [int(x) for x in ids]

    def _prompt_state(self, ids: list[int]):
        torch = self.torch
        with torch.no_grad():
            out = self.model.model(input_ids=torch.tensor([ids], device=self.device), use_cache=True)
            root = self._logprobs_from_hidden(out.last_hidden_state[:, -1])
        return root, _legacy(out.past_key_values), len(ids)

    def _forward_chunk(self, legacy, batch):
        torch = self.torch
        from transformers import DynamicCache

        c = batch.shape[0]
        cache = DynamicCache()
        for layer, (k, v) in enumerate(legacy):
            cache.update(k.expand(c, -1, -1, -1).contiguous(), v.expand(c, -1, -1, -1).contiguous(), layer)
        with torch.no_grad():
            out = self.model.model(input_ids=batch, past_key_values=cache, use_cache=True)
            lp = self._logprobs_from_hidden(out.last_hidden_state[:, -1])
        del cache, out
        return lp

    def _level_logprobs(self, state, prefixes, reduce):
        """Apply ``reduce(row_slice, logprobs)`` to the full-vocabulary next-token log-probabilities
        at every prefix (rows of ``prefixes``, all of the same length >= 1), chunk by chunk."""

        torch = self.torch
        root_lp, legacy, plen = state
        if prefixes.shape[1] == 0:
            reduce(slice(0, 1), root_lp)
            return
        rows = prefixes.shape[0]
        per_seq = self.bytes_per_token * (plen + prefixes.shape[1]) + self.bytes_per_row
        start, failures = 0, 0
        while start < rows:
            chunk = int(max(1, min(self.chunk_cap, 0.6 * self._free() // per_seq)))
            stop = min(rows, start + chunk)
            try:
                lp = self._forward_chunk(legacy, prefixes[start:stop])
                reduce(slice(start, stop), lp)  # idempotent: a retried slice is simply rewritten
                del lp
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                lp = None
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

    def build(self, issue: str, opinions: list[str]) -> dict:
        torch = self.torch
        b, depth, n = BRANCHING, DEPTH, len(opinions)
        prompts = [reference_messages(issue, opinions)] + [agent_messages(issue, op) for op in opinions]
        states = [self._prompt_state(self._prompt_ids(m)) for m in prompts]
        prefixes = torch.zeros((1, 0), dtype=torch.long, device=self.device)
        out = {}
        for t in range(depth):
            rows = b ** t
            ids = torch.zeros((rows, b), dtype=torch.long, device=self.device)
            base = torch.zeros((rows, b), dtype=torch.float32, device=self.device)
            agents = torch.zeros((n, rows, b), dtype=torch.float32, device=self.device)

            def take_top(sl, lp):
                masked = lp.masked_fill(self.banned[None, :], -math.inf)
                top = torch.topk(masked, b, dim=-1)
                ids[sl] = top.indices
                base[sl] = lp.gather(1, top.indices)

            self._level_logprobs(states[0], prefixes, take_top)
            for i in range(n):
                def take_agent(sl, lp, i=i):
                    agents[i, sl] = lp.gather(1, ids[sl])
                self._level_logprobs(states[1 + i], prefixes, take_agent)
            out[f"ids_{t}"] = ids.cpu().numpy().astype(np.int32)
            out[f"base_{t}"] = base.cpu().numpy().astype(np.float16)
            out[f"agents_{t}"] = agents.cpu().numpy().astype(np.float16)
            if t + 1 < depth:
                prefixes = torch.cat([prefixes.repeat_interleave(b, dim=0), ids.reshape(-1, 1)], dim=1)
        return out

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
    bu = np.zeros(1)
    for t in range(depth):
        lu = (lu[:, :, None] + arrays[f"agents_{t}"].astype(np.float64)).reshape(n, -1)
        bu = (bu[:, None] + arrays[f"base_{t}"].astype(np.float64)).reshape(-1)
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
    builder = None
    log = []
    for sc in scenarios:
        path = out_dir / f"{sc['id']}.npz"
        if path.exists():
            print(f"[build] {sc['id']}: exists, skipped", flush=True)
            continue
        if builder is None:
            builder = TreeBuilder(Path(args.model))
        t0 = time.time()
        arrays = builder.build(sc["issue"], sc["opinions"])
        meta = {"id": sc["id"], "model": MODEL_NAME, "branching": BRANCHING, "depth": DEPTH,
                "num_agents": len(sc["opinions"]), "stored_dtype": "float16",
                "compute": "bfloat16 weights, float32 output layer" if builder.device == "cuda" else "float32"}
        tmp = out_dir / f".{sc['id']}.tmp.npz"
        np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrays)
        tmp.rename(path)
        ex = _examples(builder, arrays, len(sc["opinions"]))
        ex.update({"id": sc["id"], "seconds": round(time.time() - t0, 1), "bytes": path.stat().st_size})
        log.append(ex)
        print(f"[build] {json.dumps(ex)}", flush=True)
    if log:
        (out_dir / "build_log.json").write_text(json.dumps(log, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
