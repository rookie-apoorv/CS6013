import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="CS6013 nb3 - SFT (self-distillation, LoRA)")

with app.setup:
    import importlib
    import json
    import os
    import shutil
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo

    NB_VERSION = "nb3-v1.0.2 (2026-11-23)"
    WORK = Path("/root/work/nb3") if Path("/root").exists() else Path.home() / "work" / "nb3"
    WORK.mkdir(parents=True, exist_ok=True)
    PY = sys.executable
    for _d in ("data", "runs", "exports", "hf_pull"):
        (WORK / _d).mkdir(exist_ok=True)

    # Shared state: every heavy step is behind a button; cells talk through this
    # dict so a stray re-run never restarts training.
    STATE = {"L": None, "base_dir": None, "base_rev": None, "data": None, "data_fp": None,
             "pm": None, "smoke": None, "im_end_id": None}

    LIB_SRC = r'''# nb3lib.py -- helper library for the CS6013 SFT notebook (notebook 3).
# Self-distillation of Qwen3.5-4B on its own shortest correct traces:
#   1. training set   (nb1 modules A/B + nb2 module T traces -> one example per problem)
#   2. model + LoRA   (pure-PyTorch Gated DeltaNet by default, chunked LM-head loss)
#   3. smoke test     (step time / peak memory per length; optional fla kernel check)
#   4. training loop  (token-mean loss, resumable checkpoints)
#   5. merge + export (fp32 merge, written into a copy of the base checkpoint layout)
# The notebook writes this file to disk and imports it.
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import re
import shutil
import time
from pathlib import Path

LIB_VERSION = "nb3lib-2026-11-23b"

# Problems kept out of training: the only clean before/after test set.
HELD_OUT_SOURCES = ("aime_2025", "aime_2026", "hmmt_feb_2025")
# nb1 modules used for training (A: AIME 2022-26, B: HMMT 25/26 + BRUMO/CMIMC/SMT 25).
NB1_MODULES = ("A", "B")
NB2_MODULES = ("T",)

# LoRA on every large linear of the decoder stack. Not: lm_head (tied to the
# embedding -- merging would untie it), in_proj_a / in_proj_b (one output per
# head, negligible), the vision tower, MTP.
LORA_TARGET_RE = (r"^model\.layers\.\d+\.(self_attn\.(q_proj|k_proj|v_proj|o_proj)"
                  r"|linear_attn\.(in_proj_qkv|in_proj_z|out_proj)"
                  r"|mlp\.(gate_proj|up_proj|down_proj))$")

DROP_REASONS = ("incorrect", "truncated", "not_stopped", "unparsed", "think_open",
                "answer_in_thinking", "looping", "cjk")


# ===========================================================================
# 1. Training set
# ===========================================================================
def sample_drop_reason(r):
    """First reason a derived sample is NOT usable for training, or None.
    Usable = correct under the grader's parser, finished normally (EOS) before
    the 32K cap, thinking closed, final answer after </think>, no loop, no CJK."""
    if not bool(r["correct_grader"]):
        return "incorrect"
    if bool(r["truncated"]):
        return "truncated"
    if r.get("finish_reason") != "stop":
        return "not_stopped"
    if not bool(r.get("parsed", True)):
        return "unparsed"
    if not bool(r["think_closed"]):
        return "think_open"
    if bool(r["pred_from_thinking"]):
        return "answer_in_thinking"
    if bool(r["looping"]):
        return "looping"
    if int(r.get("n_cjk_chars") or 0) > 0:
        return "cjk"
    return None


def select_shortest(der, exclude_sources=HELD_OUT_SOURCES):
    """One example per problem: its shortest usable sample (ties -> lowest
    sample_idx). Returns (chosen, report): chosen has uid, module, source,
    sample_idx, n_completion_tokens; report has one row per source."""
    import pandas as pd
    d = der[~der.source.isin(list(exclude_sources))].copy()
    if "derive_error" in d.columns:
        d = d[d.derive_error.isna()]
    d["drop"] = [sample_drop_reason(r) for r in d.to_dict("records")]
    ok = d[d["drop"].isna()].sort_values(["uid", "n_completion_tokens", "sample_idx"])
    chosen = ok.groupby("uid", as_index=False).first()[
        ["uid", "module", "source", "sample_idx", "n_completion_tokens"]]
    rows = []
    for src, g in d.groupby("source"):
        c = chosen[chosen.source == src]
        row = dict(module=g.module.iloc[0], source=src, problems=g.uid.nunique(),
                   samples=len(g), usable_samples=int(g["drop"].isna().sum()),
                   problems_kept=len(c),
                   kept_median_tokens=int(c.n_completion_tokens.median()) if len(c) else None,
                   kept_max_tokens=int(c.n_completion_tokens.max()) if len(c) else None)
        for reason in DROP_REASONS:
            row[f"drop_{reason}"] = int((g["drop"] == reason).sum())
        rows.append(row)
    rep = pd.DataFrame(rows).sort_values(["module", "source"]).reset_index(drop=True)
    excluded = sorted(set(der.source) & set(exclude_sources))
    return chosen.reset_index(drop=True), rep, excluded


def attach_tokens(chosen, raw, im_end_id):
    """Join the chosen samples with their raw token ids. Completions that ended
    on EOS but lack the <|im_end|> id get it appended (the model must learn to
    stop). Returns a DataFrame ready for training."""
    import pandas as pd
    need = raw[["uid", "sample_idx", "prompt_token_ids", "completion_token_ids",
                "finish_reason"]]
    m = chosen.merge(need, on=["uid", "sample_idx"], how="left", validate="one_to_one")
    missing = m.prompt_token_ids.isna() | m.completion_token_ids.isna()
    if missing.any():
        raise ValueError(f"{int(missing.sum())} chosen samples have no token ids in raw/: "
                         f"{list(m[missing].uid[:5])}")
    out = []
    n_appended = 0
    for r in m.itertuples(index=False):
        p = [int(x) for x in r.prompt_token_ids]
        c = [int(x) for x in r.completion_token_ids]
        if not c or c[-1] != im_end_id:
            c = c + [int(im_end_id)]
            n_appended += 1
        out.append(dict(uid=r.uid, module=r.module, source=r.source, sample_idx=int(r.sample_idx),
                        prompt_ids=p, completion_ids=c, n_prompt=len(p), n_completion=len(c),
                        n_total=len(p) + len(c)))
    return pd.DataFrame(out), n_appended


def train_set_fingerprint(df):
    h = hashlib.sha256()
    for r in df.sort_values("uid").itertuples(index=False):
        h.update(f"{r.uid}|{r.sample_idx}|{r.n_prompt}|{r.n_completion}|".encode())
        h.update(bytes(str(r.completion_ids[-8:]), "utf-8"))
    return h.hexdigest()[:16]


def length_table(df, col="n_total",
                 edges=(0, 2000, 4000, 8000, 12000, 16000, 20000, 24000, 28000, 33000)):
    import pandas as pd
    b = pd.cut(df[col], list(edges), right=False)
    t = df.groupby(b, observed=False).agg(examples=("uid", "size"),
                                          tokens=(col, "sum")).reset_index()
    t["tokens_share"] = (t.tokens / max(t.tokens.sum(), 1)).round(3)
    t[col] = t[col].astype(str)
    return t


# ===========================================================================
# 2. Model, kernels, LoRA, loss
# ===========================================================================
def gdn_module(model):
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            return __import__(type(m).__module__, fromlist=["x"])
    return None


def force_torch_kernels(model):
    """Every Gated DeltaNet module on transformers' pure-PyTorch path (conv,
    delta rule, gated norm). Same as compress.py's preflight."""
    mod = gdn_module(model)
    counts = {"gdn_modules": 0, "norm_swapped": 0}
    if mod is None:
        return counts
    gated_norm_cls = next((getattr(mod, n) for n in dir(mod) if n.endswith("RMSNormGated")
                           and not n.startswith("Fused")), None)
    for m in model.modules():
        if not type(m).__name__.endswith("GatedDeltaNet"):
            continue
        counts["gdn_modules"] += 1
        m.causal_conv1d_fn = None
        m.chunk_gated_delta_rule = mod.torch_chunk_gated_delta_rule
        if hasattr(mod, "torch_recurrent_gated_delta_rule"):
            m.recurrent_gated_delta_rule = mod.torch_recurrent_gated_delta_rule
        if hasattr(mod, "torch_causal_conv1d_update"):
            m.causal_conv1d_update = mod.torch_causal_conv1d_update
        if gated_norm_cls is not None and not isinstance(m.norm, gated_norm_cls):
            import torch
            old = m.norm
            new = gated_norm_cls(old.weight.shape[0], eps=getattr(old, "eps", 1e-6))
            with torch.no_grad():
                new.weight.copy_(old.weight)
            m.norm = new.to(device=old.weight.device, dtype=old.weight.dtype)
            counts["norm_swapped"] += 1
    return counts


def fla_delta_rule():
    """fla's chunked delta-rule kernel, or None (never required)."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        return chunk_gated_delta_rule
    except Exception:
        return None


def set_delta_rule(model, use_fla):
    """Switch ONLY the chunked delta rule (conv and norm stay PyTorch)."""
    mod = gdn_module(model)
    fn = fla_delta_rule() if use_fla else mod.torch_chunk_gated_delta_rule
    if fn is None:
        raise RuntimeError("flash-linear-attention is not importable")
    n = 0
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            m.chunk_gated_delta_rule = fn
            n += 1
    return n


def load_base(base_dir, device):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    counts = force_torch_kernels(model)
    model.to(device)
    for p in model.parameters():
        p.requires_grad_(False)
    return model, counts


def add_lora(model, r=64, alpha=64, dropout=0.0, grad_ckpt=True):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=int(r), lora_alpha=int(alpha), lora_dropout=float(dropout), bias="none",
                     target_modules=LORA_TARGET_RE, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    base = pm.get_base_model()
    if grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base.config.use_cache = False
    n_train = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    n_mod = sum(1 for n, m in pm.named_modules() if hasattr(m, "lora_A") and "default" in
                getattr(m, "lora_A", {}))
    return pm, dict(trainable_params=n_train, lora_modules=n_mod)


def lora_modules(pm):
    """{base-model module name (model.layers...): lora layer}"""
    out = {}
    for n, m in pm.named_modules():
        if hasattr(m, "lora_A") and hasattr(m, "base_layer") and "default" in m.lora_A:
            name = n
            for pre in ("base_model.model.",):
                if name.startswith(pre):
                    name = name[len(pre):]
            out[name] = m
    return out


def _ce_chunk(h, t, w):
    import torch.nn.functional as F
    return F.cross_entropy((h @ w.t()).float(), t, reduction="sum")


def seq_loss_sum(pm, ids, plen, chunk=2048):
    """Sum of token NLL over the completion part of one sequence. The LM head +
    cross-entropy run in checkpointed chunks, so the (len x 248K) logits are
    never materialised."""
    import torch
    from torch.utils.checkpoint import checkpoint
    base = pm.get_base_model()
    out = base.model(input_ids=ids.unsqueeze(0), use_cache=False)
    h = out.last_hidden_state[0, plen - 1:-1]
    tgt = ids[plen:]
    w = base.lm_head.weight
    total = h.new_zeros((), dtype=torch.float32)
    for s in range(0, tgt.shape[0], chunk):
        total = total + checkpoint(_ce_chunk, h[s:s + chunk], tgt[s:s + chunk], w,
                                   use_reentrant=False)
    return total, int(tgt.shape[0])


# ===========================================================================
# 3. Smoke test helpers
# ===========================================================================
def _sync():
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def time_train_step(pm, seq_len, device, plen=256, vocab_cap=150000, chunk=2048, seed=0):
    """One forward+backward at `seq_len` tokens (random ids; speed and memory
    don't depend on content). Returns seconds and peak GiB."""
    import torch
    g = torch.Generator().manual_seed(seed)
    vocab = pm.get_base_model().get_input_embeddings().weight.shape[0]
    ids = torch.randint(0, min(vocab, vocab_cap), (seq_len,), generator=g).to(device)
    pm.train()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    _sync()
    t0 = time.time()
    loss, n = seq_loss_sum(pm, ids, plen, chunk)
    (loss / n).backward()
    _sync()
    sec = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else None
    pm.zero_grad(set_to_none=True)
    return dict(seq_len=seq_len, sec=round(sec, 2), peak_gib=None if peak is None else round(peak, 2),
                tok_per_s=round(seq_len / sec, 1))


def fla_check(pm, device, seq_len=4096, seed=0):
    """Run ONE real Gated DeltaNet layer with the PyTorch and the fla delta rule
    on the same input; compare output and input-gradient, and time both.
    Returns a dict with ok / errors / timings, or reason when fla is absent."""
    import torch
    fn = fla_delta_rule()
    if fn is None:
        return dict(ok=False, reason="flash-linear-attention not importable")
    base = pm.get_base_model()
    mod = gdn_module(base)
    layer = next(m for m in base.modules() if type(m).__name__.endswith("GatedDeltaNet"))
    hid = base.config.hidden_size if hasattr(base.config, "hidden_size") else \
        base.config.text_config.hidden_size
    g = torch.Generator().manual_seed(seed)
    x0 = (torch.randn(1, seq_len, hid, generator=g) * 0.5).to(device=device, dtype=torch.bfloat16)
    res = {}
    old = layer.chunk_gated_delta_rule
    try:
        for tag, f in (("torch", mod.torch_chunk_gated_delta_rule), ("fla", fn)):
            layer.chunk_gated_delta_rule = f
            x = x0.clone().requires_grad_(True)
            _sync()
            t0 = time.time()
            y = layer(x)
            y = y[0] if isinstance(y, tuple) else y
            (y.float().pow(2).mean()).backward()
            _sync()
            res[tag] = dict(y=y.detach().float(), gx=x.grad.detach().float(), sec=time.time() - t0)
    except Exception as exc:
        return dict(ok=False, reason=f"{type(exc).__name__}: {str(exc)[:300]}")
    finally:
        layer.chunk_gated_delta_rule = old

    def rel(a, b):
        return float((a - b).norm() / b.norm().clamp_min(1e-12))
    e_y = rel(res["fla"]["y"], res["torch"]["y"])
    e_g = rel(res["fla"]["gx"], res["torch"]["gx"])
    ok = bool(e_y < 2e-2 and e_g < 5e-2 and math.isfinite(e_y) and math.isfinite(e_g))
    return dict(ok=ok, rel_err_out=round(e_y, 5), rel_err_grad=round(e_g, 5),
                torch_sec=round(res["torch"]["sec"], 3), fla_sec=round(res["fla"]["sec"], 3),
                speedup=round(res["torch"]["sec"] / max(res["fla"]["sec"], 1e-9), 2),
                seq_len=seq_len)


def estimate_epoch_hours(lengths, timings):
    """Piecewise-linear seconds(len) through the measured (len, sec) points
    (extrapolated linearly from the last two), summed over the data."""
    pts = sorted((t["seq_len"], t["sec"]) for t in timings)
    if len(pts) < 2:
        return None

    def sec(n):
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if n <= x1:
                return max(y0 + (y1 - y0) * (n - x0) / (x1 - x0), 0.0)
        (x0, y0), (x1, y1) = pts[-2], pts[-1]
        return y1 + (y1 - y0) * (n - x1) / (x1 - x0)
    return sum(sec(int(n)) for n in lengths) / 3600


# ===========================================================================
# 4. Training loop (resumable)
# ===========================================================================
def lr_at(step, total, base_lr, warmup, min_ratio=0.1):
    if step < warmup:
        return base_lr * (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return base_lr * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0))))


def epoch_order(n, seed, epoch):
    idx = list(range(n))
    random.Random(seed * 1000 + epoch).shuffle(idx)
    return idx


def plan_steps(n_examples, seqs_per_step, epochs):
    per_epoch = math.ceil(n_examples / seqs_per_step)
    return per_epoch, per_epoch * epochs


def save_state(pm, opt, state_dir, step, cfg, log_rows):
    import torch
    sd = Path(state_dir)
    tmp = sd.with_name(sd.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    pm.save_pretrained(str(tmp / "adapter"))
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(dict(step=step, cfg=cfg, lib=LIB_VERSION,
                                                    saved=time.time()), indent=2))
    with open(tmp / "train_log.jsonl", "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    if sd.exists():
        shutil.rmtree(sd)
    tmp.rename(sd)


def load_state(pm, opt, state_dir, cfg):
    """Restore adapter + optimizer + step if state_dir holds a compatible run."""
    import torch
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    sd = Path(state_dir)
    if not (sd / "state.json").exists():
        return 0, []
    st = json.loads((sd / "state.json").read_text())
    keys = ("lora_r", "lora_alpha", "lr", "seqs_per_step", "epochs", "seed", "data_fingerprint",
            "max_len")
    diff = {k: (st["cfg"].get(k), cfg.get(k)) for k in keys if st["cfg"].get(k) != cfg.get(k)}
    if diff:
        raise ValueError(f"checkpoint in {sd} was made with different settings: {diff}. "
                         "Use a new run name or restore the settings.")
    ad = sd / "adapter"
    f = ad / "adapter_model.safetensors"
    set_peft_model_state_dict(pm, load_file(str(f)))
    opt.load_state_dict(torch.load(sd / "optimizer.pt", map_location="cpu", weights_only=False))
    log = [json.loads(l) for l in (sd / "train_log.jsonl").read_text().splitlines() if l.strip()]
    return int(st["step"]), log


def train(pm, data, cfg, state_dir, device, log=print, on_checkpoint=None, stop=None):
    """data: DataFrame from attach_tokens (filtered by max_len). cfg keys:
    lora_r, lora_alpha, lr, warmup, seqs_per_step, epochs, seed, max_len,
    ckpt_every_min, chunk, data_fingerprint, max_steps (optional)."""
    import torch
    params = [p for p in pm.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg["lr"], betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    step, log_rows = load_state(pm, opt, state_dir, cfg)
    n = len(data)
    per_epoch, total = plan_steps(n, cfg["seqs_per_step"], cfg["epochs"])
    # max_steps only stops this call early; the LR schedule always follows the full plan
    stop_at = min(total, int(cfg["max_steps"])) if cfg.get("max_steps") else total
    if step:
        log(f"resumed at step {step}/{total} from {state_dir}")
    P = list(data.prompt_ids)
    C = list(data.completion_ids)
    pm.train()
    last_ckpt = time.time()
    t_run = time.time()
    try:
        while step < stop_at:
            if stop is not None and stop():
                log("stop requested")
                break
            epoch, k = divmod(step, per_epoch)
            order = epoch_order(n, cfg["seed"], epoch)
            batch = order[k * cfg["seqs_per_step"]:(k + 1) * cfg["seqs_per_step"]]
            n_tok = sum(len(C[i]) for i in batch)
            lr = lr_at(step, total, cfg["lr"], cfg["warmup"])
            for gp in opt.param_groups:
                gp["lr"] = lr
            t0 = time.time()
            loss_sum = 0.0
            for i in batch:
                ids = torch.tensor(P[i] + C[i], dtype=torch.long, device=device)
                ls, _ = seq_loss_sum(pm, ids, len(P[i]), cfg.get("chunk", 2048))
                (ls / n_tok).backward()
                loss_sum += float(ls.detach())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - t0
            row = dict(step=step, epoch=epoch, loss=round(loss_sum / n_tok, 5), tokens=n_tok,
                       seqs=len(batch), lr=lr, grad_norm=round(gn, 4), sec=round(dt, 1),
                       tok_per_s=round(sum(len(P[i]) + len(C[i]) for i in batch) / dt, 1))
            log_rows.append(row)
            eta_h = (total - step) * dt / 3600
            log(f"step {step}/{total}  loss {row['loss']:.4f}  gn {gn:.3f}  lr {lr:.2e}  "
                f"{n_tok} tok  {dt:.0f}s  ~{eta_h:.1f} h left")
            if (time.time() - last_ckpt > 60 * cfg.get("ckpt_every_min", 20)
                    or step in (total, stop_at)):
                save_state(pm, opt, state_dir, step, cfg, log_rows)
                last_ckpt = time.time()
                if on_checkpoint:
                    on_checkpoint(step)
    except KeyboardInterrupt:
        log("interrupted - saving a checkpoint")
        save_state(pm, opt, state_dir, step, cfg, log_rows)
        if on_checkpoint:
            on_checkpoint(step)
        raise
    return step, total, log_rows


# ===========================================================================
# 5. Merge + export in the base checkpoint layout
# ===========================================================================
def merged_weights(pm):
    """{module name: merged bf16 weight} computed in fp32:
    W + scaling * B @ A, rounded to bf16 once."""
    import torch
    out = {}
    for name, m in lora_modules(pm).items():
        W = m.base_layer.weight
        A = m.lora_A["default"].weight
        B = m.lora_B["default"].weight
        s = m.scaling["default"]
        with torch.no_grad():
            out[name] = (W.float() + s * (B.float() @ A.float())).to(W.dtype).cpu()
    return out


def ckpt_key_for(module_name, ckpt_keys):
    """'model.layers.N.x.y' -> the checkpoint key of its weight. The real
    checkpoint stores the text tower as model.language_model.layers..."""
    assert module_name.startswith("model.layers."), module_name
    rest = module_name[len("model."):] + ".weight"
    cands = [f"model.language_model.{rest}", f"model.{rest}", f"language_model.model.{rest}"]
    hit = [c for c in cands if c in ckpt_keys]
    if len(hit) != 1:
        raise KeyError(f"{module_name}: candidates {cands} -> {hit}")
    return hit[0]


def read_index(base_dir):
    base_dir = Path(base_dir)
    idx = base_dir / "model.safetensors.index.json"
    if idx.exists():
        return json.loads(idx.read_text())["weight_map"]
    from safetensors import safe_open
    out = {}
    for f in sorted(base_dir.glob("*.safetensors")):
        with safe_open(str(f), "pt") as fh:
            for k in fh.keys():
                out[k] = f.name
    return out


def export_merged(base_dir, out_dir, merged, log=print):
    """Copy the base checkpoint directory and overwrite exactly the merged
    tensors. Every other tensor (vision tower, MTP, embeddings, norms, ...)
    and every non-weight file is copied unchanged."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    wmap = read_index(base_dir)
    by_key = {ckpt_key_for(n, wmap): w for n, w in merged.items()}
    for f in base_dir.iterdir():
        if f.is_file() and not f.name.endswith(".safetensors"):
            shutil.copy2(f, out_dir / f.name)
    shards = sorted(set(wmap.values()))
    n_replaced = 0
    for sh in shards:
        tensors = {}
        with safe_open(str(base_dir / sh), "pt") as fh:
            meta = fh.metadata()
            for k in fh.keys():
                t = fh.get_tensor(k)
                if k in by_key:
                    nw = by_key[k]
                    if tuple(nw.shape) != tuple(t.shape) or nw.dtype != t.dtype:
                        raise ValueError(f"{k}: merged {tuple(nw.shape)} {nw.dtype} vs base "
                                         f"{tuple(t.shape)} {t.dtype}")
                    t = nw.contiguous()
                    n_replaced += 1
                tensors[k] = t
        save_file(tensors, str(out_dir / sh), metadata=meta)
        log(f"  wrote {sh} ({len(tensors)} tensors)")
    if n_replaced != len(by_key):
        raise RuntimeError(f"replaced {n_replaced} tensors, expected {len(by_key)}")
    return dict(shards=len(shards), replaced=n_replaced, keys=len(wmap))


def verify_export(base_dir, out_dir, merged):
    """Same keys/shapes/dtypes as the base; exactly the merged tensors differ
    and equal the merged weights; every other tensor is bit-identical."""
    import torch
    from safetensors import safe_open
    base_dir, out_dir = Path(base_dir), Path(out_dir)
    wmap_b, wmap_o = read_index(base_dir), read_index(out_dir)
    if wmap_b != wmap_o:
        raise AssertionError("weight map / shard assignment differs from the base")
    expect = {ckpt_key_for(n, wmap_b): w for n, w in merged.items()}
    same = changed = 0
    bad = []
    for sh in sorted(set(wmap_b.values())):
        with safe_open(str(base_dir / sh), "pt") as fb, safe_open(str(out_dir / sh), "pt") as fo:
            if set(fb.keys()) != set(fo.keys()):
                bad.append(f"{sh}: key set differs")
                continue
            for k in fb.keys():
                tb, to = fb.get_tensor(k), fo.get_tensor(k)
                if tb.shape != to.shape or tb.dtype != to.dtype:
                    bad.append(f"{k}: shape/dtype")
                elif k in expect:
                    if not torch.equal(to, expect[k]):
                        bad.append(f"{k}: not the merged weight")
                    changed += int(not torch.equal(to, tb))
                elif not torch.equal(tb, to):
                    bad.append(f"{k}: changed but not a LoRA target")
                else:
                    same += 1
    if bad:
        raise AssertionError("; ".join(bad[:10]))
    aux_b = sorted(f.name for f in base_dir.iterdir() if f.is_file() and not f.name.endswith(".safetensors"))
    aux_o = sorted(f.name for f in out_dir.iterdir() if f.is_file() and not f.name.endswith(".safetensors"))
    if aux_b != aux_o:
        raise AssertionError(f"aux files differ: {set(aux_b) ^ set(aux_o)}")
    return dict(tensors=len(wmap_b), unchanged_identical=same, merged=len(expect),
                merged_actually_changed=changed, aux_files=len(aux_o))


def mean_nll(model_or_pm, examples, device, chunk=2048, is_peft=True):
    """Token-mean NLL over completion tokens of a few examples (no grad)."""
    import torch
    tot, ntok = 0.0, 0
    m = model_or_pm
    m.eval()
    with torch.no_grad():
        for p, c in examples:
            ids = torch.tensor(list(p) + list(c), dtype=torch.long, device=device)
            if is_peft:
                ls, n = seq_loss_sum(m, ids, len(p), chunk)
            else:
                out = m.model(input_ids=ids.unsqueeze(0), use_cache=False)
                h = out.last_hidden_state[0, len(p) - 1:-1]
                ls = sum(_ce_chunk(h[s:s + chunk], ids[len(p):][s:s + chunk], m.lm_head.weight)
                         for s in range(0, len(c), chunk))
                n = len(c)
            tot += float(ls)
            ntok += n
    return tot / max(ntok, 1)
'''
    (WORK / "nb3lib.py").write_text(LIB_SRC, encoding="utf-8")
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CS6013 · Notebook 3 — SFT: self-distillation with LoRA (Qwen3.5-4B) · v1.0

    Plan A, step 2: fine-tune the **base** model on its own **shortest correct** solutions, so it
    learns to reach the answer inside the 32K budget. The result is a full bf16 checkpoint in
    the **exact base layout**, which `compress.py` takes via `--checkpoint_path`.

    | step | what | GPU |
    |---|---|---|
    | ① | inputs | – |
    | ② | environment (+ `peft`; optional `flash-linear-attention` for the smoke-test check) | – |
    | ③ | download base model | – |
    | ④ | **training set**: nb1 A/B + nb2 T traces → one example per problem (shortest usable) | – |
    | ⑤ | load model + LoRA | ✓ |
    | ⑥ | **smoke test**: seconds / peak memory per training step at 8K/16K/32K; fla check | ✓ |
    | ⑦ | **train** (resumable; checkpoint to HF every N min) | ✓ |
    | ⑧ | merge LoRA (fp32) → export into a copy of the base checkpoint → verify | ✓ |
    | ⑨ | upload merged model (private HF repo) | – |

    **Training-set rules (agreed):** usable = correct under the grader's parser, finished (EOS)
    before 32K, thinking closed, final answer after `</think>`, no loop, no CJK. One example per
    problem: the shortest usable sample. **AIME 2025 + 2026 and HMMT Feb 2025 are held out** (clean test set).
    Training uses the stored grader token ids (prompt + completion) as-is; loss on completion only.

    **Kernels:** Gated DeltaNet runs on transformers' pure-PyTorch path (what works on molab).
    The fla delta-rule kernel is used only if you select it in ⑦ *and* ⑥ showed it matches.

    **Keep this tab active during ⑦** — molab stops idle sessions. If it stops, re-run ①–⑤ and
    press ⑦ again: training resumes from the last checkpoint on HF.
    """)
    return


@app.cell(hide_code=True)
def _():
    hf_token = mo.ui.text(kind="password", label="HF token (write access)", full_width=True)
    nb1_repo = mo.ui.text(value="grey-cat/cs6013-qwen35-4b-base-traces",
                          label="nb1 traces repo (modules A, B)", full_width=True)
    nb2_repo = mo.ui.text(value="grey-cat/cs6013-qwen35-4b-train-traces",
                          label="nb2 traces repo (module T)", full_width=True)
    model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model")
    run_name = mo.ui.text(value="sft-v1", label="run name (checkpoints live under runs/<name>)")
    art_repo = mo.ui.text(value="grey-cat/cs6013-sft-artifacts",
                          label="HF dataset repo for train set / checkpoints / logs (private)",
                          full_width=True)
    out_repo = mo.ui.text(value="grey-cat/cs6013-qwen35-4b-sft-v1",
                          label="HF model repo for the merged model (private)", full_width=True)
    mo.vstack([mo.md("## ① Inputs"), hf_token, nb1_repo, nb2_repo,
               mo.hstack([model_id, run_name], justify="start"), art_repo, out_repo])
    return art_repo, hf_token, model_id, nb1_repo, nb2_repo, out_repo, run_name


@app.cell(hide_code=True)
def _(art_repo, hf_token, run_name):
    # ---------------- HF helpers ----------------
    def tok():
        return hf_token.value.strip() or None

    def api():
        from huggingface_hub import HfApi
        return HfApi(token=tok())

    def dl(repo, path, repo_type="dataset"):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(repo, path, repo_type=repo_type, token=tok(),
                               local_dir=str(WORK / "hf_pull" / repo.replace("/", "__")))

    def list_files(repo, repo_type="dataset"):
        try:
            return api().list_repo_files(repo, repo_type=repo_type)
        except Exception as exc:
            print(f"  [hf] cannot list {repo}: {type(exc).__name__}")
            return []

    def run_dir():
        return WORK / "runs" / run_name.value.strip()

    def push_run(subdir, message):
        """Upload WORK/runs/<run>/<subdir> to art_repo:runs/<run>/<subdir> (3 tries)."""
        repo = art_repo.value.strip()
        if not repo or not tok():
            print("  [hf] no artifacts repo / token -> kept local only")
            return
        api().create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for _i in range(3):
            try:
                api().upload_folder(repo_id=repo, repo_type="dataset",
                                    folder_path=str(run_dir() / subdir),
                                    path_in_repo=f"runs/{run_name.value.strip()}/{subdir}",
                                    commit_message=message)
                print(f"  [hf] pushed runs/{run_name.value.strip()}/{subdir}")
                return
            except Exception as exc:
                print(f"  [hf] push failed ({_i + 1}/3): {type(exc).__name__}: {exc}")
                time.sleep(20)
        print("  [hf] giving up on this push; files are still on local disk")

    def pull_run(subdir):
        """Download art_repo:runs/<run>/<subdir> into WORK/runs/<run>/<subdir> if present."""
        repo = art_repo.value.strip()
        prefix = f"runs/{run_name.value.strip()}/{subdir}/"
        files = [f for f in list_files(repo) if f.startswith(prefix)]
        if not files:
            return 0
        from huggingface_hub import snapshot_download
        tmp = WORK / "hf_pull" / "run_pull"
        snapshot_download(repo, repo_type="dataset", token=tok(), local_dir=str(tmp),
                          allow_patterns=[prefix + "*"])
        dst = run_dir() / subdir
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(tmp / prefix, dst)
        return len(files)

    return dl, list_files, pull_run, push_run, run_dir, tok


@app.cell(hide_code=True)
def _():
    env_btn = mo.ui.run_button(label="② Check environment + install peft")
    want_fla = mo.ui.checkbox(label="also try to install flash-linear-attention (only for the ⑥ check; failure is fine)")
    mo.vstack([mo.md("## ② Environment\nInstalls `peft` (+ `accelerate`) with **torch and "
                     "transformers pinned to the versions already installed** (nothing heavy is "
                     "upgraded). `causal-conv1d` is never installed (needs nvcc, which molab lacks)."),
               want_fla, env_btn])
    return env_btn, want_fla


@app.cell(hide_code=True)
def _(env_btn, want_fla):
    mo.stop(not env_btn.value, mo.md("*press ②*"))
    _torch = importlib.import_module("torch")
    import transformers as _tf
    _cons = WORK / "constraints.txt"
    _cons.write_text(f"torch=={_torch.__version__}\ntransformers=={_tf.__version__}\n")

    def _pip(*pkgs, optional=False):
        r = subprocess.run([PY, "-m", "pip", "install", "-q", "-c", str(_cons), *pkgs],
                           capture_output=True, text=True)
        if r.returncode:   # molab's torch/transformers builds can't be matched on PyPI ->
            r = subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", *pkgs],
                               capture_output=True, text=True)   # install without resolving
        if r.returncode:
            print(f"pip install {' '.join(pkgs)} FAILED{' (optional - fine)' if optional else ''}:\n"
                  + r.stderr[-1500:])
        return r.returncode == 0

    _need = []
    for _m, _p in (("peft", "peft==0.22.0"), ("accelerate", "accelerate"), ("huggingface_hub", "huggingface_hub"),
                   ("pandas", "pandas"), ("pyarrow", "pyarrow"), ("tabulate", "tabulate"),
                   ("matplotlib", "matplotlib")):
        try:
            importlib.import_module(_m)
        except ImportError:
            _need.append(_p)
    if _need:
        _pip(*_need)
    _fla_line = "not requested"
    if want_fla.value:
        _fla_line = "installed" if _pip("flash-linear-attention", optional=True) else "install failed (fine)"
    _L = importlib.reload(importlib.import_module("nb3lib"))
    STATE["L"] = _L
    import peft as _peft
    _lines = [f"- nb3lib `{_L.LIB_VERSION}`, notebook `{NB_VERSION}`",
              f"- torch `{_torch.__version__}` (cuda `{_torch.version.cuda}`), transformers "
              f"`{_tf.__version__}`, peft `{_peft.__version__}`"]
    if _torch.cuda.is_available():
        _p = _torch.cuda.get_device_properties(0)
        _lines.append(f"- GPU **{_p.name}**, sm{_p.major}{_p.minor}, {_p.total_memory / 2**30:.0f} GiB")
    else:
        _lines.append("- **NO GPU attached** - attach one before ⑤")
    _lines.append(f"- flash-linear-attention: {_fla_line}; importable now: "
                  f"{_L.fla_delta_rule() is not None}")
    _lines.append(f"- disk free {shutil.disk_usage(WORK).free / 2**30:.0f} GiB "
                  "(needs ~10 GiB base + ~10 GiB export)")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    model_btn = mo.ui.run_button(label="③ Download base model")
    mo.vstack([mo.md("## ③ Base model\nDownloads `Qwen/Qwen3.5-4B` (~9 GB, all files: the export "
                     "is a copy of this directory with the LoRA-merged tensors replaced)."),
               model_btn])
    return (model_btn,)


@app.cell(hide_code=True)
def _(model_btn, model_id, tok):
    mo.stop(not model_btn.value, mo.md("*press ③*"))
    _hub = importlib.import_module("huggingface_hub")
    _HfApi, _snap = _hub.HfApi, _hub.snapshot_download
    _rev = _HfApi(token=tok()).model_info(model_id.value.strip()).sha
    _path = _snap(model_id.value.strip(), revision=_rev, token=tok())
    from transformers import AutoTokenizer as _AT
    _t = _AT.from_pretrained(_path)
    STATE.update(base_dir=_path, base_rev=_rev, im_end_id=_t.convert_tokens_to_ids("<|im_end|>"))
    mo.md(f"**Model** `{model_id.value}` @ `{_rev[:12]}` at `{_path}`; "
          f"`<|im_end|>` id = {STATE['im_end_id']}")
    return


@app.cell(hide_code=True)
def _():
    data_btn = mo.ui.run_button(label="④ Build training set (CPU)")
    mo.vstack([mo.md("## ④ Training set\nDownloads `derived/` + `raw/` of nb1 (A, B) and nb2 (T), "
                     "applies the usable-sample rules, keeps the shortest usable sample per "
                     "problem, holds out AIME 2025/2026, and saves "
                     "`runs/<run>/data/train_set.parquet` (also pushed to the artifacts repo). "
                     "**Run it after nb2's top-up (⑦c) has finished.**"),
               data_btn])
    return (data_btn,)


@app.cell(hide_code=True)
def _(data_btn, dl, list_files, nb1_repo, nb2_repo, push_run, run_dir):
    mo.stop(not data_btn.value, mo.md("*press ④*"))
    mo.stop(STATE["L"] is None or STATE["im_end_id"] is None, mo.md("**Run ② and ③ first.**"))
    _pd = importlib.import_module("pandas")
    _L = STATE["L"]
    _der, _raw, _src = [], [], []
    for _repo, _mods in ((nb1_repo.value.strip(), _L.NB1_MODULES), (nb2_repo.value.strip(), _L.NB2_MODULES)):
        _files = list_files(_repo)
        mo.stop(not _files, mo.md(f"**Cannot list `{_repo}`** - check the token / repo name."))
        for _m in _mods:
            _f = f"derived/{_m}.parquet"
            mo.stop(_f not in _files, mo.md(f"**`{_repo}` has no `{_f}`** - derive that module first."))
            _der.append(_pd.read_parquet(dl(_repo, _f)))
            for _rf in sorted(f for f in _files if f.startswith(f"raw/{_m}/") and f.endswith(".parquet")):
                _raw.append(_pd.read_parquet(dl(_repo, _rf), columns=[
                    "uid", "sample_idx", "prompt_token_ids", "completion_token_ids", "finish_reason"]))
                _src.append(f"{_repo}:{_rf}")
    _der = _pd.concat(_der, ignore_index=True)
    _raw = _pd.concat(_raw, ignore_index=True).drop_duplicates(["uid", "sample_idx"])
    _chosen, _rep, _excl = _L.select_shortest(_der)
    _data, _n_app = _L.attach_tokens(_chosen, _raw, STATE["im_end_id"])
    _fp = _L.train_set_fingerprint(_data)
    _dd = run_dir() / "data"
    _dd.mkdir(parents=True, exist_ok=True)
    _data.to_parquet(_dd / "train_set.parquet", index=False)
    _rep.to_csv(_dd / "selection_report.csv", index=False)
    (_dd / "data_info.json").write_text(json.dumps(dict(
        fingerprint=_fp, examples=len(_data), completion_tokens=int(_data.n_completion.sum()),
        held_out_sources=_excl, raw_files=_src, im_end_appended=_n_app, lib=_L.LIB_VERSION,
        created=datetime.now(timezone.utc).isoformat()), indent=2))
    push_run("data", f"train set {_fp}")
    STATE.update(data=_data, data_fp=_fp)
    _lt = _L.length_table(_data)
    mo.vstack([
        mo.md(f"**{len(_data)} training examples** (one per problem), "
              f"{_data.n_completion.sum() / 1e6:.1f} M completion tokens, "
              f"{_data.n_total.sum() / 1e6:.1f} M tokens incl. prompts; fingerprint `{_fp}`. "
              f"Held out: {', '.join(_excl) or 'none found'}. `<|im_end|>` appended to {_n_app} "
              "completions that lacked it."),
        mo.md("### Per source (samples dropped per reason = first failing rule)\n"
              + _rep.to_markdown(index=False)),
        mo.md("### Length of the chosen examples (prompt + completion)\n" + _lt.to_markdown(index=False)),
    ])
    return


@app.cell(hide_code=True)
def _():
    lora_r = mo.ui.number(start=8, stop=256, step=8, value=64, label="LoRA rank r")
    lora_alpha = mo.ui.number(start=8, stop=512, step=8, value=64, label="LoRA alpha")
    load_btn = mo.ui.run_button(label="⑤ Load model + LoRA (GPU)")
    mo.vstack([mo.md("## ⑤ Model + LoRA\nbf16 base on the GPU, PyTorch Gated DeltaNet, torch SDPA, "
                     "gradient checkpointing. LoRA on q/k/v/o, in_proj_qkv/in_proj_z/out_proj, "
                     "gate/up/down of all 32 layers (not lm_head: it is tied to the embedding)."),
               mo.hstack([lora_r, lora_alpha], justify="start"), load_btn])
    return load_btn, lora_alpha, lora_r


@app.cell(hide_code=True)
def _(load_btn, lora_alpha, lora_r):
    mo.stop(not load_btn.value, mo.md("*press ⑤*"))
    mo.stop(STATE["base_dir"] is None, mo.md("**Run ③ first.**"))
    _torch = importlib.import_module("torch")
    _L = STATE["L"]
    if STATE["pm"] is not None:
        del STATE["pm"]
        STATE["pm"] = None
        _torch.cuda.empty_cache()
    _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    _base, _counts = _L.load_base(STATE["base_dir"], _dev)
    _torch.manual_seed(0)
    _pm, _info = _L.add_lora(_base, r=int(lora_r.value), alpha=int(lora_alpha.value))
    STATE.update(pm=_pm, device=_dev, lora=(int(lora_r.value), int(lora_alpha.value)))
    mo.md(f"**Loaded** on {_dev}: {_counts['gdn_modules']} Gated DeltaNet layers pinned to PyTorch "
          f"(norm swapped: {_counts['norm_swapped']}); LoRA modules {_info['lora_modules']}, "
          f"trainable params {_info['trainable_params'] / 1e6:.1f} M; GPU memory "
          f"{(_torch.cuda.memory_allocated() / 2**30) if _torch.cuda.is_available() else 0:.1f} GiB.")
    return


@app.cell(hide_code=True)
def _():
    smoke_lens = mo.ui.multiselect(options=["4096", "8192", "16384", "24576", "32768"],
                                   value=["8192", "16384", "32768"], label="sequence lengths")
    smoke_fla = mo.ui.checkbox(value=True, label="also check the fla delta-rule kernel (if importable)")
    smoke_btn = mo.ui.run_button(label="⑥ Run smoke test (GPU, a few minutes)")
    mo.vstack([mo.md("## ⑥ Smoke test\nOne training step (forward + backward, random tokens) per "
                     "length on the real model, after a short warm-up. Then, if fla is importable, "
                     "one real Gated DeltaNet layer is run with both kernels and outputs + gradients "
                     "are compared; only if they match is fla timed on the full model. Ends with an "
                     "estimate of one epoch on the training set for several length caps."),
               mo.hstack([smoke_lens, smoke_fla], justify="start"), smoke_btn])
    return smoke_btn, smoke_fla, smoke_lens


@app.cell(hide_code=True)
def _(smoke_btn, smoke_fla, smoke_lens):
    mo.stop(not smoke_btn.value, mo.md("*press ⑥*"))
    mo.stop(STATE["pm"] is None, mo.md("**Run ⑤ first.**"))
    _torch = importlib.import_module("torch")
    _pd = importlib.import_module("pandas")
    _L, _pm, _dev = STATE["L"], STATE["pm"], STATE["device"]
    _lens = sorted(int(x) for x in smoke_lens.value)

    def _measure(tag):
        _L.time_train_step(_pm, 1024, _dev)          # warm-up (allocator, SDPA/Triton init)
        out = []
        for n in _lens:
            try:
                r = _L.time_train_step(_pm, n, _dev)
                r["kernel"] = tag
                out.append(r)
                print(f"  {tag:5s} {n:>6} tok: {r['sec']:.1f} s, peak {r['peak_gib']} GiB", flush=True)
            except _torch.cuda.OutOfMemoryError:
                _pm.zero_grad(set_to_none=True)
                _torch.cuda.empty_cache()
                out.append(dict(seq_len=n, sec=None, peak_gib="OOM", tok_per_s=None, kernel=tag))
                print(f"  {tag:5s} {n:>6} tok: OUT OF MEMORY", flush=True)
        return out

    _rows = _measure("torch")
    _fla = dict(ok=False, reason="not requested")
    if smoke_fla.value:
        _fla = _L.fla_check(_pm, _dev)
        print("fla check:", _fla, flush=True)
        if _fla.get("ok"):
            _L.set_delta_rule(_pm.get_base_model(), True)
            try:
                _rows += _measure("fla")
            finally:
                _L.set_delta_rule(_pm.get_base_model(), False)
    STATE["smoke"] = dict(rows=_rows, fla=_fla)
    _tab = _pd.DataFrame(_rows)
    _est = []
    _data = STATE["data"]
    if _data is not None:
        for _k in ("torch", "fla"):
            _pts = [r for r in _rows if r["kernel"] == _k and r["sec"] is not None]
            if len(_pts) < 2:
                continue
            for _cap in (0, 24576, 20480, 16384):
                _d = _data if not _cap else _data[_data.n_total <= _cap]
                _h = _L.estimate_epoch_hours(list(_d.n_total), _pts)
                _est.append(dict(kernel=_k, max_len=_cap or "none", examples=len(_d),
                                 dropped=len(_data) - len(_d),
                                 hours_per_epoch=None if _h is None else round(_h, 1)))
    mo.vstack([
        mo.md("### Step time and peak memory\n" + _tab.to_markdown(index=False)),
        mo.md(f"### fla kernel check\n`{_fla}`" + ("\n\n**fla matches PyTorch - you may select it "
              "in ⑦.**" if _fla.get("ok") else "\n\n**Use the PyTorch kernel in ⑦.**")),
        mo.md("### Estimated time for one epoch\n" + (_pd.DataFrame(_est).to_markdown(index=False)
              if _est else "*run ④ first to get an estimate on the real training set*")),
    ])
    return


@app.cell(hide_code=True)
def _():
    lr = mo.ui.number(start=1e-6, stop=1e-3, step=1e-6, value=1e-4, label="learning rate")
    epochs = mo.ui.number(start=1, stop=5, step=1, value=1, label="epochs")
    seqs = mo.ui.number(start=1, stop=64, step=1, value=16, label="sequences per optimizer step")
    warmup = mo.ui.number(start=0, stop=50, step=1, value=5, label="warm-up steps")
    max_len = mo.ui.number(start=0, stop=33000, step=1024, value=0,
                           label="max length prompt+completion (0 = no cap)")
    kernel = mo.ui.dropdown(options=["torch", "fla"], value="torch", label="delta-rule kernel")
    ckpt_min = mo.ui.number(start=5, stop=120, step=5, value=20, label="checkpoint every N min")
    seed = mo.ui.number(start=0, stop=10_000, step=1, value=0, label="data-order seed")
    train_btn = mo.ui.run_button(label="⑦ Train (resumes automatically)")
    mo.vstack([mo.md("## ⑦ Train\nToken-mean loss over completion tokens, AdamW, warm-up + "
                     "cosine to 10% of the LR, grad-clip 1.0. A checkpoint (adapter + optimizer + "
                     "step) is saved locally and pushed to the artifacts repo every N minutes, on "
                     "a manual stop, and at the end. Pressing ⑦ again resumes - with the **same "
                     "settings** (a changed setting is refused; use a new run name instead)."),
               mo.hstack([lr, epochs, seqs, warmup], justify="start"),
               mo.hstack([max_len, kernel, ckpt_min, seed], justify="start"), train_btn])
    return ckpt_min, epochs, kernel, lr, max_len, seed, seqs, train_btn, warmup


@app.cell(hide_code=True)
def _(ckpt_min, epochs, kernel, lr, max_len, pull_run, push_run, run_dir, seed, seqs,
      train_btn, warmup):
    mo.stop(not train_btn.value, mo.md("*press ⑦*"))
    mo.stop(STATE["pm"] is None, mo.md("**Run ⑤ first.**"))
    _pd = importlib.import_module("pandas")
    _L, _pm, _dev = STATE["L"], STATE["pm"], STATE["device"]
    _data = STATE["data"]
    if _data is None:                               # restarted session: reuse the frozen set
        _f = run_dir() / "data" / "train_set.parquet"
        if not _f.exists():
            pull_run("data")
        mo.stop(not _f.exists(), mo.md("**No training set** - run ④."))
        _data = _pd.read_parquet(_f)
        STATE["data"] = _data
    _cap = int(max_len.value)
    _d = _data if not _cap else _data[_data.n_total <= _cap].reset_index(drop=True)
    if kernel.value == "fla":
        mo.stop(not (STATE.get("smoke") or {}).get("fla", {}).get("ok"),
                mo.md("**fla was not verified in ⑥ in this session** - run ⑥ or pick torch."))
    _L.set_delta_rule(_pm.get_base_model(), kernel.value == "fla")
    _cfg = dict(lora_r=STATE["lora"][0], lora_alpha=STATE["lora"][1], lr=float(lr.value),
                warmup=int(warmup.value), seqs_per_step=int(seqs.value), epochs=int(epochs.value),
                seed=int(seed.value), max_len=_cap, ckpt_every_min=float(ckpt_min.value),
                chunk=2048, data_fingerprint=_L.train_set_fingerprint(_d), kernel=kernel.value,
                base_rev=STATE["base_rev"], lib=_L.LIB_VERSION)
    _sd = run_dir() / "state"
    if not (_sd / "state.json").exists():
        _n = pull_run("state")
        if _n:
            print(f"pulled checkpoint ({_n} files) from the artifacts repo", flush=True)
    _per, _tot = _L.plan_steps(len(_d), _cfg["seqs_per_step"], _cfg["epochs"])
    print(f"{len(_d)} examples ({len(_data) - len(_d)} over the cap), {_tot} optimizer steps, "
          f"kernel {kernel.value}", flush=True)
    _step, _total, _log = _L.train(_pm, _d, _cfg, _sd, _dev, log=lambda s: print(s, flush=True),
                                   on_checkpoint=lambda st: push_run("state", f"checkpoint step {st}"))
    _lg = _pd.DataFrame(_log)
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    _fig, _ax = _plt.subplots(1, 2, figsize=(9, 3))
    _ax[0].plot(_lg.step, _lg.loss, color="#2a78d6", lw=1.2)
    _ax[0].set_title("training loss (token mean)", fontsize=9)
    _ax[1].plot(_lg.step, _lg.grad_norm, color="#eb6834", lw=1.2)
    _ax[1].set_title("grad norm (before clip)", fontsize=9)
    for _a in _ax:
        _a.spines[["top", "right"]].set_visible(False)
        _a.set_xlabel("step", fontsize=8)
        _a.tick_params(labelsize=7)
    _fig.tight_layout()
    mo.vstack([mo.md(f"### Training {'finished' if _step >= _total else 'stopped'}: step "
                     f"{_step}/{_total}"), _fig,
               mo.ui.table(_lg.tail(50), page_size=10)])
    return


@app.cell(hide_code=True)
def _():
    export_btn = mo.ui.run_button(label="⑧ Merge + export + verify (GPU)")
    mo.vstack([mo.md("## ⑧ Merge + export\nLoads the **last saved checkpoint** into the LoRA "
                     "model, merges `W + scaling·B·A` in fp32 (one bf16 rounding), writes a copy "
                     "of the base checkpoint directory with exactly those tensors replaced, then "
                     "verifies: same keys / shapes / dtypes / shards / aux files as the base, every "
                     "other tensor (vision tower, MTP, embeddings, norms) bit-identical. Finally "
                     "the exported model is loaded back and its loss on a few training examples "
                     "is compared with the LoRA model (should match) and the base (should be higher)."),
               export_btn])
    return (export_btn,)


@app.cell(hide_code=True)
def _(export_btn, pull_run, run_dir):
    mo.stop(not export_btn.value, mo.md("*press ⑧ after training*"))
    mo.stop(STATE["pm"] is None, mo.md("**Run ⑤ first.**"))
    _torch = importlib.import_module("torch")
    from peft import set_peft_model_state_dict as _set
    from safetensors.torch import load_file as _load
    _L, _pm, _dev = STATE["L"], STATE["pm"], STATE["device"]
    _sd = run_dir() / "state"
    if not (_sd / "state.json").exists():
        pull_run("state")
    mo.stop(not (_sd / "state.json").exists(), mo.md("**No checkpoint found** - train first (⑦)."))
    _st = json.loads((_sd / "state.json").read_text())
    _set(_pm, _load(str(_sd / "adapter" / "adapter_model.safetensors")))
    print(f"loaded adapter from step {_st['step']}", flush=True)
    _merged = _L.merged_weights(_pm)
    _out = WORK / "exports" / run_dir().name
    _info = _L.export_merged(STATE["base_dir"], _out, _merged, log=lambda s: print(s, flush=True))
    _ver = _L.verify_export(STATE["base_dir"], _out, _merged)
    print("verify:", _ver, flush=True)
    # functional check on a few training examples (cut to <= 6K tokens for speed)
    _data = STATE["data"]
    _ex = []
    if _data is not None:
        for _r in _data.sort_values("n_total").iloc[:: max(1, len(_data) // 4)].head(4).itertuples():
            _c = list(_r.completion_ids)[: max(1, 6000 - len(_r.prompt_ids))]
            _ex.append((list(_r.prompt_ids), _c))
    _nll = {}
    if _ex:
        _nll["lora"] = _L.mean_nll(_pm, _ex, _dev)
        with _pm.disable_adapter():
            _nll["base"] = _L.mean_nll(_pm, _ex, _dev)
        from transformers import AutoModelForCausalLM as _AM
        _re = _AM.from_pretrained(str(_out), dtype=_torch.bfloat16, attn_implementation="sdpa").to(_dev)
        _L.force_torch_kernels(_re)
        _nll["exported"] = _L.mean_nll(_re, _ex, _dev, is_peft=False)
        del _re
        _torch.cuda.empty_cache()
    _ok = (not _ex) or abs(_nll["exported"] - _nll["lora"]) < 0.02
    STATE["export_dir"] = _out
    (_out.parent / f"{_out.name}_export_info.json").write_text(json.dumps(dict(
        step=_st["step"], cfg=_st["cfg"], verify=_ver, nll=_nll, base_rev=STATE["base_rev"],
        created=datetime.now(timezone.utc).isoformat()), indent=2, default=str))
    mo.md(f"**Exported** to `{_out}` ({_info['shards']} shards, {_info['replaced']} tensors "
          f"replaced of {_info['keys']}).  \n**Verify:** {_ver}  \n**Loss on 4 training examples** "
          f"(token mean): " + ", ".join(f"{k} {v:.4f}" for k, v in _nll.items())
          + ("  \n✅ exported model matches the LoRA model." if _ok else
             "  \n❌ **exported model does NOT match the LoRA model - do not upload; send me this output.**"))
    return


@app.cell(hide_code=True)
def _():
    up_btn = mo.ui.run_button(label="⑨ Upload merged model (private)")
    mo.vstack([mo.md("## ⑨ Upload\nCreates the private model repo and uploads the exported "
                     "directory (~9 GB). This is the `--checkpoint_path` / model id for the 20% "
                     "compression run and for local evaluation."), up_btn])
    return (up_btn,)


@app.cell(hide_code=True)
def _(out_repo, push_run, run_dir, tok, up_btn):
    mo.stop(not up_btn.value, mo.md("*press ⑨ after ⑧ passed*"))
    mo.stop(STATE.get("export_dir") is None, mo.md("**Run ⑧ first (in this session).**"))
    _HfApi = importlib.import_module("huggingface_hub").HfApi
    _api = _HfApi(token=tok())
    _repo = out_repo.value.strip()
    _api.create_repo(_repo, repo_type="model", private=True, exist_ok=True)
    _api.upload_folder(repo_id=_repo, repo_type="model", folder_path=str(STATE["export_dir"]),
                       commit_message=f"SFT merged model ({run_dir().name})")
    _info = STATE["export_dir"].parent / f"{STATE['export_dir'].name}_export_info.json"
    (run_dir() / "export").mkdir(parents=True, exist_ok=True)
    shutil.copy(_info, run_dir() / "export" / "export_info.json")
    push_run("export", f"export info -> {_repo}")
    _rev = _api.model_info(_repo).sha
    mo.md(f"**Uploaded** to `{_repo}` @ `{_rev[:12]}`.")
    return


if __name__ == "__main__":
    app.run()
