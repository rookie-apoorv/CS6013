"""Size-neutral recovery training of a codec-E checkpoint (post-compression distillation).

Every coded weight is  w = (symbol + q_min) * step[row, group]  (fp16 step per group),
except escape positions, which hold an exact fp16 outlier value. The integer codes
(the rANS streams) are kept FIXED. Trained, and written back in place:

* every per-group step, as a multiplier  step = step0 * exp(s)  (s starts at 0);
* every outlier value;
* every tensor stored raw (bf16 norms, linear-attention gates/conv, small tensors).

All of these are stored as plain floats of a fixed dtype, so the rewritten
checkpoint has exactly the same tensors, shapes, dtypes and byte size, and the
unchanged ``decompress.py`` restores it. Objective: KL(teacher || student) per
token, teacher = the uncompressed (fine-tuned) bf16 model, on teacher-forced
training traces. Works for format v1 (``mode: ec``) and v2 (``mode: ec_parts``,
P3-lite pruned / split tensors).

Only needs: torch, numpy, safetensors, transformers and this submission's
``decompression`` package (rANS decoder + shard reader).
"""

from __future__ import annotations

import json
import math
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LIB_VERSION = "recovery-2026-11-23a"


# ===========================================================================
# 1. Reading the compressed checkpoint
# ===========================================================================
def load_config(cdir):
    return json.loads((Path(cdir) / "compression_config.json").read_text())


def coded_parts(config, store):
    """[(ckpt name, shape, dtype, [part dict])] for every coded tensor. A part has
    vname, meta, axis (None = whole tensor) and keep (bool np array or None)."""
    out = []
    for name, meta in config["tensors"].items():
        if meta["mode"] == "ec":
            out.append((name, tuple(meta["shape"]), meta["dtype"],
                        [dict(vname=name, meta=meta, axis=None, keep=None)]))
        elif meta["mode"] == "ec_parts":
            parts = []
            for pm in meta["parts"]:
                keep = np.unpackbits(store.get(pm["keep_key"]).numpy().view(np.uint8))
                keep = keep[: int(pm["axis_len"])].astype(bool)
                parts.append(dict(vname=pm["name"], meta=pm, axis=int(pm["axis"]), keep=keep))
            out.append((name, tuple(meta["shape"]), meta["dtype"], parts))
    return out


def raw_names(config):
    return [n for n, m in config["tensors"].items() if m["mode"] == "raw"]


def decode_part(store, vname, meta):
    """-> codes (int8/int16 [rows, cols], 0 at escapes), flat escape indices (int64),
    steps (fp16 [rows, cols/G]), outlier values (fp16 [n])."""
    from decompression.rans import decode
    rows, cols = int(meta["rows"]), int(meta["cols"])
    dts = meta["stream_dtypes"]
    sym = decode(store.get(f"{vname}::words").numpy().view(np.dtype(dts["words"])),
                 store.get(f"{vname}::state").numpy().view(np.dtype(dts["state"])),
                 store.get(f"{vname}::count").numpy().view(np.dtype(dts["count"])),
                 store.get(f"{vname}::freq").numpy().view(np.dtype(dts["freq"])),
                 rows * cols)
    esc = int(meta["escape"])
    is_esc = sym == esc
    codes = sym.astype(np.int64) + int(meta["q_min"])
    codes[is_esc] = 0
    lo, hi = int(codes.min()), int(codes.max())
    dt = np.int8 if (lo >= -128 and hi <= 127) else np.int16
    if lo < -32768 or hi > 32767:
        dt = np.int32
    codes = torch.from_numpy(codes.astype(dt).reshape(rows, cols))
    esc_idx = torch.from_numpy(np.flatnonzero(is_esc).astype(np.int64))
    steps = store.get(f"{vname}::steps").to(torch.float16)
    n_out = int(meta.get("outliers", 0))
    outl = (store.get(f"{vname}::outliers").to(torch.float16) if n_out
            else torch.zeros(0, dtype=torch.float16))
    if outl.numel() != esc_idx.numel():
        raise ValueError(f"{vname}: {outl.numel()} outliers vs {esc_idx.numel()} escapes")
    return codes, esc_idx, steps, outl


# ===========================================================================
# 2. Student: weights rebuilt from fixed codes + trainable floats
# ===========================================================================
class CodecPart(nn.Module):
    def __init__(self, vname, codes, esc_idx, steps, outl, G, axis, keep):
        super().__init__()
        self.vname, self.G, self.axis = vname, int(G), axis
        self.rows, self.cols = codes.shape
        self.register_buffer("codes", codes, persistent=False)
        self.register_buffer("esc_idx", esc_idx, persistent=False)
        self.register_buffer("steps0", steps.float(), persistent=False)
        self.log_s = nn.Parameter(torch.zeros_like(self.steps0))
        self.outl = nn.Parameter(outl.float().clone())
        if keep is not None:
            self.register_buffer("keep_idx", torch.from_numpy(np.flatnonzero(keep)),
                                 persistent=False)
        else:
            self.keep_idx = None

    def steps(self):
        return self.steps0 * torch.exp(self.log_s)

    def weight_fp32(self, rounded=False):
        st = self.steps()
        if rounded:
            st = st.to(torch.float16).float()
        w = self.codes.float() * st.repeat_interleave(self.G, dim=1)
        if self.esc_idx.numel():
            o = self.outl.to(torch.float16).float() if rounded else self.outl
            w = w.reshape(-1).index_put((self.esc_idx,), o).reshape(self.rows, self.cols)
        return w

    def stored(self):
        """(steps fp16, outliers fp16) exactly as they will be written."""
        with torch.no_grad():
            st = self.steps().to(torch.float16)
            if not torch.isfinite(st).all() or bool(((st == 0) & (self.steps0 != 0)).any()):
                raise FloatingPointError(f"{self.vname}: step underflow/overflow in fp16")
            return st.cpu(), self.outl.detach().to(torch.float16).cpu()


class CodecWeight(nn.Module):
    """torch.nn.utils.parametrize module: ignores the stored tensor and returns the
    weight rebuilt from its codec parts (zeros where no part writes = pruned)."""

    def __init__(self, shape, dtype, parts):
        super().__init__()
        self.shape, self.dtype = tuple(shape), dtype
        self.parts = nn.ModuleList(parts)

    def build(self, rounded=False):
        if len(self.parts) == 1 and self.parts[0].keep_idx is None:
            w = self.parts[0].weight_fp32(rounded)
        else:
            w = torch.zeros(self.shape, dtype=torch.float32, device=self.parts[0].codes.device)
            for p in self.parts:
                w = w.index_copy(p.axis, p.keep_idx, p.weight_fp32(rounded))
        return w.reshape(self.shape).to(self.dtype)

    def forward(self, _ignored):
        return self.build(False)


def ckpt_to_module_param(name):
    """checkpoint key -> parameter name in AutoModelForCausalLM (text model)."""
    for pre in ("model.language_model.", "language_model.model."):
        if name.startswith(pre):
            return "model." + name[len(pre):]
    if name.startswith("model."):
        return name
    return None


def _get_module(model, dotted):
    m = model
    for part in dotted.split("."):
        m = getattr(m, part)
    return m


def build_student(restored_dir, compressed_dir, device, log=print):
    """Load the decompressed (restored) checkpoint as the model, then replace every
    coded weight by a CodecWeight parametrization and make raw tensors trainable."""
    from transformers import AutoModelForCausalLM
    from torch.nn.utils import parametrize
    from decompression.io_utils import CompressedStore
    model = AutoModelForCausalLM.from_pretrained(str(restored_dir), dtype=torch.bfloat16,
                                                 attn_implementation="sdpa", low_cpu_mem_usage=True)
    model.config.use_cache = False
    n_gdn = force_torch_kernels(model)    # before collecting params: may swap norm modules
    cfg = load_config(compressed_dir)
    store = CompressedStore(Path(compressed_dir))
    params = dict(model.named_parameters())
    for p in model.parameters():
        p.requires_grad_(False)
    codec_mods, n_codec = {}, 0
    t0 = time.time()
    for name, shape, dt, parts in coded_parts(cfg, store):
        pname = ckpt_to_module_param(name)
        if pname is None or pname not in params:
            continue                      # e.g. MTP / vision (not in the text model)
        mods = []
        for p in parts:
            codes, esc, steps, outl = decode_part(store, p["vname"], p["meta"])
            mods.append(CodecPart(p["vname"], codes, esc, steps, outl, p["meta"]["group_size"],
                                  p["axis"], p["keep"]))
        cw = CodecWeight(shape, getattr(torch, dt), mods).to(device)
        modname, attr = pname.rsplit(".", 1)
        mod = _get_module(model, modname)
        setattr(mod, attr, nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False))
        parametrize.register_parametrization(mod, attr, cw, unsafe=True)
        codec_mods[name] = cw
        n_codec += 1
        if n_codec % 50 == 0:
            log(f"  decoded {n_codec} coded tensors ({time.time() - t0:.0f}s)")
    store.close()
    # the tied LM head would still hold the old embedding copy: drop it (logits are
    # computed from the parametrized embedding)
    if hasattr(model, "lm_head"):
        model.lm_head.weight = nn.Parameter(torch.empty(0, dtype=torch.bfloat16), requires_grad=False)
    raw = {}
    for n in raw_names(cfg):
        pn = ckpt_to_module_param(n)
        if pn is not None and pn in params and params[pn].numel() > 0:
            params[pn].requires_grad_(True)
            raw[n] = params[pn]
    model.to(device)
    log(f"[student] {len(codec_mods)} coded tensors parametrized, {len(raw)} raw tensors trainable, "
        f"{n_gdn} Gated DeltaNet layers on the PyTorch path ({time.time() - t0:.0f}s)")
    return model, codec_mods, raw


def trainable_groups(codec_mods, raw, lr_scale, lr_outlier, lr_raw):
    s = [p.log_s for cw in codec_mods.values() for p in cw.parts]
    o = [p.outl for cw in codec_mods.values() for p in cw.parts if p.outl.numel()]
    r = list(raw.values())
    groups = [dict(params=s, lr=lr_scale, name="log_step"),
              dict(params=o, lr=lr_outlier, name="outliers"),
              dict(params=r, lr=lr_raw, name="raw")]
    counts = {g["name"]: sum(p.numel() for p in g["params"]) for g in groups}
    return [g for g in groups if g["params"]], counts


@torch.no_grad()
def exactness_check(codec_mods, restored_dir, log=print):
    """At init the student must equal the decompressed checkpoint bit for bit."""
    from safetensors import safe_open
    files = sorted(Path(restored_dir).glob("*.safetensors"))
    where = {}
    for f in files:
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                where[k] = f
    bad, n = [], 0
    for name, cw in codec_mods.items():
        with safe_open(str(where[name]), "pt") as h:
            ref = h.get_tensor(name)
        got = cw.build(False).cpu()
        n += 1
        if not torch.equal(got, ref):
            bad.append((name, float((got.float() - ref.float()).abs().max())))
    log(f"[check] student == decompressed checkpoint: {n - len(bad)}/{n} coded tensors bit-identical")
    return bad


# ===========================================================================
# 3. Kernels (same policy as compress.py / nb3) and the KD loss
# ===========================================================================
def force_torch_kernels(model):
    mod = None
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            mod = __import__(type(m).__module__, fromlist=["x"])
            break
    if mod is None:
        return 0
    norm_cls = next((getattr(mod, n) for n in dir(mod) if n.endswith("RMSNormGated")
                     and not n.startswith("Fused")), None)
    n = 0
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            m.causal_conv1d_fn = None
            m.chunk_gated_delta_rule = mod.torch_chunk_gated_delta_rule
            if hasattr(mod, "torch_causal_conv1d_update"):
                m.causal_conv1d_update = mod.torch_causal_conv1d_update
            if norm_cls is not None and not isinstance(m.norm, norm_cls):
                old = m.norm
                new = norm_cls(old.weight.shape[0], eps=getattr(old, "eps", 1e-6))
                with torch.no_grad():
                    new.weight.copy_(old.weight)
                new.weight.requires_grad_(old.weight.requires_grad)
                m.norm = new.to(device=old.weight.device, dtype=old.weight.dtype)
            n += 1
    return n


def set_delta_rule(model, use_fla):
    fn = None
    if use_fla:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fn
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            mod = __import__(type(m).__module__, fromlist=["x"])
            m.chunk_gated_delta_rule = fn if use_fla else mod.torch_chunk_gated_delta_rule


def _kl_chunk(hs, ht, ws, wt):
    ls = (hs @ ws.t()).float().log_softmax(-1)
    lt = (ht @ wt.t()).float().log_softmax(-1)
    return (lt.exp() * (lt - ls)).sum()


def teacher_hidden(teacher, ids, plen):
    with torch.no_grad():
        h = teacher.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state
        return h[0, plen - 1:-1].detach()


def teacher_head(teacher):
    w = teacher.lm_head.weight
    if w.numel() == 0:
        w = teacher.model.embed_tokens.weight
    return w


def kd_loss_sum(student, ids, plen, ht, wt, chunk=1024):
    """Sum over completion positions of KL(teacher || student); logits in
    checkpointed chunks (never the full [len x vocab] matrix)."""
    from torch.utils.checkpoint import checkpoint
    hs = student.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state[0, plen - 1:-1]
    ws = student.model.embed_tokens.weight          # parametrized (rebuilt) tied head
    total = hs.new_zeros((), dtype=torch.float32)
    for a in range(0, hs.shape[0], chunk):
        total = total + checkpoint(_kl_chunk, hs[a:a + chunk], ht[a:a + chunk], ws, wt,
                                   use_reentrant=False)
    return total, int(hs.shape[0])


@torch.no_grad()
def mean_kl(student_or_model, teacher, examples, device, chunk=1024, parametrized=True):
    """Token-mean KL(teacher || model) over completion tokens (no grad)."""
    tot, n = 0.0, 0
    wt = teacher_head(teacher)
    for p, c in examples:
        ids = torch.tensor(list(p) + list(c), dtype=torch.long, device=device)
        ht = teacher_hidden(teacher, ids, len(p))
        m = student_or_model
        hs = m.model(input_ids=ids.unsqueeze(0), use_cache=False).last_hidden_state[0, len(p) - 1:-1]
        ws = m.model.embed_tokens.weight
        if not parametrized and m.lm_head.weight.numel():
            ws = m.lm_head.weight
        tot += float(sum(_kl_chunk(hs[a:a + chunk], ht[a:a + chunk], ws, wt)
                         for a in range(0, hs.shape[0], chunk)))
        n += hs.shape[0]
    return tot / max(n, 1)


# ===========================================================================
# 4. Training loop (resumable)
# ===========================================================================
def lr_factor(step, total, warmup, min_ratio=0.1):
    if step < warmup:
        return (step + 1) / warmup
    p = (step - warmup) / max(1, total - warmup)
    return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))


def _trainables(codec_mods, raw):
    d = {}
    for cw in codec_mods.values():
        for p in cw.parts:
            d[f"{p.vname}::log_s"] = p.log_s
            if p.outl.numel():
                d[f"{p.vname}::outl"] = p.outl
    for n, p in raw.items():
        d[f"raw::{n}"] = p
    return d


def save_state(state_dir, codec_mods, raw, opt, step, cfg, log_rows):
    sd = Path(state_dir)
    tmp = sd.with_name(sd.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in _trainables(codec_mods, raw).items()},
               tmp / "trainables.pt")
    torch.save(opt.state_dict(), tmp / "optimizer.pt")
    (tmp / "state.json").write_text(json.dumps(dict(step=step, cfg=cfg, lib=LIB_VERSION), indent=2))
    with open(tmp / "train_log.jsonl", "w") as f:
        for r in log_rows:
            f.write(json.dumps(r) + "\n")
    shutil.rmtree(sd, ignore_errors=True)
    tmp.rename(sd)


def load_state(state_dir, codec_mods, raw, opt, cfg):
    sd = Path(state_dir)
    if not (sd / "state.json").exists():
        return 0, []
    st = json.loads((sd / "state.json").read_text())
    keys = ("lr_scale", "lr_outlier", "lr_raw", "seqs_per_step", "epochs", "seed", "max_len",
            "data_fingerprint", "compressed_sha")
    diff = {k: (st["cfg"].get(k), cfg.get(k)) for k in keys if st["cfg"].get(k) != cfg.get(k)}
    if diff:
        raise ValueError(f"checkpoint in {sd} made with different settings: {diff}")
    saved = torch.load(sd / "trainables.pt", map_location="cpu")
    tr = _trainables(codec_mods, raw)
    with torch.no_grad():
        for k, v in saved.items():
            tr[k].copy_(v.to(tr[k].device, tr[k].dtype))
    opt.load_state_dict(torch.load(sd / "optimizer.pt", map_location="cpu", weights_only=False))
    log = [json.loads(l) for l in (sd / "train_log.jsonl").read_text().splitlines() if l.strip()]
    return int(st["step"]), log


def train(student, teacher, codec_mods, raw, data, cfg, state_dir, device, log=print,
          on_checkpoint=None):
    """data: list of (prompt ids, completion ids) already cut to cfg['max_len']."""
    groups, counts = trainable_groups(codec_mods, raw, cfg["lr_scale"], cfg["lr_outlier"],
                                      cfg["lr_raw"])
    base_lr = [g["lr"] for g in groups]
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8)
    step, log_rows = load_state(state_dir, codec_mods, raw, opt, cfg)
    n = len(data)
    per_epoch = math.ceil(n / cfg["seqs_per_step"])
    total = per_epoch * cfg["epochs"]
    stop_at = min(total, int(cfg["max_steps"])) if cfg.get("max_steps") else total
    if step:
        log(f"resumed at step {step}/{total}")
    wt = teacher_head(teacher)
    params = [p for g in groups for p in g["params"]]
    student.train()
    last = time.time()
    try:
        while step < stop_at:
            epoch, k = divmod(step, per_epoch)
            order = list(range(n))
            random.Random(cfg["seed"] * 1000 + epoch).shuffle(order)
            batch = order[k * cfg["seqs_per_step"]:(k + 1) * cfg["seqs_per_step"]]
            n_tok = sum(len(data[i][1]) for i in batch)
            f = lr_factor(step, total, cfg["warmup"])
            for g, lr in zip(opt.param_groups, base_lr):
                g["lr"] = lr * f
            t0 = time.time()
            kl_sum = 0.0
            for i in batch:
                p, c = data[i]
                ids = torch.tensor(list(p) + list(c), dtype=torch.long, device=device)
                ht = teacher_hidden(teacher, ids, len(p))
                ls, _ = kd_loss_sum(student, ids, len(p), ht, wt, cfg.get("chunk", 1024))
                (ls / n_tok).backward()
                kl_sum += float(ls.detach())
            gn = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            dt = time.time() - t0
            mult = torch.cat([p.log_s.detach().flatten() for cw in codec_mods.values()
                              for p in cw.parts])
            row = dict(step=step, kl=round(kl_sum / n_tok, 6), tokens=n_tok, grad_norm=round(gn, 4),
                       lr_factor=round(f, 4), sec=round(dt, 1),
                       step_mult_absmax=round(float(mult.abs().max().exp()), 4))
            log_rows.append(row)
            log(f"step {step}/{total}  KL {row['kl']:.5f}  gn {gn:.3f}  "
                f"max step change x{row['step_mult_absmax']:.3f}  {dt:.0f}s  "
                f"~{(total - step) * dt / 3600:.1f} h left")
            if time.time() - last > 60 * cfg.get("ckpt_every_min", 20) or step in (total, stop_at):
                save_state(state_dir, codec_mods, raw, opt, step, cfg, log_rows)
                last = time.time()
                if on_checkpoint:
                    on_checkpoint(step)
    except KeyboardInterrupt:
        log("interrupted - saving a checkpoint")
        save_state(state_dir, codec_mods, raw, opt, step, cfg, log_rows)
        if on_checkpoint:
            on_checkpoint(step)
        raise
    return step, total, log_rows, counts


# ===========================================================================
# 5. Write back + verify
# ===========================================================================
def export(compressed_dir, out_dir, codec_mods, raw, info=None, log=print):
    """Copy the compressed checkpoint and overwrite, in place, every ::steps and
    ::outliers tensor and every raw tensor with the trained values (same dtype and
    shape). The rANS streams, masks, config and file sizes are unchanged."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    src, out = Path(compressed_dir), Path(out_dir)
    shutil.rmtree(out, ignore_errors=True)
    shutil.copytree(src, out)
    new = {}
    for cw in codec_mods.values():
        for p in cw.parts:
            st, ol = p.stored()
            new[f"{p.vname}::steps"] = st
            if ol.numel():
                new[f"{p.vname}::outliers"] = ol
    for n, prm in raw.items():
        new[n] = prm.detach().cpu()
    done = set()
    for f in sorted(out.glob("*.safetensors")):
        size0 = f.stat().st_size
        with safe_open(str(f), "pt") as h:
            meta = h.metadata()
            keys = list(h.keys())
            if not any(k in new for k in keys):
                continue
            tensors = {k: h.get_tensor(k) for k in keys}
        for k in keys:
            if k in new:
                old = tensors[k]
                t = new[k].to(old.dtype).reshape(old.shape).contiguous()
                tensors[k] = t
                done.add(k)
        tmp = f.with_suffix(".tmp")
        save_file(tensors, str(tmp), metadata=meta)
        if tmp.stat().st_size != size0:
            raise RuntimeError(f"{f.name}: size changed {size0} -> {tmp.stat().st_size}")
        tmp.replace(f)
        log(f"  rewrote {f.name} ({sum(k in new for k in keys)} tensors, size unchanged)")
    missing = set(new) - done
    if missing:
        raise RuntimeError(f"{len(missing)} trained tensors not found in the shards: {sorted(missing)[:5]}")
    if info is not None:
        (out.parent / f"{out.name}_recovery_info.json").write_text(json.dumps(info, indent=2, default=str))
    return dict(replaced=len(done))


def dir_bytes(d, suffix=".safetensors"):
    return sum(f.stat().st_size for f in Path(d).glob(f"*{suffix}"))


@torch.no_grad()
def compare_restored(codec_mods, restored_dir, log=print):
    """Decompressed (exported) weights vs the student rebuilt with the fp16-rounded
    steps/outliers it was written with: must be bit-identical."""
    from safetensors import safe_open
    where = {}
    for f in sorted(Path(restored_dir).glob("*.safetensors")):
        with safe_open(str(f), "pt") as h:
            for k in h.keys():
                where[k] = f
    bad, n = [], 0
    for name, cw in codec_mods.items():
        with safe_open(str(where[name]), "pt") as h:
            ref = h.get_tensor(name)
        got = cw.build(rounded=True).cpu()
        n += 1
        if not torch.equal(got, ref):
            bad.append((name, float((got.float() - ref.float()).abs().max())))
    log(f"[check] decompressed recovered checkpoint == student (fp16-rounded): {n - len(bad)}/{n}")
    return bad


def cut_examples(prompt_ids, completion_ids, max_len):
    """[(p, c)] with p + c cut to max_len tokens (completion part only cut)."""
    out = []
    for p, c in zip(prompt_ids, completion_ids):
        p, c = list(p), list(c)
        keep = max(0, max_len - len(p))
        if keep >= 64:
            out.append((p, c[:keep]))
    return out


def data_fingerprint(examples):
    import hashlib
    h = hashlib.sha256()
    for p, c in examples:
        h.update(f"{len(p)}|{len(c)}|{c[:4]}|{c[-4:]}".encode())
    return h.hexdigest()[:16]
