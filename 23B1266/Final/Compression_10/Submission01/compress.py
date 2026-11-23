"""Compression entry point (final_sft_p3dtw_10 = P3-lite 10% + decision-token-weighted Fisher).

    python compress.py --model_name <name> --checkpoint_path <base dir> --output_path <out dir>

Method: codec E grid + Fisher bit allocation + Fisher outliers + GPTQ on the
grid + numpy rANS. All constants are in ``compression/config.py``; the
calibration traces are bundled in ``compression/calib_data/aime_calib.npz``.
``CS6013_SMOKE=1`` runs a fast structural test (never submit its output).

Steps
-----
1. inventory from safetensors headers (coded / bf16 / dropped tensors)
2. load bf16 model (pure-PyTorch kernels) and the calibration file
3. Fisher pass -> P3-lite structure (MLP neurons to remove by OBD saliency;
   embedding split into common / rare token rows) -> every coded tensor or
   part becomes a "virtual tensor" -> outlier masks -> rate/loss tables ->
   per-virtual-tensor k under the budget, checked against the exact entropy
4. tensors outside the decoder stack (tied embedding): RTN on the grid
5. GPTQ layer by layer over the calibration traces (pruned down_proj columns
   are first folded into the kept ones by least squares); after each layer the bits
   actually used are compared with the plan and the remaining layers are
   re-planned if they drift (rate control)
6. rANS-encode every code stream (each one is decoded again and compared),
   write shards, copy config/tokenizer, re-measure size_frac from the headers
Side outputs go to ``<output_path>_artifacts/``.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import struct
import time
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from compression import FORMAT_VERSION, config
from compression import structure as S
from compression.io_utils import (ShardedSafetensorsWriter, copy_auxiliary_files,
                                  iter_tensors, list_weight_files, resolve_checkpoint_dir)

_VISION = {"visual", "vision", "vision_tower", "vision_model"}


def _log(msg):
    print(msg, flush=True)


def is_scored(name):
    n = name.replace("\\", "/").lower()
    return not ("visual." in n or any(p in _VISION for p in n.split(".")))


def is_dropped(name):
    return bool(set(name.split(".")) & set(config.DROP_COMPONENTS))


def inventory(base_dir):
    cands, raw, dropped = {}, {}, {}
    scored_bytes = fixed_bytes = 0
    for f in list_weight_files(base_dir):
        with safe_open(str(f), framework="pt") as h:
            for k in h.keys():
                sl = h.get_slice(k)
                shape = tuple(sl.get_shape())
                dt = sl.get_dtype()
                nb = int(np.prod(shape)) * {"BF16": 2, "F16": 2, "F32": 4}.get(dt, 1)
                if is_scored(k):
                    scored_bytes += nb
                if is_dropped(k):
                    dropped[k] = shape
                elif (len(shape) == 2 and dt in ("BF16", "F16", "F32")
                      and int(np.prod(shape)) >= config.MIN_NUMEL and is_scored(k)):
                    cands[k] = shape
                else:
                    raw[k] = shape
                    if is_scored(k):
                        fixed_bytes += nb
    return cands, raw, dropped, scored_bytes, fixed_bytes


def measure_scored_bytes(directory):
    total = 0
    for f in sorted(Path(directory).glob("*.safetensors")):
        with open(f, "rb") as fh:
            hdr = json.loads(fh.read(struct.unpack("<Q", fh.read(8))[0]))
        for k, v in hdr.items():
            if k != "__metadata__" and is_scored(k):
                total += v["data_offsets"][1] - v["data_offsets"][0]
    return total


def load_calib(path):
    z = np.load(path)
    ids, off, ln, pl = z["ids"], z["offsets"], z["lengths"], z["prompt_lens"]
    seqs = [torch.from_numpy(ids[o:o + n].astype(np.int64))[None] for o, n in zip(off, ln)]
    gptq = [s for s, g in zip(seqs, z["is_gptq"]) if g]
    fisher = [(s, int(p)) for s, p, f in zip(seqs, pl, z["is_fisher"]) if f]
    return gptq, fisher, hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cap_tokens(samples, max_tokens):
    out, tok = [], 0
    for s in samples:
        if tok >= max_tokens:
            break
        out.append(s)
        tok += s.shape[1]
    return out, tok


def build_parts(names, cands, params, fisher, log):
    """P3-lite: {ckpt name: [Part, ...]} for pruned MLP tensors and the split
    embedding, plus a summary dict."""
    parts, info = {}, {}
    if config.PRUNE_FRAC > 0:
        tri = S.mlp_triples(names)
        if not tri:
            raise RuntimeError("PRUNE_FRAC > 0 but no gate/up/down triples among the coded tensors")
        sal = {li: S.neuron_saliency(params[t["down"]].detach(), fisher[t["down"]])
               for li, t in tri.items()}
        keep, n_rem, tot = S.select_prune(sal, config.PRUNE_FRAC, config.PRUNE_GRANULE,
                                          config.PRUNE_LAYER_CAP)
        for li, t in tri.items():
            k = keep[li]
            if bool(k.all()):
                continue
            for kind, axis in (("gate", 0), ("up", 0), ("down", 1)):
                n = t[kind]
                parts[n] = [S.Part(f"{n}#k", n, axis, k, "k")]
        per_layer = {int(li): int((~keep[li]).sum()) for li in tri}
        s_rem = sum(float(sal[li][~keep[li]].sum()) for li in tri)
        s_tot = sum(float(sal[li].sum()) for li in tri)
        info["prune"] = dict(removed=n_rem, total=tot, frac=n_rem / max(tot, 1),
                             per_layer=per_layer, saliency_removed_share=s_rem / max(s_tot, 1e-30))
        log(f"[prune] removed {n_rem:,}/{tot:,} MLP neurons ({n_rem / max(tot, 1) * 100:.1f}%), "
            f"{s_rem / max(s_tot, 1e-30) * 100:.2f}% of the total OBD saliency | per layer: "
            + " ".join(f"{li}:{v}" for li, v in per_layer.items() if v))
        info["prune_keep"] = {int(li): keep[li] for li in tri}
    if config.EMB_SPLIT:
        emb = [n for n in names if n.endswith("embed_tokens.weight")]
        if len(emb) != 1:
            raise RuntimeError(f"EMB_SPLIT: expected one embed_tokens tensor, found {emb}")
        e = emb[0]
        if not Path(config.TOKEN_FREQ_FILE).exists():
            raise FileNotFoundError(
                f"{config.TOKEN_FREQ_FILE} missing - build it with "
                "`python -m compression.build_vocab_freq` (see README) or set EMB_SPLIT = False")
        z = np.load(config.TOKEN_FREQ_FILE)
        rows = cands[e][0]
        common = S.common_tokens(z["counts"], rows, config.EMB_COVERAGE, config.EMB_MIN_COMMON,
                                 config.EMB_MAX_COMMON, z["force_ids"])
        cnt = np.zeros(rows)
        cnt[:min(rows, len(z["counts"]))] = z["counts"][:rows]
        cov = float(cnt[common.numpy()].sum() / max(cnt.sum(), 1))
        parts[e] = [S.Part(f"{e}#c", e, 0, common, "c"), S.Part(f"{e}#r", e, 0, ~common, "r")]
        info["emb"] = dict(tensor=e, rows=rows, common=int(common.sum()), coverage=cov,
                           freq_sha256=hashlib.sha256(Path(config.TOKEN_FREQ_FILE).read_bytes())
                           .hexdigest(), counted_tokens=int(cnt.sum()))
        log(f"[embed] {int(common.sum()):,} common token rows (cover {cov * 100:.4f}% of "
            f"{int(cnt.sum()):,} counted tokens) + {rows - int(common.sum()):,} rare rows "
            f"(>= {config.RARE_MIN_BITS} bits/weight)")
    return parts, info


# ---------------------------------------------------------------------------
def convert_from_hf_checkpoint(model_name, checkpoint_path, output_path):
    from compression import codec, rans
    from compression.fisher import collect_fisher, load_model, map_params
    from compression.gptq_ec import Hessian, gptq_on_grid
    from compression.modelscan import (chunk_by_memory, find_decoder_layers, linear_targets,
                                       make_key_resolver)
    from compression.replay import capture_layer0_inputs, run_layer
    from decompression import rans as rans_dec

    t_all = time.time()
    base_dir = resolve_checkpoint_dir(checkpoint_path, model_name)
    out_dir = Path(output_path).expanduser()
    art_dir = out_dir.parent / f"{out_dir.name}_artifacts"
    art_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache_device = config.CACHE_DEVICE if torch.cuda.is_available() else "cpu"
    import transformers
    _log(f"[env] torch {torch.__version__}, transformers {transformers.__version__}, "
         f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'} | method "
         f"{config.METHOD}, target {config.TARGET_RATIO}"
         + ("   *** SMOKE MODE -- DO NOT SUBMIT ***" if config.SMOKE else ""))

    # 1. inventory -------------------------------------------------------------
    cands, raw, dropped, scored_bytes, fixed_bytes = inventory(base_dir)
    denom = config.GRADER_TEXT_GIB * 2**30
    n_w = sum(int(np.prod(s)) for s in cands.values())
    _log(f"[inventory] {len(cands)} coded tensors ({n_w / 1e9:.3f} B weights), {len(raw)} bf16 "
         f"({fixed_bytes / 2**20:.1f} MiB), {len(dropped)} dropped | scored text "
         f"{scored_bytes / 2**30:.4f} GiB")

    # 2. model + calibration ---------------------------------------------------------
    model, tok, n_gdn = load_model(base_dir, device)
    params, missing = map_params(model, cands)
    if missing:
        raise RuntimeError(f"{len(missing)} checkpoint tensors not in the model, e.g. {missing[:3]}")
    gptq_all, fisher_all, calib_sha = load_calib(config.CALIB_FILE)
    fisher_set = fisher_all[: config.FISHER_SAMPLES]
    fisher_set = [(s[:, : config.FISHER_MAX_TOKENS], p) for s, p in fisher_set
                  if min(s.shape[1], config.FISHER_MAX_TOKENS) - p >= 16]
    gptq_set, gptq_tok = cap_tokens(gptq_all, config.GPTQ_MAX_TOKENS)
    gdn_kernel = "torch"
    if config.USE_FLA:
        from compression.fisher import enable_fla_delta_rule
        _p_ids, _p_len = fisher_set[0]
        gdn_kernel = enable_fla_delta_rule(model, probe=(_p_ids[:, :2048].to(device), _p_len))
    _log(f"[calib] {config.CALIB_FILE.name} sha {calib_sha[:12]} | GPTQ {len(gptq_set)} seqs / "
         f"{gptq_tok:,} tokens (max {max(s.shape[1] for s in gptq_set):,}) | Fisher "
         f"{len(fisher_set)} seqs <= {config.FISHER_MAX_TOKENS} | {n_gdn} GDN layers, "
         f"delta-rule kernel: {gdn_kernel}")

    # 3. Fisher -> masks -> plan ---------------------------------------------------------
    names = list(cands)
    from compression.fisher import decision_mask, decision_token_sets
    dsets = decision_token_sets(tok)
    _pos_weight = None
    if config.DECISION_WEIGHT != 1.0:
        _frac = sum(int(decision_mask(s, p, dsets).sum()) for s, p in fisher_set) / max(
            sum(s.shape[1] - p for s, p in fisher_set), 1)
        _log(f"[fisher] decision-token weighting x{config.DECISION_WEIGHT}: {len(dsets[0])} "
             f"decision-word ids, {len(dsets[1])} boundary ids, {_frac * 100:.1f}% of Fisher "
             f"positions weighted")

        def _pos_weight(ids, start):
            return 1.0 + (config.DECISION_WEIGHT - 1.0) * decision_mask(ids, start, dsets).float()
    fisher = collect_fisher(model, params, fisher_set, device, log=_log, pos_weight=_pos_weight)
    parts, struct_info = build_parts(names, cands, params, fisher, _log)

    # virtual tensors: a whole coded tensor (vname == ckpt name) or one part of it
    part_of, vnames_of, vparams, vfisher, vshape = {}, {}, {}, {}, {}
    for n in names:
        if n in parts:
            vnames_of[n] = []
            for P in parts[n]:
                vparams[P.vname] = S.take(params[n].detach(), P.axis, P.keep)
                vfisher[P.vname] = S.take(fisher[n], P.axis, P.keep)
                vshape[P.vname] = tuple(vparams[P.vname].shape)
                part_of[P.vname] = P
                vnames_of[n].append(P.vname)
        else:
            vparams[n], vfisher[n], vshape[n] = params[n], fisher[n], tuple(cands[n])
            vnames_of[n] = [n]
    vnames = [v for n in names for v in vnames_of[n]]
    del fisher
    n_v = sum(int(np.prod(vshape[v])) for v in vnames)
    _log(f"[struct] {len(vnames)} virtual tensors ({len(part_of)} parts) | coded weights "
         f"{n_v / 1e9:.3f} B of {n_w / 1e9:.3f} B ({(1 - n_v / n_w) * 100:.2f}% removed)")

    groups = {v: codec.pick_group(vshape[v][1], config.GROUP_SIZE) for v in vnames}
    masks_idx = {v: codec.outlier_index(vparams[v], vfisher[v], groups[v], config.OUTLIER_FRAC,
                                        device) for v in vnames}
    masks = {v: codec.mask_from_index(masks_idx[v], vshape[v], "cpu") for v in vnames}
    kgrid = codec.kgrid_from(config.K_MIN, config.K_MAX, config.K_STEPS)
    tabs = codec.build_tables(vnames, vparams, vfisher, groups, masks, kgrid, device,
                              max_rows=config.TABLE_MAX_ROWS, log=_log)
    for v, P in part_of.items():
        if P.tag == "r":     # rare-token rows: never fewer than RARE_MIN_BITS bits/weight
            low = tabs[v]["bits"] / tabs[v]["numel"] < config.RARE_MIN_BITS
            if bool(low.all()):
                raise RuntimeError(f"{v}: no k in the grid gives >= {config.RARE_MIN_BITS} bits")
            tabs[v]["loss"][low] = float("inf")
    del vfisher
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    mask_bytes = sum((int(P.keep.numel()) + 7) // 8 for P in part_of.values())
    overhead = sum(rans.lanes_for(int(np.prod(vshape[v]))) * 8 + 1024 for v in vnames) + mask_bytes
    budget_bits = (config.TARGET_RATIO * denom - fixed_bytes - overhead) * 8
    _log(f"[plan] target {config.TARGET_RATIO} -> {budget_bits / n_v:.4f} bits/weight on average "
         f"over the kept weights ({budget_bits / n_w:.4f} over the original count)")

    def exact_rtn_bits(n, k):
        m = masks[n].to(device)
        q, _ = codec.rtn(vparams[n], groups[n], k, m, device)
        sym, nsym, _ = codec.to_symbols(q, m)
        return codec.stored_bits(sym, nsym, groups[n], int(m.sum()))

    budget = budget_bits
    for rnd in range(4):
        kidx, pred = codec.allocate(tabs, budget)
        planned = {n: exact_rtn_bits(n, kgrid[kidx[n]]) for n in vnames}
        exact = sum(planned.values())
        _log(f"  plan round {rnd + 1}: table {pred / 8 / 2**30:.4f} GiB, exact RTN "
             f"{exact / 8 / 2**30:.4f} GiB, budget {budget_bits / 8 / 2**30:.4f} GiB")
        if exact <= budget_bits:
            break
        budget -= (exact - budget_bits) * 1.05
    else:
        raise RuntimeError("could not meet the size budget after re-planning")
    kmap = {n: kgrid[kidx[n]] for n in vnames}
    # per-tensor correction from table estimate to exact bits, used when re-planning
    corr = {n: planned[n] / max(tabs[n]["bits"][kidx[n]].item(), 1.0) for n in vnames}

    # 4-5. quantize ------------------------------------------------------------------
    results = {}   # name -> dict(q, steps, outl, G, k, bits)

    def finish(n, q, w_hat, k):
        G = groups[n]
        m = masks[n].to(q.device)
        sym, nsym, qmin = codec.to_symbols(q, m)
        bits = codec.stored_bits(sym, nsym, G, int(m.sum()))
        steps = codec.group_steps(vparams[n].detach().float(), G, k).to(torch.float16).cpu()
        outl = w_hat.reshape(-1)[masks_idx[n].to(w_hat.device)].to(torch.float16).cpu()
        lo, hi = int(q.min()), int(q.max())
        qd = torch.int16 if -32768 <= lo and hi <= 32767 else torch.int32
        results[n] = {"q": q.to(qd).cpu(), "steps": steps, "outl": outl, "G": G, "k": k,
                      "bits": bits, "rel_mse": ((w_hat - vparams[n].float()).pow(2).sum()
                                                / vparams[n].float().pow(2).sum()).item()}
        return bits

    layers_attr, layers = find_decoder_layers(model)
    resolve = make_key_resolver(names, n_layers=len(layers))
    layer_targets = []
    for li, layer in enumerate(layers):
        tg = {}
        for mn, mod in linear_targets(layer, 64, config.MIN_NUMEL).items():
            key = resolve(layers_attr, li, mn)
            if key in vnames_of and len(vnames_of[key]) == 1:
                tg[mn] = (mod, key, vnames_of[key][0])
        layer_targets.append(tg)
    gptq_names = {v for tg in layer_targets for _, _, v in tg.values()}
    if config.METHOD != "gptq":
        gptq_names = set()
    other = [n for n in vnames if n not in gptq_names]
    _log(f"[quant] {len(gptq_names)} tensors by GPTQ, {len(other)} by RTN ({other[:2]}...)")

    with torch.no_grad():
        for n in other:
            q, w_hat = codec.rtn(vparams[n], groups[n], kmap[n], masks[n].to(device), device)
            finish(n, q, w_hat, kmap[n])
            P = part_of.get(n)
            if P is None:
                params[n].data.copy_(w_hat.to(params[n].dtype))
            else:   # parts of one tensor are written into the live parameter one by one
                S.put(params[P.base].data, w_hat, P.axis, P.keep)
            del q, w_hat
        # RTN-path tensors whose parts do not cover every row/column: zero the rest
        for n in names:
            if n in parts and all(P.vname in other for P in parts[n]):
                cov = torch.zeros_like(parts[n][0].keep)
                for P in parts[n]:
                    cov |= P.keep
                if not bool(cov.all()):
                    S.put(params[n].data, torch.zeros_like(S.take(params[n].data, parts[n][0].axis,
                                                                  ~cov)), parts[n][0].axis, ~cov)

    used_actual = sum(results[n]["bits"] for n in other)
    used_planned = sum(planned[n] for n in other)
    n_replans = 0
    if gptq_names:
        t0 = time.time()
        hidden, kws = capture_layer0_inputs(model, layers, gptq_set, device, cache_device)
        _log(f"[gptq] cached {len(hidden)} samples / {gptq_tok:,} tokens on {cache_device} "
             f"({time.time() - t0:.0f}s)")
        g_act = g_plan = 0.0
        for li, layer in enumerate(layers):
            tl = time.time()
            tg = layer_targets[li]
            for chunk in chunk_by_memory({mn: m for mn, (m, _, _) in tg.items()},
                                         int(config.HESSIAN_BUDGET_GB * 2**30)):
                hs, handles = {}, []
                for mn in chunk:
                    mod = tg[mn][0]
                    hs[mn] = Hessian(mod.in_features, device)
                    handles.append(mod.register_forward_hook(
                        lambda m, inp, out, _h=hs[mn]: _h.add(inp[0].detach())))
                run_layer(layer, hidden, kws, device, update=False)
                for h in handles:
                    h.remove()
                for mn in chunk:
                    mod, key, v = tg[mn]
                    G, k = groups[v], kmap[v]
                    P = part_of.get(v)
                    H = hs[mn].H
                    if P is None:
                        W = mod.weight
                    elif P.axis == 0:          # gate/up: drop the removed neurons' rows
                        W = vparams[v]
                    else:                      # down: fold removed columns into kept ones
                        W = S.compensate_columns(mod.weight.detach().float(), H, P.keep,
                                                 config.PERCDAMP)
                        vparams[v] = W         # steps (here and in finish) from the folded weights
                        kk = P.keep.to(H.device)
                        H = H[kk][:, kk].contiguous()
                    steps = codec.group_steps(W.detach().float(), G, k)
                    q, w_hat = gptq_on_grid(W, H, steps, G, masks[v].to(device), config.PERCDAMP,
                                            config.BLOCK_SIZE, config.ACT_ORDER)
                    b = finish(v, q, w_hat, k)
                    if P is None:
                        mod.weight.data.copy_(w_hat.to(mod.weight.dtype))
                    else:                      # removed rows/columns become exact zeros
                        full = torch.zeros_like(mod.weight.data)
                        mod.weight.data.copy_(S.put(full, w_hat, P.axis, P.keep))
                    g_act += b
                    g_plan += planned[v]
                    hs[mn] = None
                    del q, w_hat
                del hs
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            run_layer(layer, hidden, kws, device, update=True)

            # rate control: re-plan the layers still to come
            remaining = [v for tg2 in layer_targets[li + 1:] for _, _, v in tg2.values()]
            used_actual = sum(r["bits"] for r in results.values())
            drift = g_act / max(g_plan, 1.0) - 1.0
            if remaining and abs(drift) > config.RATE_TOLERANCE:
                infl = 1.0 + drift
                sub = {n: {"bits": tabs[n]["bits"] * corr[n], "loss": tabs[n]["loss"]}
                       for n in remaining}
                try:
                    kidx2, _ = codec.allocate(sub, budget_bits - used_actual, inflate=infl)
                except RuntimeError:
                    _log("  [rate] remaining budget too small; coarsest step for the rest "
                         "(the final size check still applies)")
                    kidx2 = {n: len(kgrid) - 1 for n in remaining}
                for n in remaining:
                    kmap[n] = kgrid[kidx2[n]]
                    planned[n] = sub[n]["bits"][kidx2[n]].item()
                n_replans += 1
            el = time.time() - t_all
            _log(f"[gptq] layer {li + 1:>2}/{len(layers)} ({time.time() - tl:4.0f}s) | bits used "
                 f"{used_actual / 8 / 2**30:.4f} GiB | GPTQ vs plan {drift * 100:+.2f}%"
                 + (" -> re-planned" if remaining and abs(drift) > config.RATE_TOLERANCE else "")
                 + f" | elapsed {el / 60:.1f} min")
        del hidden, kws

    del model, params, vparams
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. encode + write -------------------------------------------------------------
    _log("[write] rANS-encoding and writing shards ...")
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = ShardedSafetensorsWriter(out_dir, config.SHARD_PREFIX, config.MAX_SHARD_BYTES)
    metas, rows_out = {}, []
    t0 = time.time()
    for name, tensor in iter_tensors(base_dir):
        dt = str(tensor.dtype).replace("torch.", "")
        if name in dropped:
            metas[name] = {"mode": "zeros", "shape": list(tensor.shape), "dtype": dt}
            continue
        if name not in vnames_of:
            writer.add(name, tensor)
            metas[name] = {"mode": "raw", "shape": list(tensor.shape), "dtype": dt}
            continue
        pmetas = []
        for v in vnames_of[name]:
            R = results.pop(v)
            m = masks[v]
            sym, nsym, qmin = codec.to_symbols(R["q"], m)
            sym = sym.numpy()
            enc = rans.encode(sym, nsym)
            dec = rans_dec.decode(enc["words"], enc["state"], enc["count"], enc["freq"], sym.size)
            if not np.array_equal(dec, sym):
                raise RuntimeError(f"rANS round trip failed for {v}")
            writer.add(f"{v}::words", torch.from_numpy(enc["words"].view(np.int16)))
            writer.add(f"{v}::state", torch.from_numpy(enc["state"].view(np.int32)))
            cnt_signed = np.int16 if enc["count"].dtype == np.uint16 else np.int32
            writer.add(f"{v}::count", torch.from_numpy(enc["count"].view(cnt_signed)))
            writer.add(f"{v}::freq", torch.from_numpy(enc["freq"].view(np.int32)))
            writer.add(f"{v}::steps", R["steps"])
            if R["outl"].numel():
                writer.add(f"{v}::outliers", R["outl"])
            r, c = vshape[v]
            pm = {"rows": r, "cols": c, "group_size": R["G"], "k": R["k"], "q_min": qmin,
                  "escape": nsym - 1, "outliers": int(R["outl"].numel()),
                  "method": "gptq" if v in gptq_names else "rtn",
                  "stream_dtypes": {"words": "uint16", "state": "uint32", "freq": "uint32",
                                    "count": str(enc["count"].dtype)}}
            P = part_of.get(v)
            if P is not None:
                writer.add(f"{v}::keep", S.pack_mask(P.keep))
                pm.update({"name": v, "axis": P.axis, "keep_key": f"{v}::keep", "tag": P.tag,
                           "axis_len": int(P.keep.numel())})
            pmetas.append(pm)
            bits = enc["bytes"] * 8 + R["steps"].numel() * 16 + R["outl"].numel() * 16
            rows_out.append({"tensor": v, "k": R["k"], "group": R["G"], "bpw": bits / (r * c),
                             "rel_mse": R["rel_mse"], "method": pm["method"],
                             "shape": [r, c]})
            del sym, dec, enc, R
        if name in parts:
            metas[name] = {"mode": "ec_parts", "shape": list(tensor.shape), "dtype": dt,
                           "parts": pmetas}
        else:
            metas[name] = {"mode": "ec", "shape": list(tensor.shape), "dtype": dt, **pmetas[0]}
        if len(rows_out) % 25 == 0:
            _log(f"  encoded {len(rows_out)}/{len(vnames)} ({time.time() - t0:.0f}s)")
    writer.finalize()

    scored = measure_scored_bytes(out_dir)
    comp = {"format": FORMAT_VERSION, "model_name": model_name,
            "method": f"codecE-fisher-dtw{config.DECISION_WEIGHT:g}-{config.METHOD}-p3lite",
            "target_ratio": config.TARGET_RATIO, "decision_weight": config.DECISION_WEIGHT,
            "prune_frac": config.PRUNE_FRAC, "emb_split": config.EMB_SPLIT,
            "gdn_kernel": gdn_kernel,
            "outlier_frac": config.OUTLIER_FRAC, "group_size": config.GROUP_SIZE,
            "calib_sha256": calib_sha, "smoke": config.SMOKE,
            "drop_components": list(config.DROP_COMPONENTS), "compressed_text_bytes": scored,
            "original_text_bytes": scored_bytes, "tensors": metas}
    (out_dir / "compression_config.json").write_text(json.dumps(comp, indent=2))
    copied = copy_auxiliary_files(base_dir, out_dir)
    _log(f"[write] copied: {', '.join(copied)}")

    frac_g, frac_m = scored / denom, scored / scored_bytes
    (art_dir / "plan.json").write_text(json.dumps({"tensors": rows_out, "kgrid": kgrid}, indent=2))
    (art_dir / "structure.json").write_text(json.dumps(
        {k: v for k, v in struct_info.items() if k != "prune_keep"}, indent=2))
    if "prune_keep" in struct_info:
        np.savez_compressed(art_dir / "prune_keep.npz", **{f"layer{li}": k.numpy() for li, k in
                                                           struct_info["prune_keep"].items()})
    (art_dir / "size_report.json").write_text(json.dumps({
        "compressed_text_bytes": scored, "size_frac_grader": frac_g, "size_frac_measured": frac_m,
        "fixed_bytes": fixed_bytes, "replans": n_replans, "gptq_tokens": gptq_tok,
        "fisher_samples": len(fisher_set), "minutes": (time.time() - t_all) / 60,
        "smoke": config.SMOKE}, indent=2))
    ok = max(frac_g, frac_m) <= config.HARD_LIMIT
    bpw = [r["bpw"] for r in rows_out]
    _log(f"\n[compress] done in {(time.time() - t_all) / 60:.1f} min | bits/weight min "
         f"{min(bpw):.2f} median {sorted(bpw)[len(bpw) // 2]:.2f} max {max(bpw):.2f} | "
         f"re-plans {n_replans}")
    _log(f"[compress] SIZE_FRAC = {frac_g:.4f} (graders' 8.0585 GiB) / {frac_m:.4f} (measured) "
         + (f"-- within {config.HARD_LIMIT}" if ok else f"-- *** OVER {config.HARD_LIMIT} ***"))
    if not ok:
        raise SystemExit(2)
    return comp


def main():
    p = argparse.ArgumentParser(description="Compress a Hugging Face checkpoint.")
    p.add_argument("--model_name", "--model-name", dest="model_name", required=True)
    p.add_argument("--checkpoint_path", "--checkpoint-path", dest="checkpoint_path", required=True)
    p.add_argument("--output_path", "--output-path", dest="output_path", required=True)
    a = p.parse_args()
    convert_from_hf_checkpoint(a.model_name, a.checkpoint_path, a.output_path)


if __name__ == "__main__":
    main()
