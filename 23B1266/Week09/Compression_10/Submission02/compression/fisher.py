"""Model loading and the diagonal-Fisher pass.

Fisher ``F_ij = E[(dL/dw_ij)^2]`` of the trace NLL, one value per weight. The
planner uses ``0.5 * sum F * dW^2`` as the predicted loss increase of a
quantization choice; in codec_lab2 this proxy ranked 8 configurations in the
same order as the measured KL.

No flash kernels: Gated DeltaNet is pinned to transformers' PyTorch path and
full attention uses torch SDPA.
"""

from __future__ import annotations

import re
import time

import torch


def force_torch_kernels(model):
    n = 0
    for m in model.modules():
        if type(m).__name__.endswith("GatedDeltaNet"):
            mod = __import__(type(m).__module__, fromlist=["x"])
            m.causal_conv1d_fn = None
            m.chunk_gated_delta_rule = mod.torch_chunk_gated_delta_rule
            if hasattr(mod, "torch_recurrent_gated_delta_rule"):
                m.recurrent_gated_delta_rule = mod.torch_recurrent_gated_delta_rule
            if hasattr(mod, "torch_causal_conv1d_update"):
                m.causal_conv1d_update = mod.torch_causal_conv1d_update
            n += 1
    return n


def torch_gated_norms(model):
    """Use transformers' PyTorch gated RMSNorm in every Gated DeltaNet layer even if
    flash-linear-attention is installed (its fused Triton norm is not needed)."""
    n = 0
    for m in model.modules():
        if not type(m).__name__.endswith("GatedDeltaNet"):
            continue
        mod = __import__(type(m).__module__, fromlist=["x"])
        cls = next((getattr(mod, k) for k in dir(mod) if k.endswith("RMSNormGated")
                    and not k.startswith("Fused")), None)
        if cls is not None and not isinstance(m.norm, cls):
            old = m.norm
            new = cls(old.weight.shape[0], eps=getattr(old, "eps", 1e-6))
            with torch.no_grad():
                new.weight.copy_(old.weight)
            new.weight.requires_grad_(old.weight.requires_grad)
            m.norm = new.to(device=old.weight.device, dtype=old.weight.dtype)
            n += 1
    return n


@torch.no_grad()
def _probe(model, ids, plen, first_gdn):
    """(completion NLL, output of the first Gated DeltaNet layer) for one sequence."""
    got = {}
    h = first_gdn.register_forward_hook(
        lambda m, i, o: got.__setitem__("y", (o[0] if isinstance(o, tuple) else o).float()))
    try:
        logits = model(input_ids=ids, use_cache=False).logits[0, plen - 1:-1].float()
    finally:
        h.remove()
    nll = torch.nn.functional.cross_entropy(logits, ids[0, plen:]).item()
    return nll, got["y"]


def enable_fla_delta_rule(model, probe=None, tol_layer=2e-2, tol_nll=5e-3):
    """Switch ONLY the chunked delta rule of every Gated DeltaNet layer to
    flash-linear-attention's Triton kernel (conv + gated norm stay PyTorch).
    `probe` = (ids [1, L], prompt_len): one forward pass with each kernel; fla is
    kept only if the first Gated DeltaNet layer's output (identical input under
    both kernels) agrees to `tol_layer` (relative L2) and the completion NLL to
    `tol_nll` (relative). Returns a status string; on any problem the PyTorch
    kernel stays in place."""
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule as fla_fn
    except Exception as exc:
        return f"torch (fla not importable: {type(exc).__name__}: {str(exc)[:160]})"
    torch_gated_norms(model)
    gdn = [m for m in model.modules() if type(m).__name__.endswith("GatedDeltaNet")]
    if not gdn:
        return "torch (no Gated DeltaNet layers)"
    torch_fn = [m.chunk_gated_delta_rule for m in gdn]
    try:
        if probe is not None:
            ids, plen = probe
            nll_t, y_t = _probe(model, ids, plen, gdn[0])
            for m in gdn:
                m.chunk_gated_delta_rule = fla_fn
            nll_f, y_f = _probe(model, ids, plen, gdn[0])
            e_layer = float((y_f - y_t).norm() / y_t.norm().clamp_min(1e-12))
            e_nll = abs(nll_f - nll_t) / max(abs(nll_t), 1e-6)
            if not (e_layer <= tol_layer and e_nll <= tol_nll and nll_f == nll_f):
                raise ValueError(f"layer-0 output rel err {e_layer:.2e}, NLL torch {nll_t:.5f} vs "
                                 f"fla {nll_f:.5f} (rel {e_nll:.2e})")
            status = (f"fla (probe: layer rel err {e_layer:.1e}, NLL {nll_t:.5f} / {nll_f:.5f})")
        else:
            status = "fla (no probe)"
        for m in gdn:
            m.chunk_gated_delta_rule = fla_fn
        return status
    except Exception as exc:
        for m, f in zip(gdn, torch_fn):
            m.chunk_gated_delta_rule = f
        return f"torch (fla rejected: {type(exc).__name__}: {str(exc)[:200]})"


def load_model(base_dir, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=torch.bfloat16,
                                                 low_cpu_mem_usage=True,
                                                 attn_implementation="sdpa")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    model.config.use_cache = False
    n = force_torch_kernels(model)
    model.to(device)
    tok = AutoTokenizer.from_pretrained(str(base_dir))
    return model, tok, n


_TAIL = re.compile(r"((?:layers\.\d+\..*)|(?:embed_tokens\.weight)|(?:lm_head\.weight))$")


def tail(name):
    m = _TAIL.search(name)
    return m.group(1) if m else name


def map_params(model, ckpt_names):
    """ckpt name -> live model parameter (matched on the part after the prefix)."""
    by_tail = {}
    for n, p in model.named_parameters():
        by_tail.setdefault(tail(n), p)
    out, missing = {}, []
    for c in ckpt_names:
        p = by_tail.get(tail(c))
        if p is None or tuple(p.shape) != tuple(ckpt_names[c]):
            missing.append(c)
        else:
            out[c] = p
    return out, missing


DECISION_WORDS = ("Wait", "But", "Alternatively", "Hmm", "Actually", "However", "Hold")


def decision_token_sets(tok):
    """Token ids of decision words (first sub-token of each spelling) and of
    sentence/paragraph boundary tokens (decoded text contains a newline or ends
    in '.', '?', '!' or ':'), plus the </think> id."""
    words = set()
    for w in DECISION_WORDS:
        for v in (w, " " + w, w.lower(), " " + w.lower()):
            ids = tok(v, add_special_tokens=False).input_ids
            if ids:
                words.add(int(ids[0]))
    think_end = tok.convert_tokens_to_ids("</think>")
    if isinstance(think_end, int) and think_end >= 0:
        words.add(think_end)
    boundary = set()
    for i in range(len(tok)):
        t = tok.decode([i])
        if "\n" in t or t.rstrip().endswith((".", "?", "!", ":")):
            boundary.add(i)
    return torch.tensor(sorted(words)), torch.tensor(sorted(boundary))


def decision_mask(ids, start, sets):
    """Bool mask over predicted positions ids[0, start:]: the target is a decision
    word, or the token before it ends a sentence/paragraph (where the model
    chooses between continuing and second-guessing)."""
    words, boundary = sets
    tgt = ids[0, start:]
    prev = ids[0, start - 1:-1]
    return torch.isin(tgt, words.to(ids.device)) | torch.isin(prev, boundary.to(ids.device))


def collect_fisher(model, params, samples, device, log=print, pos_weight=None):
    """params: ckpt name -> parameter. Returns ckpt name -> bf16 Fisher tensor.

    pos_weight: optional callable (ids, start) -> float weights over the predicted
    positions; the per-sample loss becomes a weighted mean of token NLLs."""
    live = {id(p) for p in params.values()}
    for p in model.parameters():
        p.requires_grad_(id(p) in live)
    try:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    except Exception as exc:
        log(f"  (gradient checkpointing unavailable: {exc!r})")
    model.train()
    acc = {c: torch.zeros_like(p, dtype=torch.float32) for c, p in params.items()}
    t0 = time.time()
    try:
        for i, (ids, start) in enumerate(samples):
            ids = ids.to(device)
            logits = model(input_ids=ids, use_cache=False).logits[0, start - 1:-1].float()
            if pos_weight is None:
                loss = torch.nn.functional.cross_entropy(logits, ids[0, start:])
            else:
                w = pos_weight(ids, start).to(logits.device, torch.float32)
                nll = torch.nn.functional.cross_entropy(logits, ids[0, start:], reduction="none")
                loss = (w * nll).sum() / w.sum()
            loss.backward()
            for c, p in params.items():
                if p.grad is not None:
                    acc[c].add_(p.grad.float().pow(2))
            for p in params.values():
                p.grad = None
            del logits, loss
            if (i + 1) % 8 == 0 or i + 1 == len(samples):
                log(f"  [fisher] {i + 1}/{len(samples)} samples ({time.time() - t0:.0f}s)")
    finally:
        for p in model.parameters():
            p.requires_grad_(False)
            p.grad = None
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        model.eval()
    out = {}
    for c in list(acc):
        out[c] = (acc.pop(c) / max(len(samples), 1)).to(torch.bfloat16)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out
