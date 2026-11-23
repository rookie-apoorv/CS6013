"""P3-lite structure: MLP neuron pruning and the split-precision embedding.

A coded checkpoint tensor is stored either whole (as in week 8) or as *parts*:
row or column subsets selected by a boolean mask over one axis. Every part is
an ordinary codec-E tensor with its own step multiplier k, so the planner,
GPTQ and the rANS writer treat parts exactly like tensors ("virtual tensors",
named ``<ckpt name>#<tag>``). Positions not covered by any part are restored
as zeros.

* MLP neuron j removed -> gate/up keep rows ~j (axis 0, one part ``#k``),
  down keeps columns ~j (axis 1, one part ``#k``).
* Embedding -> two parts on axis 0: ``#c`` (common tokens) and ``#r`` (rest).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import torch

_MLP_RE = re.compile(r"^(?P<pre>.*\.layers\.(?P<idx>\d+)\.mlp\.)(?P<kind>gate_proj|up_proj|down_proj)"
                     r"\.weight$")


@dataclass
class Part:
    vname: str          # virtual tensor name
    base: str           # checkpoint tensor name
    axis: int           # 0 = rows, 1 = columns
    keep: torch.Tensor  # bool [shape[axis]] on CPU
    tag: str


def mlp_triples(names):
    """{layer index: {"gate": name, "up": name, "down": name}} for complete triples."""
    out = {}
    for n in names:
        m = _MLP_RE.match(n)
        if m:
            out.setdefault((m["pre"], int(m["idx"])), {})[m["kind"].split("_")[0]] = n
    return {li: t for (pre, li), t in sorted(out.items(), key=lambda x: x[0][1])
            if set(t) == {"gate", "up", "down"}}


@torch.no_grad()
def neuron_saliency(W_down, F_down):
    """OBD loss increase of removing each neuron: 0.5 * sum_i F[i,j] * W[i,j]^2."""
    return 0.5 * (F_down.float() * W_down.float().pow(2)).sum(0).cpu()


def select_prune(saliency, frac, granule, layer_cap):
    """saliency: {layer: 1-D tensor}. Removes granules of `granule` neurons
    (lowest saliency first inside a layer), greedily across layers by the
    summed saliency of the next granule, until >= frac of all neurons are
    removed. Returns {layer: bool keep mask}."""
    total = sum(int(s.numel()) for s in saliency.values())
    target = int(round(frac * total))
    order = {l: torch.argsort(s) for l, s in saliency.items()}
    sorted_s = {l: s[order[l]].double() for l, s in saliency.items()}
    removed = {l: 0 for l in saliency}
    n_removed = 0
    while n_removed < target:
        best, best_cost = None, None
        for l, s in sorted_s.items():
            n = s.numel()
            a = removed[l]
            b = a + granule
            if b > int(layer_cap * n) or b >= n:
                continue
            cost = float(s[a:b].sum())
            if best_cost is None or cost < best_cost:
                best, best_cost = l, cost
        if best is None:
            break
        removed[best] += granule
        n_removed += granule
    keep = {}
    for l, s in saliency.items():
        k = torch.ones(s.numel(), dtype=torch.bool)
        k[order[l][: removed[l]]] = False
        keep[l] = k
    return keep, n_removed, total


def common_tokens(counts, n_rows, coverage, n_min, n_max, force_ids=()):
    """Bool mask over embedding rows: the most frequent tokens covering
    `coverage` of all counted tokens (clamped to [n_min, n_max]) + force_ids."""
    c = np.zeros(n_rows, dtype=np.float64)
    m = min(n_rows, len(counts))
    c[:m] = np.asarray(counts[:m], dtype=np.float64)
    order = np.argsort(-c, kind="stable")
    cum = np.cumsum(c[order])
    tot = cum[-1] if len(cum) and cum[-1] > 0 else 1.0
    n = int(np.searchsorted(cum / tot, coverage) + 1)
    n = max(n_min, min(n_max, n, n_rows))
    mask = np.zeros(n_rows, dtype=bool)
    mask[order[:n]] = True
    f = np.asarray([i for i in force_ids if 0 <= int(i) < n_rows], dtype=np.int64)
    if f.size:
        mask[f] = True
    return torch.from_numpy(mask)


def take(t, axis, keep):
    """Slice rows (axis 0) or columns (axis 1) of a 2-D tensor by a bool mask."""
    k = keep.to(t.device)
    return (t[k] if axis == 0 else t[:, k]).contiguous()


def put(full, part, axis, keep):
    """Write `part` into `full` (in place) at the kept rows / columns."""
    k = keep.to(full.device)
    if axis == 0:
        full[k] = part.to(full.dtype)
    else:
        full[:, k] = part.to(full.dtype)
    return full


@torch.no_grad()
def compensate_columns(W, H, keep, percdamp=0.01):
    """Least-squares fold of removed input columns into the kept ones:
    argmin_V || X_K V^T - X W^T ||  =>  V = W_K + W_R H_RK H_KK^-1,
    with H = X^T X (here the GPTQ Hessian). W [out, in] fp32, H [in, in] fp32,
    keep bool [in]. Returns V [out, |K|] fp32."""
    dev = H.device
    W = W.to(dev, torch.float32)
    k = keep.to(dev)
    r = ~k
    if not bool(r.any()):
        return W[:, k].clone()
    Hkk = H[k][:, k].clone()
    Hkr = H[k][:, r]
    idx = torch.arange(Hkk.shape[0], device=dev)
    dead = torch.diag(Hkk) == 0
    Hkk[dead, dead] = 1.0
    Hkk[idx, idx] += percdamp * torch.mean(torch.diag(Hkk))
    B = torch.linalg.solve(Hkk, Hkr)            # [K, R] = H_KK^-1 H_KR
    return W[:, k] + W[:, r] @ B.T


def pack_mask(keep):
    return torch.from_numpy(np.packbits(keep.numpy().astype(np.uint8)))


def unpack_mask(packed, n):
    return np.unpackbits(np.asarray(packed, dtype=np.uint8))[:n].astype(bool)
