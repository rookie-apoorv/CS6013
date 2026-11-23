"""Inverse of codec E (see compression/codec.py). numpy + torch only, CPU.

Format v2 adds ``mode == "ec_parts"``: a tensor stored as row or column subsets
(each an ordinary codec-E stream with its own step multiplier), selected by a
packed boolean mask over one axis. Positions no part covers are zeros (pruned
MLP neurons); the embedding's two parts (common / rare token rows) cover all rows.
"""

from __future__ import annotations

import numpy as np
import torch

from decompression.rans import decode

ROW_CHUNK = 8192


def restore_tensor(store, name: str, meta: dict) -> torch.Tensor:
    rows, cols = int(meta["rows"]), int(meta["cols"])
    G = int(meta["group_size"])
    n = rows * cols
    # streams are stored as signed int views of unsigned data; reinterpret
    dts = meta["stream_dtypes"]
    sym = decode(
        store.get(f"{name}::words").numpy().view(np.dtype(dts["words"])),
        store.get(f"{name}::state").numpy().view(np.dtype(dts["state"])),
        store.get(f"{name}::count").numpy().view(np.dtype(dts["count"])),
        store.get(f"{name}::freq").numpy().view(np.dtype(dts["freq"])),
        n,
    )
    esc = int(meta["escape"])
    qmin = int(meta["q_min"])
    is_esc = sym == esc
    sym2d = torch.from_numpy(sym).reshape(rows, cols)
    steps = store.get(f"{name}::steps").float()          # [rows, cols // G]
    out = torch.empty((rows, cols), dtype=torch.float32)
    for r0 in range(0, rows, ROW_CHUNK):
        r1 = min(r0 + ROW_CHUNK, rows)
        out[r0:r1] = (sym2d[r0:r1].float() + qmin) * steps[r0:r1].repeat_interleave(G, 1)
    n_out = int(meta.get("outliers", 0))
    if n_out:
        vals = store.get(f"{name}::outliers").float()
        if vals.numel() != n_out or int(is_esc.sum()) != n_out:
            raise ValueError(f"{name}: outlier count mismatch")
        out.view(-1)[torch.from_numpy(np.flatnonzero(is_esc))] = vals
    dtype = getattr(torch, meta["dtype"])
    return out.reshape(tuple(meta["shape"])).to(dtype).contiguous()


def restore_parts(store, meta: dict) -> torch.Tensor:
    shape = tuple(meta["shape"])
    out = torch.zeros(shape, dtype=torch.float32)
    for pm in meta["parts"]:
        n_axis = int(pm["axis_len"])
        if n_axis != shape[pm["axis"]]:
            raise ValueError(f"{pm['name']}: mask length {n_axis} != shape {shape}")
        keep = np.unpackbits(store.get(pm["keep_key"]).numpy().view(np.uint8))[:n_axis].astype(bool)
        sub = restore_tensor(store, pm["name"], {**pm, "shape": [pm["rows"], pm["cols"]],
                                                 "dtype": "float32"})
        k = torch.from_numpy(keep)
        if int(k.sum()) != sub.shape[pm["axis"]]:
            raise ValueError(f"{pm['name']}: mask selects {int(k.sum())}, part has "
                             f"{sub.shape[pm['axis']]}")
        if pm["axis"] == 0:
            out[k] = sub
        else:
            out[:, k] = sub
    return out.to(getattr(torch, meta["dtype"])).contiguous()
