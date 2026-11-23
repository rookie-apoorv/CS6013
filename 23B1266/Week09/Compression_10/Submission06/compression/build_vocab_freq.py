"""Build ``calib_data/token_freq.npz`` (token counts for the split embedding). Run once.

    python -m compression.build_vocab_freq --tokenizer /path/to/Qwen3.5-4B \
        --traces_repo <user>/cs6013-qwen35-4b-base-traces \
        --traces_repo <user>/cs6013-qwen35-4b-train-traces

Counts every prompt and completion token id in ``raw/<module>/*.parquet`` of
each repo (the base model's own maths traces, grader settings). Not used by
compress.py at run time except through this file, which is committed with the
code so compression needs no network and is reproducible.

Stored: ``counts`` (int64 [vocab]), ``force_ids`` (special / control tokens and
single printable-ASCII tokens, always treated as common), and provenance.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

COLS = ("prompt_token_ids", "completion_token_ids")


def count_file(path, counts):
    import pyarrow.parquet as pq
    t = pq.read_table(str(path), columns=[c for c in COLS])
    n = 0
    for c in COLS:
        arr = t.column(c).combine_chunks().flatten().to_numpy(zero_copy_only=False)
        if arr.size:
            arr = arr.astype(np.int64)
            hi = int(arr.max()) + 1
            if hi > counts.size:
                counts = np.concatenate([counts, np.zeros(hi - counts.size, np.int64)])
            counts += np.bincount(arr, minlength=counts.size)[: counts.size]
            n += int(arr.size)
    return counts, n


def force_ids_for(tok):
    ids = set(int(i) for i in (tok.all_special_ids or []))
    for s in ("<think>", "</think>", "<|im_start|>", "<|im_end|>", "<|endoftext|>"):
        i = tok.convert_tokens_to_ids(s)
        if isinstance(i, int) and i >= 0 and i != tok.unk_token_id:
            ids.add(i)
    for ch in (chr(c) for c in range(32, 127)):
        enc = tok(ch, add_special_tokens=False).input_ids
        if len(enc) == 1:
            ids.add(int(enc[0]))
    return sorted(ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces_repo", action="append", default=[])
    ap.add_argument("--local_dir", action="append", default=[],
                    help="local copy of a traces repo (raw/<module>/*.parquet)")
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--token", default=None)
    ap.add_argument("--out", default=str(Path(__file__).parent / "calib_data" / "token_freq.npz"))
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer)
    counts = np.zeros(len(tok), dtype=np.int64)
    files, total = [], 0
    for repo in a.traces_repo:
        from huggingface_hub import HfApi, hf_hub_download
        names = [f for f in HfApi(token=a.token).list_repo_files(repo, repo_type="dataset")
                 if f.startswith("raw/") and f.endswith(".parquet")]
        for f in sorted(names):
            p = hf_hub_download(repo, f, repo_type="dataset", token=a.token)
            counts, n = count_file(p, counts)
            files.append(f"{repo}:{f}")
            total += n
            print(f"  {repo}:{f}: {n:,} tokens", flush=True)
    for d in a.local_dir:
        for p in sorted(Path(d).glob("raw/*/*.parquet")):
            counts, n = count_file(p, counts)
            files.append(str(p))
            total += n
            print(f"  {p}: {n:,} tokens", flush=True)
    if total == 0:
        raise SystemExit("no tokens counted - check the repos")
    force = np.array(force_ids_for(tok), dtype=np.int64)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, counts=counts, force_ids=force)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    srt = np.sort(counts)[::-1]
    cum = np.cumsum(srt) / max(srt.sum(), 1)
    cover = {str(q): int(np.searchsorted(cum, q) + 1) for q in (0.99, 0.999, 0.9999, 0.99999)}
    info = dict(files=files, tokens=int(total), distinct=int((counts > 0).sum()),
                vocab=int(counts.size), force_ids=int(force.size), tokens_for_coverage=cover,
                sha256=sha)
    out.with_suffix(".json").write_text(json.dumps(info, indent=2))
    print(json.dumps({k: v for k, v in info.items() if k != "files"}, indent=2))


if __name__ == "__main__":
    main()
