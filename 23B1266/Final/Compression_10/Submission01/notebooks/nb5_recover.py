import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="CS6013 nb5 - post-compression recovery (distillation)")

with app.setup:
    import importlib
    import json
    import os
    import re
    import shutil
    import struct
    import subprocess
    import sys
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo

    NB_VERSION = "nb5-v1.0 (2026-11-23)"
    WORK = Path("/root/work/nb5") if Path("/root").exists() else Path.home() / "work" / "nb5"
    WORK.mkdir(parents=True, exist_ok=True)
    PY = sys.executable
    GRADER_TEXT_GIB = 8.0585
    STATE = {"R": None, "code_dir": None, "comp_dir": None, "teacher_dir": None,
             "restored0": None, "student": None, "teacher": None, "codec": None, "raw": None,
             "train": None, "held": None, "kl_before": None, "device": None, "out_dir": None,
             "verified": False}

    def run_streaming(cmd, cwd=None, env=None, check=True):
        merged = {**os.environ, **(env or {})}
        proc = subprocess.Popen([str(c) for c in cmd], cwd=str(cwd) if cwd else None, env=merged,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                bufsize=1)
        for line in proc.stdout:
            print(line, end="")
        proc.wait()
        if check and proc.returncode != 0:
            raise RuntimeError(f"failed ({proc.returncode}): {' '.join(map(str, cmd))}")
        return proc.returncode

    def scored_gib(directory):
        """Grader-style text bytes from the safetensors headers."""
        total = 0
        for f in sorted(Path(directory).glob("*.safetensors")):
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                hdr = json.loads(fh.read(n))
            for k, v in hdr.items():
                if k == "__metadata__":
                    continue
                kk = k.lower()
                if "visual." in kk or any(p in {"visual", "vision", "vision_tower", "vision_model"}
                                          for p in kk.split(".")):
                    continue
                total += v["data_offsets"][1] - v["data_offsets"][0]
        return total / 2**30


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CS6013 · Notebook 5 — post-compression recovery (size-neutral distillation) · v1.0

    Takes a **compressed submission** (codec-E format, week-8 20% or P3-lite 10%) and trains
    only what is stored as plain floats — **every per-group step** (as a multiplier, starts at
    ×1), **every outlier value**, **every raw bf16 tensor** (norms, gates, conv) — while **all
    integer codes stay fixed**. Objective: KL(teacher ‖ student) per token, teacher = the
    uncompressed fine-tuned bf16 model, on the SFT training traces cut to 8K tokens.
    The rewritten checkpoint has **byte-identical size** and is restored by the **unchanged**
    `decompress.py`.

    | step | what |
    |---|---|
    | ① | inputs |
    | ② | environment (+ optional flash-linear-attention) |
    | ③ | clone the submission code (needs its `recovery/` folder) |
    | ④ | download compressed checkpoint + teacher; decompress (= the student's start) |
    | ⑤ | build student (bit-exactness check), teacher, data; **KL before** on held-out traces |
    | ⑥ | smoke: time of one training sequence → time estimate |
    | ⑦ | **train** (resumable; checkpoint to HF every N min) |
    | ⑧ | export → size check → decompress → exactness → **KL after** + greedy sample |
    | ⑨ | upload the recovered checkpoint (only if ⑧ says it is better) |

    Held-out KL uses the base model's traces of the **held-out problems** (AIME 2025/26 and
    HMMT Feb 2025; not in the training set or the calibration). Keep this tab active during ⑦.
    """)
    return


@app.cell(hide_code=True)
def _():
    hf_token = mo.ui.text(kind="password", label="HF token (write)", full_width=True)
    gh_token = mo.ui.text(kind="password", label="GitHub token (if the repo is private)")
    code_url = mo.ui.text(placeholder="https://github.com/<you>/CS6013/tree/main/<path>/<submission folder>",
                          label="Code: GitHub tree URL of the submission folder (with recovery/)",
                          full_width=True)
    comp_repo = mo.ui.text(placeholder="grey-cat/<roll>-WeekNN-Track1-20-SubmissionNN",
                           label="HF repo of the compressed checkpoint to recover", full_width=True)
    teacher_repo = mo.ui.text(value="grey-cat/cs6013-qwen35-4b-sft-v1",
                              label="teacher (uncompressed model that was compressed)", full_width=True)
    train_set = mo.ui.text(value="grey-cat/cs6013-sft-artifacts:runs/sft-v1/data/train_set.parquet",
                           label="training set (<hf dataset repo>:<path>)", full_width=True)
    traces_repo = mo.ui.text(value="grey-cat/cs6013-qwen35-4b-base-traces",
                             label="nb1 traces repo (held-out KL check)", full_width=True)
    heldout = mo.ui.text(value="aime_2025,aime_2026,hmmt_feb_2025", label="held-out sources")
    run_name = mo.ui.text(value="rec20-v1", label="run name")
    art_repo = mo.ui.text(value="grey-cat/cs6013-sft-artifacts",
                          label="HF dataset repo for checkpoints / logs (runs/<run name>/...)",
                          full_width=True)
    out_repo = mo.ui.text(placeholder="grey-cat/<roll>-WeekNN-Track1-20-SubmissionNN",
                          label="HF model repo for the recovered checkpoint (new)", full_width=True)
    mo.vstack([mo.md("## ① Inputs"), hf_token, gh_token, code_url, comp_repo, teacher_repo, train_set,
               mo.hstack([traces_repo, heldout], justify="start"),
               mo.hstack([run_name], justify="start"), art_repo, out_repo])
    return (art_repo, code_url, comp_repo, gh_token, heldout, hf_token, out_repo, run_name,
            teacher_repo, traces_repo, train_set)


@app.cell(hide_code=True)
def _(art_repo, hf_token, run_name):
    def tok():
        return hf_token.value.strip() or None

    def run_dir():
        d = WORK / "runs" / run_name.value.strip()
        d.mkdir(parents=True, exist_ok=True)
        return d

    def push_run(subdir, message):
        from huggingface_hub import HfApi
        repo = art_repo.value.strip()
        if not repo or not tok():
            print("  [hf] no artifacts repo / token - kept local")
            return
        api = HfApi(token=tok())
        api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
        for _i in range(3):
            try:
                api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=str(run_dir() / subdir),
                                  path_in_repo=f"runs/{run_name.value.strip()}/{subdir}",
                                  commit_message=message)
                print(f"  [hf] pushed runs/{run_name.value.strip()}/{subdir}")
                return
            except Exception as exc:
                print(f"  [hf] push failed ({_i + 1}/3): {type(exc).__name__}: {exc}")
                time.sleep(20)

    def pull_run(subdir):
        from huggingface_hub import HfApi, snapshot_download
        repo = art_repo.value.strip()
        prefix = f"runs/{run_name.value.strip()}/{subdir}/"
        try:
            files = [f for f in HfApi(token=tok()).list_repo_files(repo, repo_type="dataset")
                     if f.startswith(prefix)]
        except Exception:
            files = []
        if not files:
            return 0
        tmp = WORK / "pull"
        snapshot_download(repo, repo_type="dataset", token=tok(), local_dir=str(tmp),
                          allow_patterns=[prefix + "*"])
        dst = run_dir() / subdir
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(tmp / prefix, dst)
        return len(files)

    return pull_run, push_run, run_dir, tok


@app.cell(hide_code=True)
def _():
    env_btn = mo.ui.run_button(label="② Environment")
    want_fla = mo.ui.checkbox(value=True, label="install flash-linear-attention (--no-deps; ~4x faster training)")
    mo.vstack([mo.md("## ② Environment"), want_fla, env_btn])
    return env_btn, want_fla


@app.cell(hide_code=True)
def _(env_btn, want_fla):
    mo.stop(not env_btn.value, mo.md("*press ②*"))
    _lines = []
    if want_fla.value:
        _r = subprocess.run([PY, "-m", "pip", "install", "-q", "--no-deps", "flash-linear-attention",
                             "fla-core", "einops"], capture_output=True, text=True)
        _ok = subprocess.run([PY, "-c", "from fla.ops.gated_delta_rule import chunk_gated_delta_rule"],
                             capture_output=True, text=True)
        _lines.append("- flash-linear-attention (+ fla-core, einops): "
                      + ("importable" if _ok.returncode == 0 else
                         f"NOT importable (fine, PyTorch path): {_ok.stderr.strip()[-200:]}"))
    import torch as _t
    import transformers as _tf
    _lines.append(f"- torch `{_t.__version__}`, transformers `{_tf.__version__}`")
    if _t.cuda.is_available():
        _p = _t.cuda.get_device_properties(0)
        _lines.append(f"- GPU **{_p.name}**, {_p.total_memory / 2**30:.0f} GiB")
    else:
        _lines.append("- **NO GPU** - attach one")
    _lines.append(f"- disk free {shutil.disk_usage(WORK).free / 2**30:.0f} GiB (needs ~30 GiB)")
    mo.md("\n".join(_lines))
    return


@app.cell(hide_code=True)
def _():
    clone_btn = mo.ui.run_button(label="③ Clone code")
    clone_btn
    return (clone_btn,)


@app.cell(hide_code=True)
def _(clone_btn, code_url, gh_token):
    mo.stop(not clone_btn.value, mo.md("*press ③*"))
    _m = re.match(r"^https?://github\.com/([^/]+)/([^/]+?)(?:\.git)?/tree/([^/]+)/(.+)$",
                  code_url.value.strip().rstrip("/"))
    mo.stop(not _m, mo.md("**Code URL must look like https://github.com/<user>/<repo>/tree/<branch>/<path>**"))
    _owner, _repo, _branch, _sub = _m.groups()
    _url = f"https://github.com/{_owner}/{_repo}.git"
    if gh_token.value:
        _url = _url.replace("https://", f"https://oauth2:{gh_token.value}@", 1)
    _rd = WORK / "code_repo"
    shutil.rmtree(_rd, ignore_errors=True)
    run_streaming(["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", "--no-checkout",
                   "--branch", _branch, _url, _rd])
    run_streaming(["git", "-C", _rd, "sparse-checkout", "set", "--cone", _sub])
    run_streaming(["git", "-C", _rd, "checkout", _branch])
    _cd = _rd / _sub
    _need = ["decompress.py", "decompression/__init__.py", "decompression/rans.py",
             "decompression/io_utils.py", "recovery/__init__.py", "recovery/recover.py"]
    _miss = [f for f in _need if not (_cd / f).is_file()]
    mo.stop(bool(_miss), mo.md(f"**missing in the folder:** {_miss} - push the `recovery/` folder too"))
    for _k in [k for k in sys.modules if k == "decompression" or k.startswith(("decompression.", "recovery"))]:
        del sys.modules[_k]
    if str(_cd) not in sys.path:
        sys.path.insert(0, str(_cd))
    _R = importlib.import_module("recovery.recover")
    STATE.update(R=_R, code_dir=_cd)
    mo.md(f"code at `{_cd}`; recovery lib `{_R.LIB_VERSION}`")
    return


@app.cell(hide_code=True)
def _():
    dl_btn = mo.ui.run_button(label="④ Download + decompress")
    dl_btn
    return (dl_btn,)


@app.cell(hide_code=True)
def _(comp_repo, dl_btn, teacher_repo, tok):
    mo.stop(not dl_btn.value, mo.md("*press ④*"))
    mo.stop(STATE["code_dir"] is None, mo.md("**Run ③ first.**"))
    from huggingface_hub import snapshot_download as _snap
    if tok():
        os.environ["HF_TOKEN"] = tok()
    _cd = Path(_snap(comp_repo.value.strip(), token=tok(), local_dir=str(WORK / "compressed")))
    _td = Path(_snap(teacher_repo.value.strip(), token=tok(), local_dir=str(WORK / "teacher")))
    _frac = scored_gib(_cd) / GRADER_TEXT_GIB
    print(f"compressed: {_cd} size_frac {_frac:.4f} | teacher: {_td}", flush=True)
    _r0 = WORK / "restored0"
    shutil.rmtree(_r0, ignore_errors=True)
    run_streaming([PY, "decompress.py", "--model_name", "Qwen-3.5-4B", "--checkpoint_path", _cd,
                   "--output_path", _r0], cwd=STATE["code_dir"])
    STATE.update(comp_dir=_cd, teacher_dir=_td, restored0=_r0, size_frac0=_frac)
    import hashlib as _h
    _sha = _h.sha256()
    for _f in sorted(_cd.glob("*.safetensors")):
        _sha.update(_f.name.encode())
        _sha.update(str(_f.stat().st_size).encode())
    _sha.update((_cd / "compression_config.json").read_bytes())
    STATE["comp_sha"] = _sha.hexdigest()[:16]
    mo.md(f"**Compressed** size_frac = {_frac:.4f}; decompressed to `{_r0}`.")
    return


@app.cell(hide_code=True)
def _():
    max_len = mo.ui.number(start=1024, stop=32768, step=1024, value=8192, label="tokens per training sequence")
    use_fla = mo.ui.checkbox(value=True, label="use the fla delta-rule kernel (checked against PyTorch in ⑤)")
    build_btn = mo.ui.run_button(label="⑤ Build student + teacher + data (GPU)")
    mo.vstack([mo.md("## ⑤ Build\nStudent = decompressed checkpoint with every coded weight rebuilt "
                     "from its fixed codes (must be **bit-identical** to the decompressed weights). "
                     "Then the KL before training on ~24K held-out tokens."),
               mo.hstack([max_len, use_fla], justify="start"), build_btn])
    return build_btn, max_len, use_fla


@app.cell(hide_code=True)
def _(build_btn, heldout, max_len, tok, traces_repo, train_set, use_fla):
    mo.stop(not build_btn.value, mo.md("*press ⑤*"))
    mo.stop(STATE["restored0"] is None, mo.md("**Run ④ first.**"))
    _torch = importlib.import_module("torch")
    import pandas as _pd
    _AM = importlib.import_module("transformers").AutoModelForCausalLM
    _hub = importlib.import_module("huggingface_hub")
    _Api, _hd = _hub.HfApi, _hub.hf_hub_download
    _R = STATE["R"]
    _dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    for _k in ("student", "teacher"):
        if STATE[_k] is not None:
            STATE[_k] = None
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    _student, _codec, _raw = _R.build_student(STATE["restored0"], STATE["comp_dir"], _dev,
                                              log=lambda s: print(s, flush=True))
    _bad = _R.exactness_check(_codec, STATE["restored0"], log=lambda s: print(s, flush=True))
    mo.stop(bool(_bad), mo.md(f"**Student does not match the decompressed checkpoint:** {_bad[:5]} - send me this"))
    _student.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    _teacher = _AM.from_pretrained(str(STATE["teacher_dir"]), dtype=_torch.bfloat16,
                                   attn_implementation="sdpa").to(_dev).eval()
    _R.force_torch_kernels(_teacher)
    for _p in _teacher.parameters():
        _p.requires_grad_(False)
    # data
    _repo, _, _path = train_set.value.strip().partition(":")
    _ts = _pd.read_parquet(_hd(_repo, _path, repo_type="dataset", token=tok()))
    _train = _R.cut_examples(_ts.prompt_ids, _ts.completion_ids, int(max_len.value))
    # held-out traces (base model, seed 42, finished), first 4096 tokens, ~24K completion tokens
    _src = [x.strip() for x in heldout.value.split(",") if x.strip()]
    _files = [f for f in _Api(token=tok()).list_repo_files(traces_repo.value.strip(), repo_type="dataset")
              if f.startswith("raw/") and f.endswith(".parquet") and Path(f).stem in _src]
    _df = _pd.concat([_pd.read_parquet(_hd(traces_repo.value.strip(), f, repo_type="dataset", token=tok()))
                      for f in _files], ignore_index=True)
    _df = _df[(_df.seed == 42) & (_df.finish_reason == "stop")].sample(frac=1.0, random_state=0)
    _held, _n = [], 0
    for _r in _df.itertuples():
        _p, _c = list(_r.prompt_token_ids), list(_r.completion_token_ids)
        _c = _c[: max(0, 4096 - len(_p))]
        if len(_c) < 256:
            continue
        _held.append((_p, _c))
        _n += len(_c)
        if _n >= 24000:
            break
    # fla: check on one real sequence that the student's loss matches PyTorch's
    _fla_ok = False
    if use_fla.value:
        try:
            _ids = _torch.tensor(list(_train[0][0]) + list(_train[0][1])[:4000], device=_dev)
            _vals = []
            for _k in (False, True):
                _R.set_delta_rule(_student, _k)
                _R.set_delta_rule(_teacher, _k)
                with _torch.no_grad():
                    _ht = _R.teacher_hidden(_teacher, _ids, len(_train[0][0]))
                    _ls, _nn = _R.kd_loss_sum(_student, _ids, len(_train[0][0]), _ht, _R.teacher_head(_teacher))
                _vals.append(float(_ls) / _nn)
            _fla_ok = abs(_vals[0] - _vals[1]) <= 0.02 * max(abs(_vals[0]), 1e-6) + 1e-5
            print(f"fla check: KL torch {_vals[0]:.6f} vs fla {_vals[1]:.6f} -> {'OK' if _fla_ok else 'MISMATCH'}")
        except Exception as _e:
            print("fla not usable:", type(_e).__name__, str(_e)[:200])
    _R.set_delta_rule(_student, _fla_ok)
    _R.set_delta_rule(_teacher, _fla_ok)
    _student.eval()
    _kl0 = _R.mean_kl(_student, _teacher, _held, _dev)
    STATE.update(student=_student, teacher=_teacher, codec=_codec, raw=_raw, train=_train, held=_held,
                 kl_before=_kl0, device=_dev, fla=_fla_ok)
    _groups, _counts = _R.trainable_groups(_codec, _raw, 1, 1, 1)
    mo.md(f"**Student ready**: {len(_codec)} coded tensors (bit-identical to the decompressed weights), "
          f"trainable {', '.join(f'{k} {v / 1e6:.2f} M' for k, v in _counts.items())}.  \n"
          f"**Data**: {len(_train)} training sequences (≤ {int(max_len.value)} tokens), "
          f"{sum(len(c) for _, c in _train) / 1e6:.1f} M completion tokens; held-out {len(_held)} traces / {_n} tokens.  \n"
          f"Kernel: {'fla' if _fla_ok else 'PyTorch'}.  \n**KL(teacher ‖ compressed) before = {_kl0:.5f}** nats/token")
    return


@app.cell(hide_code=True)
def _():
    smoke_btn = mo.ui.run_button(label="⑥ Time one training sequence")
    smoke_btn
    return (smoke_btn,)


@app.cell(hide_code=True)
def _(smoke_btn):
    mo.stop(not smoke_btn.value, mo.md("*press ⑥*"))
    mo.stop(STATE["student"] is None, mo.md("**Run ⑤ first.**"))
    _torch = importlib.import_module("torch")
    _R, _s, _t, _dev = STATE["R"], STATE["student"], STATE["teacher"], STATE["device"]
    _p, _c = max(STATE["train"], key=lambda e: len(e[1]))
    _ids = _torch.tensor(list(_p) + list(_c), device=_dev)
    _s.train()
    _cuda = _torch.cuda.is_available()
    if _cuda:
        _torch.cuda.reset_peak_memory_stats()
    _times = []
    for _i in range(2):
        if _cuda:
            _torch.cuda.synchronize()
        _t0 = time.time()
        _ht = _R.teacher_hidden(_t, _ids, len(_p))
        _ls, _n = _R.kd_loss_sum(_s, _ids, len(_p), _ht, _R.teacher_head(_t))
        (_ls / _n).backward()
        if _cuda:
            _torch.cuda.synchronize()
        _times.append(time.time() - _t0)
        _s.zero_grad(set_to_none=True)
    _sec = _times[-1]
    _tot = sum(len(p) + len(c) for p, c in STATE["train"])
    _est = _tot / (len(_p) + len(_c)) * _sec / 3600
    mo.md(f"Longest sequence ({len(_p) + len(_c)} tokens): **{_sec:.1f} s** (first call {_times[0]:.1f} s), "
          f"peak {(_torch.cuda.max_memory_allocated() / 2**30) if _cuda else 0:.1f} GiB.  \n"
          f"**Estimated 1 epoch: ~{_est:.1f} h** (≈ linear in tokens).")
    return


@app.cell(hide_code=True)
def _():
    lr_scale = mo.ui.number(start=1e-5, stop=1e-1, step=1e-5, value=2e-3, label="LR log-step multipliers")
    lr_out = mo.ui.number(start=1e-6, stop=1e-2, step=1e-6, value=1e-4, label="LR outlier values")
    lr_raw = mo.ui.number(start=0.0, stop=1e-2, step=1e-6, value=5e-5, label="LR raw tensors (norms...)")
    seqs = mo.ui.number(start=1, stop=64, step=1, value=8, label="sequences per step")
    epochs = mo.ui.number(start=1, stop=4, step=1, value=1, label="epochs")
    warmup = mo.ui.number(start=0, stop=50, step=1, value=5, label="warm-up steps")
    ckpt = mo.ui.number(start=5, stop=120, step=5, value=20, label="checkpoint every N min")
    train_btn = mo.ui.run_button(label="⑦ Train (resumes automatically)")
    mo.vstack([mo.md("## ⑦ Train\nAdamW, warm-up + cosine to 10%, grad-clip 1.0, KL token-mean. "
                     "Pressing ⑦ again resumes with the same settings (a change is refused)."),
               mo.hstack([lr_scale, lr_out, lr_raw], justify="start"),
               mo.hstack([seqs, epochs, warmup, ckpt], justify="start"), train_btn])
    return ckpt, epochs, lr_out, lr_raw, lr_scale, seqs, train_btn, warmup


@app.cell(hide_code=True)
def _(ckpt, epochs, lr_out, lr_raw, lr_scale, pull_run, push_run, run_dir, seqs, train_btn, warmup):
    mo.stop(not train_btn.value, mo.md("*press ⑦*"))
    mo.stop(STATE["student"] is None, mo.md("**Run ⑤ first.**"))
    _R = STATE["R"]
    _cfg = dict(lr_scale=float(lr_scale.value), lr_outlier=float(lr_out.value), lr_raw=float(lr_raw.value),
                seqs_per_step=int(seqs.value), epochs=int(epochs.value), seed=0,
                max_len=max(len(p) + len(c) for p, c in STATE["train"]), warmup=int(warmup.value),
                ckpt_every_min=float(ckpt.value), chunk=1024,
                data_fingerprint=_R.data_fingerprint(STATE["train"]), compressed_sha=STATE["comp_sha"])
    _sd = run_dir() / "state"
    if not (_sd / "state.json").exists() and pull_run("state"):
        print("pulled checkpoint from HF", flush=True)
    _step, _total, _log, _counts = _R.train(
        STATE["student"], STATE["teacher"], STATE["codec"], STATE["raw"], STATE["train"], _cfg, _sd,
        STATE["device"], log=lambda s: print(s, flush=True),
        on_checkpoint=lambda st: push_run("state", f"recovery checkpoint step {st}"))
    _pd = importlib.import_module("pandas")
    _lg = _pd.DataFrame(_log)
    import matplotlib as _mpl
    _mpl.use("Agg")
    import matplotlib.pyplot as _plt
    _fig, _ax = _plt.subplots(1, 2, figsize=(9, 3))
    _ax[0].plot(_lg.step, _lg.kl)
    _ax[0].set_title("KL(teacher || student), token mean", fontsize=9)
    _ax[1].plot(_lg.step, _lg.grad_norm, color="#eb6834")
    _ax[1].set_title("grad norm", fontsize=9)
    for _a in _ax:
        _a.set_xlabel("step", fontsize=8)
    _fig.tight_layout()
    _fig.savefig(run_dir() / "state" / "recovery_curves.png", dpi=130)
    mo.vstack([mo.md(f"### Training {'finished' if _step >= _total else 'stopped'}: step {_step}/{_total}"),
               _fig, mo.ui.table(_lg.tail(40), page_size=10)])
    return


@app.cell(hide_code=True)
def _():
    export_btn = mo.ui.run_button(label="⑧ Export + verify")
    mo.vstack([mo.md("## ⑧ Export + verify\nWrites trained steps / outliers / raw tensors into a copy "
                     "of the compressed checkpoint (asserts identical byte size), decompresses it with "
                     "the unchanged `decompress.py`, checks it equals the trained student exactly, "
                     "then measures the held-out KL of the **decompressed** model (what the graders "
                     "run) and prints a greedy sample."), export_btn])
    return (export_btn,)


@app.cell(hide_code=True)
def _(export_btn, pull_run, run_dir):
    mo.stop(not export_btn.value, mo.md("*press ⑧*"))
    mo.stop(STATE["student"] is None, mo.md("**Run ⑤ first.**"))
    _torch = importlib.import_module("torch")
    _R, _dev = STATE["R"], STATE["device"]
    _sd = run_dir() / "state"
    if not (_sd / "state.json").exists():
        pull_run("state")
    mo.stop(not (_sd / "state.json").exists(), mo.md("**No checkpoint - train first.**"))
    _saved = _torch.load(_sd / "trainables.pt", map_location="cpu")
    _tr = _R._trainables(STATE["codec"], STATE["raw"])
    with _torch.no_grad():
        for _k, _v in _saved.items():
            _tr[_k].copy_(_v.to(_tr[_k].device, _tr[_k].dtype))
    _st = json.loads((_sd / "state.json").read_text())
    _out = WORK / "recovered"
    _info = dict(step=_st["step"], cfg=_st["cfg"], lib=_R.LIB_VERSION, kl_before=STATE["kl_before"],
                 created=datetime.now(timezone.utc).isoformat())
    _R.export(STATE["comp_dir"], _out, STATE["codec"], STATE["raw"], log=lambda s: print(s, flush=True))
    _b0, _b1 = _R.dir_bytes(STATE["comp_dir"]), _R.dir_bytes(_out)
    _frac = scored_gib(_out) / GRADER_TEXT_GIB
    print(f"bytes {_b0} -> {_b1}; size_frac {_frac:.4f}", flush=True)
    mo.stop(_b0 != _b1, mo.md("**Size changed - do not upload; send me this.**"))
    _r1 = WORK / "restored1"
    shutil.rmtree(_r1, ignore_errors=True)
    run_streaming([PY, "decompress.py", "--model_name", "Qwen-3.5-4B", "--checkpoint_path", _out,
                   "--output_path", _r1], cwd=STATE["code_dir"])
    _bad = _R.compare_restored(STATE["codec"], _r1, log=lambda s: print(s, flush=True))
    mo.stop(bool(_bad), mo.md(f"**Decompressed != student:** {_bad[:5]} - send me this"))
    _tfm = importlib.import_module("transformers")
    _AM, _AT = _tfm.AutoModelForCausalLM, _tfm.AutoTokenizer
    _m = _AM.from_pretrained(str(_r1), dtype=_torch.bfloat16, attn_implementation="sdpa").to(_dev).eval()
    _R.force_torch_kernels(_m)
    _R.set_delta_rule(_m, STATE.get("fla", False))
    _kl1 = _R.mean_kl(_m, STATE["teacher"], STATE["held"], _dev, parametrized=False)
    try:
        _tk = _AT.from_pretrained(str(_r1))
        _msgs = [{"role": "user", "content": "What is 17 * 23? Put the answer in \\boxed{}."}]
        _pr = _tk.apply_chat_template(_msgs, tokenize=False, add_generation_prompt=True,
                                      enable_thinking=False)
        _x = _tk(_pr, return_tensors="pt").input_ids.to(_dev)
        _g = _m.generate(_x, max_new_tokens=60, do_sample=False)
        _sample = _tk.decode(_g[0, _x.shape[1]:], skip_special_tokens=True)
    except Exception as _e:
        _sample = f"(no sample: {type(_e).__name__}: {str(_e)[:120]})"
    del _m
    if _torch.cuda.is_available():
        _torch.cuda.empty_cache()
    _info.update(kl_after=_kl1, size_frac=_frac, greedy_sample=_sample)
    (run_dir() / "export").mkdir(exist_ok=True)
    (run_dir() / "export" / "recovery_info.json").write_text(json.dumps(_info, indent=2, default=str))
    _better = _kl1 < STATE["kl_before"]
    STATE.update(out_dir=_out, verified=_better, info=_info)
    mo.md(f"**Size unchanged** ({_b1:,} bytes, size_frac {_frac:.4f}); decompressed == student.  \n"
          f"**Held-out KL(teacher ‖ model): before {STATE['kl_before']:.5f} → after {_kl1:.5f} "
          f"({100 * (_kl1 / STATE['kl_before'] - 1):+.1f}%)**  \n"
          f"Greedy sample: `{_sample!r}`  \n"
          + ("✅ better than the plain compressed model - OK to upload (⑨)." if _better else
             "❌ not better - keep the plain compressed submission."))
    return


@app.cell(hide_code=True)
def _():
    up_btn = mo.ui.run_button(label="⑨ Upload recovered checkpoint")
    up_btn
    return (up_btn,)


@app.cell(hide_code=True)
def _(out_repo, push_run, run_dir, tok, up_btn):
    mo.stop(not up_btn.value, mo.md("*press ⑨ after ⑧*"))
    mo.stop(not STATE["verified"], mo.md("**⑧ did not pass (or was not run) - not uploading.**"))
    _api = importlib.import_module("huggingface_hub").HfApi(token=tok())
    _repo = out_repo.value.strip()
    _api.create_repo(_repo, repo_type="model", private=True, exist_ok=True)
    _api.upload_folder(repo_id=_repo, repo_type="model", folder_path=str(STATE["out_dir"]),
                       commit_message="codec-E checkpoint + size-neutral recovery (KD from the fine-tuned model)")
    push_run("export", "recovery info")
    mo.md(f"**Uploaded** to `{_repo}` @ `{_api.model_info(_repo).sha[:12]}`.")
    return


if __name__ == "__main__":
    app.run()
