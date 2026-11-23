import marimo

__generated_with = "0.25.0"
app = marimo.App(width="medium", app_title="CS6013 nb2 - base-model training-trace run")

with app.setup:
    import importlib
    import json
    import os
    import shutil
    import socket
    import subprocess
    import sys
    import time
    import urllib.request
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datetime import datetime, timezone
    from pathlib import Path

    import marimo as mo

    NB_VERSION = "nb2-v1.1 (2026-11-23, adaptive sampling)"
    WORK = Path("/root/work/nb2") if Path("/root").exists() else Path.home() / "work" / "nb2"
    WORK.mkdir(parents=True, exist_ok=True)
    PY = sys.executable
    RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Local layout (mirrors the HF dataset repo):
    #   pool/pool.parquet  raw/<module>/<source>.jsonl  derived/<module>.parquet
    #   meta/run_meta_<run_id>.json   stage/ (parquet copies staged for upload)
    for _d in ("pool", "raw", "derived", "meta", "stage"):
        (WORK / _d).mkdir(exist_ok=True)

    # Shared mutable state. Every heavy step is behind a button, so cells talk
    # through this dict instead of marimo's reactive graph (a stray re-run must
    # never restart inference).
    STATE = {"L": None, "pool": None, "model_path": None, "meta": None,
             "vllm_proc": None, "port": None}

    # nb2lib: grader code (copied from evalkit/evaluation/common.py), pool
    # builders, request function, raw storage, derived stats. Written to disk so
    # stats can be derived in a subprocess (math_verify's timeout needs the
    # main thread of a process).
    LIB_SRC = r'''# nb2lib.py -- helper library for the CS6013 training-trace run (notebook 2;
# a copy of nb1lib with the training module T added).
# The notebook writes this file to disk and imports it; derived stats are
# computed by running it as a subprocess (math_verify's timeout uses
# signal.alarm, which only works in a process's main thread).
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

LIB_VERSION = "nb2lib-2026-11-23b"

# ===========================================================================
# 1. GRADER CODE -- copied verbatim from evalkit/evaluation/common.py
#    (extract_boxed_answer, normalize_answer, answers_match). Do not edit:
#    correct_grader must mean exactly what the hidden grader means.
# ===========================================================================
from math_verify import parse, verify
from math_verify.parser import (
    LatexExtractionConfig,
    ExprExtractionConfig,
)

latex_config = LatexExtractionConfig(
    boxed_match_priority=0,
    try_extract_without_anchor=True,
)

expr_config = ExprExtractionConfig()


def extract_boxed_answer(text: str) -> str | None:
    """Extract the last balanced \\boxed{...} answer."""

    i = text.rfind(r"\boxed")

    if i < 0:
        return None

    start = text.find("{", i)

    if start < 0:
        return None

    depth = 0

    for j in range(start, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1

            if depth == 0:
                return text[start + 1:j].strip()

    return None


def normalize_answer(answer: str) -> str:
    """Normalize common formatting differences."""

    answer = answer.strip()

    # Remove surrounding $ delimiters.
    answer = answer.strip("$").strip()

    # Remove LaTeX spacing commands.
    answer = re.sub(r"\\[,;:!]", "", answer)

    # Normalize common LaTeX commands.
    answer = answer.replace(r"\left", "")
    answer = answer.replace(r"\right", "")

    # Normalize whitespace.
    answer = re.sub(r"\s+", "", answer)

    # Normalize \dfrac and \tfrac to \frac.
    answer = answer.replace(r"\dfrac", r"\frac")
    answer = answer.replace(r"\tfrac", r"\frac")

    # Remove thousands separators.
    answer = re.sub(r"(?<=\d),(?=\d)", "", answer)

    # Normalize braces around simple values.
    if (
        len(answer) >= 2
        and answer.startswith("{")
        and answer.endswith("}")
    ):
        answer = answer[1:-1]

    return answer


def answers_match(prediction, gold):
    if prediction is None:
        return (False, None, [], normalize_answer(gold), [])

    prediction_normalized = normalize_answer(prediction)
    gold_normalized = normalize_answer(gold)

    if prediction_normalized is None:
        return (False, None, [], gold_normalized, [])

    try:
        prediction_wrapped = f"\\({prediction_normalized}\\)"
        gold_wrapped = f"\\({gold_normalized}\\)"
        predicted_parsed_raw = parse(
            prediction_wrapped, extraction_config=[latex_config, expr_config])
        gold_parsed_raw = parse(
            gold_wrapped, extraction_config=[latex_config, expr_config])
        predicted_parsed = [str(x) for x in predicted_parsed_raw]
        gold_parsed = [str(x) for x in gold_parsed_raw]
    except Exception as exc:
        logging.warning("math-verify failed: prediction=%r gold=%r error=%s",
                        prediction, gold, exc)
        return (False, prediction_normalized, [], gold_normalized, [])

    # Exact match after normalization.
    if prediction_normalized == gold_normalized:
        return (True, prediction_normalized, predicted_parsed,
                gold_normalized, gold_parsed)

    # Mathematical equivalence.
    if not predicted_parsed_raw or not gold_parsed_raw:
        return (False, prediction_normalized, predicted_parsed,
                gold_normalized, gold_parsed)

    try:
        correct = verify(gold_parsed_raw, predicted_parsed_raw)
    except Exception as exc:
        logging.warning("verify failed: prediction=%r gold=%r error=%s",
                        prediction, gold, exc)
        correct = False

    return (correct, prediction_normalized, predicted_parsed,
            gold_normalized, gold_parsed)


def build_math_prompt(problem: str, math_instruction: str) -> str:
    return f"{math_instruction}\n\nProblem:\n{problem.strip()}"


# ===========================================================================
# 2. Grader constants (from configs/eval_config.yaml + the graders' eval.sh)
# ===========================================================================
MATH_INSTRUCTION = (
    "Answer the following mathematics problem. \n"
    "Please reason step by step, but keep the reasoning concise. \n"
    "After reasoning, output the final answer on its own last line using "
    "exactly this format: \\boxed{X}, \n"
    "where X is the final mathematical answer. Generate \\boxed{X} exactly once. \n"
    "Do not write anything after that line.\n"
)
SAMPLING = dict(
    max_new_tokens=32000,
    temperature=1.0,
    top_p=0.95,
    top_k=20,
    min_p=0.0,
    presence_penalty=1.5,
    repetition_penalty=1.0,
    enable_thinking=True,
)
GRADER_SEED = 42           # hard-coded in common.py; sample i uses seed 42 + i
MAX_MODEL_LEN = 33000      # 1000 + max_new_tokens, as in the graders' eval.sh
SERVED_MODEL_NAME = "qwen-3.5-4b"

# nb2 generates only module T. A and B are listed so that the pool builder can
# drop T problems that already have base traces in nb1's repo (A: AIME 2022-26,
# B: HMMT Feb 25/26, HMMT Nov 25, BRUMO/CMIMC/SMT 25); they are never run here.
MODULES = ["T"]
MODULE_K = {"S": 2, "M": 2, "A": 4, "B": 4, "C": 4, "T": 4}
MODULE_DESC = {
    "S": "sanity: MATH-500 level 5",
    "A": "AIME by year 2022-2026",
    "B": "fresh AIME-like (HMMT/BRUMO/CMIMC/SMT 2025-26)",
    "M": "medium: GaoKao 2023 En, AMC 2022-23, OlympiadBench",
    "C": "hard: Apex Shortlist, OlymMATH-hard, BeyondAIME, Omni-MATH d>=7",
    "T": "training pool: HMMT Feb/Nov archive 1999-2024 (main) + AIME archive",
}


# ===========================================================================
# 3. Pool building
# ===========================================================================
def clean_answer(value):
    """Gold answer -> clean string (42.0 -> '42'). Same as evalkit's _clean."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(int(value)) if value.is_integer() else str(value)
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            return None
        return clean_answer(value[0])
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"-?\d+\.0+", text):
        text = text.split(".")[0]
    return text


def gold_parses(answer: str) -> bool:
    """Can the grader's math_verify config parse this gold answer?
    (parsing_timeout=None so it is safe inside a notebook thread.)"""
    try:
        g = normalize_answer(answer)
        return bool(parse(f"\\({g}\\)", extraction_config=[latex_config, expr_config],
                          parsing_timeout=None))
    except Exception:
        return False


_IMG_RE = re.compile(r"!\[|<img|\.png\b|\.jpe?g\b|\\includegraphics", re.I)
_MC_ANS_RE = re.compile(r"^\(?[A-E]\)?(\s*,?\s*\(?[A-E]\)?)*$")
_MC_Q_RE = re.compile(r"(^|\n|\s)\(?A[\.\):．]\s*.+(\n|\s)\(?B[\.\):．]", re.S)
_WORDY_RE = re.compile(r"[A-Za-z]{3,}")


# Only GaoKao actually contains multiple-choice items. The question-text pattern
# is loose ("... point A. ... B.") and wrongly hit 3 AIME-2025 + 1 HMMT problems,
# so it is applied to these sources only; elsewhere only a letter-valued gold
# answer counts as multiple choice.
MC_QUESTION_SOURCES = {"gaokao2023_en"}


def is_multiple_choice(problem: str, answer: str, check_question: bool = False) -> bool:
    if _MC_ANS_RE.match(answer.strip()):
        return True
    return bool(check_question and _MC_Q_RE.search(problem))


def is_wordy(answer: str) -> bool:
    """Answer that is words rather than math (after dropping LaTeX commands)."""
    stripped = re.sub(r"\\[A-Za-z]+", " ", answer)
    return bool(_WORDY_RE.search(stripped))


def dedup_key(problem: str) -> str:
    return re.sub(r"[^a-z0-9]", "", problem.lower())[:300]


def _pick_split(dsd, prefer=("test", "train", "validation")):
    names = list(dsd.keys())
    for p in prefer:
        if p in names:
            return dsd[p], p
    return dsd[names[0]], names[0]


def _first(row, *fields):
    for f in fields:
        v = row.get(f)
        if v is not None and (not isinstance(v, str) or v.strip()):
            return v
    return None


# --- per-source row adapters: row -> dict(problem, answer, idx, year, topic, level)
def _ad_math500(row, i):
    lvl = row.get("level")
    try:
        lvl = int(str(lvl).replace("Level", "").strip())
    except Exception:
        lvl = None
    return dict(problem=row.get("problem"), answer=clean_answer(row.get("answer")),
                idx=str(row.get("unique_id") or i), year=None,
                topic=row.get("subject") or row.get("type"), level=lvl)


def _ad_aimo(kind):
    pat = (re.compile(r"(\d{4})_AIME_(I{1,2})_Problems/Problem_(\d+)") if kind == "aime"
           else re.compile(r"(\d{4})_AMC_(\w+?)_Problems/Problem_(\d+)"))

    def f(row, i):
        m = pat.search(str(row.get("url") or ""))
        year, idx = None, str(row.get("id", i))
        if m:
            year = int(m.group(1))
            idx = f"{m.group(1)}_{m.group(2)}_{int(m.group(3)):02d}"
        return dict(problem=row.get("problem"), answer=clean_answer(row.get("answer")),
                    idx=idx, year=year, topic=None, level=None)
    return f


def _ad_matharena(year):
    def f(row, i):
        pt = row.get("problem_type")
        if isinstance(pt, (list, tuple)):
            pt = ",".join(map(str, pt))
        return dict(problem=row.get("problem"), answer=clean_answer(row.get("answer")),
                    idx=str(row.get("problem_idx", i + 1)), year=year,
                    topic=pt, level=None)
    return f


def _ad_generic(row, i):
    return dict(problem=_first(row, "problem", "question", "Problem", "Question"),
                answer=clean_answer(_first(row, "answer", "final_answer", "Answer")),
                idx=str(_first(row, "unique_id", "id", "problem_idx") or i), year=None,
                topic=_first(row, "subject", "domain", "type"), level=None)


def _ad_olympiadbench(row, i):
    if row.get("is_multiple_answer") or (row.get("unit") or "").strip():
        return None
    fa = row.get("final_answer")
    if isinstance(fa, (list, tuple)):
        if len(fa) != 1:
            return None
        fa = fa[0]
    ans = clean_answer(fa)
    if ans:
        ans = ans.strip().strip("$").strip()
    q = row.get("question") or ""
    ctx = row.get("context")
    if ctx:
        q = f"{ctx}\n\n{q}"
    return dict(problem=q, answer=ans, idx=str(row.get("id", i)), year=None,
                topic=row.get("subfield") or row.get("subject"), level=None)


def _ad_omni(row, i):
    d = row.get("difficulty")
    try:
        d = float(d)
    except Exception:
        d = None
    dom = row.get("domain")
    if isinstance(dom, (list, tuple)):
        dom = dom[0] if dom else None
    return dict(problem=row.get("problem"), answer=clean_answer(row.get("answer")),
                idx=str(i), year=None, topic=dom, level=d)


# (module, source, hf_id, config, adapter, cap, row_filter, strict_gold)
#   cap: None = keep all; otherwise deterministic random sample (seed 0)
#   strict_gold: drop problems whose gold answer the grader's parser can't read
#               (used for the noisier free-form sources)
def source_specs(include_math_l4=False, include_olymmath_easy=False, caps=None):
    caps = caps or {}
    c = lambda k, d: caps.get(k, d)
    specs = [
        # ---- A: AIME by year (closest to the grader; per-year = contamination test)
        ("A", "aime_2022", "AI-MO/aimo-validation-aime", None, _ad_aimo("aime"), None,
         lambda r: r["year"] == 2022, False),
        ("A", "aime_2023", "AI-MO/aimo-validation-aime", None, _ad_aimo("aime"), None,
         lambda r: r["year"] == 2023, False),
        ("A", "aime_2024", "AI-MO/aimo-validation-aime", None, _ad_aimo("aime"), None,
         lambda r: r["year"] == 2024, False),
        ("A", "aime_2025", "MathArena/aime_2025", None, _ad_matharena(2025), None, None, False),
        ("A", "aime_2026", "MathArena/aime_2026", None, _ad_matharena(2026), None, None, False),
        # ---- B: fresh AIME-like
        ("B", "hmmt_feb_2025", "MathArena/hmmt_feb_2025", None, _ad_matharena(2025), None, None, False),
        ("B", "hmmt_feb_2026", "MathArena/hmmt_feb_2026", None, _ad_matharena(2026), None, None, False),
        ("B", "hmmt_nov_2025", "MathArena/hmmt_nov_2025", None, _ad_matharena(2025), None, None, False),
        ("B", "brumo_2025", "MathArena/brumo_2025", None, _ad_matharena(2025), None, None, False),
        ("B", "cmimc_2025", "MathArena/cmimc_2025", None, _ad_matharena(2025), None, None, False),
        ("B", "smt_2025", "MathArena/smt_2025", None, _ad_matharena(2025), None, None, False),
        # ---- S: sanity
        ("S", "math500_l5", "HuggingFaceH4/MATH-500", None, _ad_math500, None,
         lambda r: r["level"] == 5, False),
        # ---- M: medium
        ("M", "gaokao2023_en", "MARIO-Math-Reasoning/Gaokao2023-Math-En", None, _ad_generic,
         c("gaokao2023_en", 120), None, True),
        ("M", "amc_2022_23", "AI-MO/aimo-validation-amc", None, _ad_aimo("amc"), None, None, False),
        ("M", "olympiadbench", "Hothan/OlympiadBench", "OE_TO_maths_en_COMP", _ad_olympiadbench,
         c("olympiadbench", 100), None, True),
        # ---- C: hard
        ("C", "apex_shortlist", "MathArena/apex_shortlist", None, _ad_matharena(2025), None, None, False),
        ("C", "olymmath_hard", "RUC-AIBOX/OlymMATH", "en-hard", _ad_generic,
         c("olymmath_hard", 100), None, True),
        ("C", "beyondaime", "ByteDance-Seed/BeyondAIME", None, _ad_generic,
         c("beyondaime", 100), None, True),
        ("C", "omni_math_d7", "KbsdJames/Omni-MATH", None, _ad_omni, c("omni_math_d7", 50),
         lambda r: (r["level"] or 0) >= 7, True),
    ]
    if include_math_l4:
        specs.append(("S", "math500_l4", "HuggingFaceH4/MATH-500", None, _ad_math500, None,
                      lambda r: r["level"] == 4, False))
    if include_olymmath_easy:
        specs.append(("C", "olymmath_easy", "RUC-AIBOX/OlymMATH", "en-easy", _ad_generic,
                      c("olymmath_easy", 100), None, True))
    return specs


# ---------------------------------------------------------------------------
# 3b. Training pool (module T) -- nb2 only
# ---------------------------------------------------------------------------
# Easy2Hard-Bench / E2H-AMC (CC BY-SA 4.0): AMC, AIME and HMMT 1999-2024 with
# answers and difficulty ratings, split into "train" (1000) and "eval" (2975)
# for the authors' purposes. Both splits are used here.
E2H_ID, E2H_CONFIG = "furonghuang-lab/Easy2Hard-Bench", "E2H-AMC"
ALL_SPLITS = {E2H_ID}     # hf_ids whose splits are concatenated instead of picking one

# HMMT estimation problems are scored by closeness, not exact match -> no gold.
# Kept narrow on purpose: ordinary problems also talk about teams "receiving points".
_ESTIMATE_RE = re.compile(
    r"\ban\s+estimate\s+of\b|\byour\s+(answer|estimate|submission)\s+is\b|"
    r"\b(correct|actual)\s+answer\s+is\s+\$?[A-Z]\$?\b[^.]{0,160}\bpoints?\b|"
    r"\bpoints?\b[^.]{0,100}\b(correct|actual)\s+answer\b", re.I)
# Guts problems that use another problem's answer ("Let A be the answer to problem 13").
_XREF_RE = re.compile(r"answer\s+to\s+(this|problem|question|the\s+(previous|next|other|"
                      r"preceding|following))|answers?\s+to\s+problems?\s+\d", re.I)


def e2h_drop_reason(problem, answer, subtest, index):
    """Why an E2H HMMT/AIME row can't be scored by the grader, or None."""
    p = problem or ""
    a = (answer or "").strip()
    if subtest == "guts" and index is not None and index >= 33:
        return "estimation"            # last guts set (33-36) is estimation, Feb and Nov
    if _ESTIMATE_RE.search(p):
        return "estimation"
    if _XREF_RE.search(p):
        return "cross_reference"
    # several values / ordered pairs / intervals: the grader's normalize_answer
    # deletes digit,digit commas, so "98,118" would become the number 98118.
    # "98,118,122,142" is indistinguishable from a thousands-separated number,
    # so every comma answer goes (a few genuine "1,000,000" answers are lost).
    if "," in a:
        return "multi_value"
    if re.search(r"\\(text|mathrm)\{\s*(and|or)\s*\}|\b(and|or)\b", a):
        return "multi_value"
    return None


def _ad_e2h(row, i):
    """E2H-AMC row -> pool row. Extra keys: contest, month, subtest, index,
    _drop (reason or None)."""
    contest = row.get("contest")
    month = row.get("month")
    subtest = str(row.get("subtest") or "")
    try:
        index = int(row.get("index"))
    except Exception:
        index = None
    year = row.get("year")
    try:
        year = int(year)
    except Exception:
        year = None
    ans = clean_answer(row.get("answer"))
    if ans:
        ans = ans.strip().strip("$").strip()
    if contest == "HMMT":
        idx = f"{year}_{month}_{subtest}_{index:02d}" if index is not None else str(i)
    else:
        idx = f"{year}_{subtest}_{index:02d}" if index is not None else str(i)
    try:
        rating = float(row.get("rating"))
    except Exception:
        rating = None
    return dict(problem=row.get("problem"), answer=ans, idx=idx, year=year,
                topic=row.get("test_tag") or row.get("tag"), level=rating,
                contest=contest, month=month, subtest=subtest, index=index,
                _drop=e2h_drop_reason(row.get("problem"), ans, subtest, index))


def train_specs():
    """Module T. Order = de-duplication priority inside T (MathArena's cleaner
    2024 copy first, then the E2H archive)."""
    return [
        ("T", "hmmt_feb_2024", "MathArena/hmmt_feb_2024", None, _ad_matharena(2024), None,
         None, False),
        ("T", "e2h_hmmt_feb", E2H_ID, E2H_CONFIG, _ad_e2h, None,
         lambda r: r["contest"] == "HMMT" and r["month"] == "Feb", True),
        ("T", "e2h_hmmt_nov", E2H_ID, E2H_CONFIG, _ad_e2h, None,
         lambda r: r["contest"] == "HMMT" and r["month"] == "Nov", True),
        ("T", "e2h_aime", E2H_ID, E2H_CONFIG, _ad_e2h, None,
         lambda r: r["contest"] == "AIME", True),
    ]


def build_train_pool(load_dataset, log=print):
    """Module-T pool. nb1's A and B specs are loaded first so that T problems
    which already have base traces there are dropped (counted as
    duplicate_of_<source>); only T rows are returned."""
    dedup = [s for s in source_specs() if s[0] in ("A", "B")]
    rows, report = build_pool(load_dataset, dedup + train_specs(), log=log)
    for r in report:
        if r["module"] != "T":
            r["error"] = (r.get("error") or "") + " (dedup reference only)"
    return [r for r in rows if r["module"] == "T"], report


def topup_select(der, first_pass_k=2, max_k=4, long_tokens=16000):
    """Adaptive sampling (nb2 option 3). After a first pass of `first_pass_k`
    samples per problem, choose the problems that get samples up to `max_k`:
      no_usable   - no sample is both correct (grader's parser) and finished < 32K
      long_only   - the shortest usable sample is longer than `long_tokens`
    Problems that don't have all first-pass samples yet are reported as
    first_pass_incomplete and NOT selected. Returns one row per problem."""
    import pandas as pd
    d = der.copy()
    d["usable"] = d.correct_grader.astype(bool) & ~d.truncated.astype(bool)
    rows = []
    for (src, uid), g in d.groupby(["source", "uid"]):
        have = set(int(x) for x in g.sample_idx)
        first_done = all(s in have for s in range(first_pass_k))
        ok = g[g.usable]
        shortest = int(ok.n_completion_tokens.min()) if len(ok) else None
        if not first_done:
            reason = "first_pass_incomplete"
        elif shortest is None:
            reason = "no_usable"
        elif shortest > long_tokens:
            reason = "long_only"
        else:
            reason = "done"
        missing = [s for s in range(max_k) if s not in have]
        rows.append(dict(source=src, uid=uid, samples=len(have), usable=int(len(ok)),
                         shortest_usable=shortest, reason=reason,
                         selected=reason in ("no_usable", "long_only") and bool(missing),
                         new_samples=len(missing) if reason in ("no_usable", "long_only") else 0))
    return pd.DataFrame(rows)


def _load_rows(load_dataset, hf_id, config):
    dsd = load_dataset(hf_id, config) if config else load_dataset(hf_id)
    if hf_id in ALL_SPLITS:
        names = list(dsd.keys())
        data, cols = [], None
        for n in names:
            data += list(dsd[n])
            cols = cols or list(dsd[n].column_names)
        return data, "+".join(names), cols
    ds, split = _pick_split(dsd)
    return list(ds), split, list(ds.column_names)


def build_pool(load_dataset, specs, log=print):
    """Returns (rows, report). Priority for de-duplication: A, B, S, M, C, T."""
    order = {m: i for i, m in enumerate(["A", "B", "S", "M", "C", "T"])}
    specs = sorted(specs, key=lambda s: order[s[0]])
    cache, seen, rows, report = {}, {}, [], []
    for (module, source, hf_id, config, adapter, cap, rfilter, strict) in specs:
        st = Counter()
        try:
            key = (hf_id, config)
            if key not in cache:
                cache[key] = _load_rows(load_dataset, hf_id, config)
                split = cache[key][1]
                log(f"  loaded {hf_id}{'/' + config if config else ''} [{split}] "
                    f"{len(cache[key][0])} rows, columns={cache[key][2]}")
            data, split, cols = cache[key]
        except Exception as exc:
            log(f"  [FAIL] {source}: {hf_id}: {type(exc).__name__}: {str(exc)[:300]}")
            report.append(dict(module=module, source=source, hf_id=hf_id, loaded=0, kept=0,
                               error=f"{type(exc).__name__}: {str(exc)[:200]}"))
            continue
        cand = []
        for i, raw in enumerate(data):
            try:
                r = adapter(raw, i)
            except Exception:
                st["adapter_error"] += 1
                continue
            if r is None:
                st["dropped_by_adapter"] += 1
                continue
            if rfilter is not None and not rfilter(r):
                continue
            st["candidates"] += 1
            if r.get("_drop"):
                st[r["_drop"]] += 1
                continue
            p, a = (r["problem"] or "").strip(), r["answer"]
            if not p or not a:
                st["missing_problem_or_answer"] += 1
                continue
            if _IMG_RE.search(p):
                st["image"] += 1
                continue
            if is_multiple_choice(p, a, check_question=source in MC_QUESTION_SOURCES):
                st["multiple_choice"] += 1
                continue
            gp = gold_parses(a)
            if strict and (not gp or is_wordy(a) or len(a) > 60):
                st["gold_unscorable"] += 1
                continue
            r["problem"], r["gold_parse_ok"] = p, gp
            cand.append(r)
        if cap is not None and len(cand) > cap:
            rng = random.Random(0)
            keep = sorted(rng.sample(range(len(cand)), cap))
            st["capped_out"] = len(cand) - cap
            cand = [cand[j] for j in keep]
        kept = 0
        for r in cand:
            k = dedup_key(r["problem"])
            if k in seen:
                st[f"duplicate_of_{seen[k]}"] += 1
                continue
            seen[k] = source
            uid = f"{source}/{r['idx']}"
            rows.append(dict(
                uid=uid, module=module, source=source, hf_id=hf_id, idx=r["idx"],
                year=r["year"], topic=None if r["topic"] is None else str(r["topic"]),
                level=None if r["level"] is None else float(r["level"]),
                problem=r["problem"], answer=r["answer"], gold_parse_ok=bool(r["gold_parse_ok"]),
                k=MODULE_K[module], prompt=build_math_prompt(r["problem"], MATH_INSTRUCTION)))
            kept += 1
        report.append(dict(module=module, source=source, hf_id=hf_id, loaded=len(data),
                           kept=kept, error="", **{f"drop_{kk}": v for kk, v in st.items()}))
        log(f"  {module} {source:<16} kept {kept:>4}   {dict(st)}")
    # uid must be unique
    cnt = Counter(r["uid"] for r in rows)
    for r in rows:
        if cnt[r["uid"]] > 1:
            r["uid"] = f"{r['uid']}#{hashlib.md5(r['problem'].encode()).hexdigest()[:6]}"
    return rows, report


# ===========================================================================
# 4. Inference: one request, exactly as the grader sends it
#    (+ per-sample seed and return_token_ids)
# ===========================================================================
def request_one(client, prompt, seed, model=SERVED_MODEL_NAME):
    s = SAMPLING
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=s["max_new_tokens"],
        temperature=s["temperature"],
        top_p=s["top_p"],
        presence_penalty=s["presence_penalty"],
        seed=seed,
        extra_body={
            "top_k": s["top_k"],
            "min_p": s["min_p"],
            "repetition_penalty": s["repetition_penalty"],
            "chat_template_kwargs": {"enable_thinking": s["enable_thinking"]},
            "return_token_ids": True,
        },
    )
    t1 = time.time()
    d = resp.model_dump()
    ch = d["choices"][0]
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    comp_ids = ch.get("token_ids")
    prompt_ids = d.get("prompt_token_ids")
    return dict(
        completion=msg.get("content") or "",
        reasoning_content=msg.get("reasoning_content") or msg.get("reasoning"),
        finish_reason=ch.get("finish_reason"),
        stop_reason=None if ch.get("stop_reason") is None else str(ch.get("stop_reason")),
        prompt_token_ids=prompt_ids,
        completion_token_ids=comp_ids,
        n_prompt_tokens=usage.get("prompt_tokens"),
        n_completion_tokens=usage.get("completion_tokens"),
        t_start=t0, t_end=t1, latency_s=round(t1 - t0, 2),
    )


# ===========================================================================
# 5. Raw storage helpers (local working copy = JSONL, published = parquet)
# ===========================================================================
_STR_COLS = {"uid", "module", "source", "prompt", "completion", "reasoning_content",
             "finish_reason", "stop_reason", "run_id"}
RAW_COLUMNS = [
    "uid", "module", "source", "sample_idx", "seed", "prompt", "prompt_token_ids",
    "completion", "reasoning_content", "completion_token_ids", "n_prompt_tokens",
    "n_completion_tokens", "finish_reason", "stop_reason", "t_start", "t_end",
    "latency_s", "run_id",
]


def read_jsonl(path):
    out = []
    p = Path(path)
    if not p.exists():
        return out
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # half-written last line after a crash
    return out


def jsonl_to_parquet(jsonl_path, parquet_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    recs = read_jsonl(jsonl_path)
    # de-duplicate on (uid, sample_idx): keep the first finished copy
    seen, uniq = set(), []
    for r in recs:
        key = (r["uid"], r["sample_idx"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)
    schema = pa.schema([
        ("uid", pa.string()), ("module", pa.string()), ("source", pa.string()),
        ("sample_idx", pa.int32()), ("seed", pa.int64()), ("prompt", pa.string()),
        ("prompt_token_ids", pa.list_(pa.int32())), ("completion", pa.string()),
        ("reasoning_content", pa.string()), ("completion_token_ids", pa.list_(pa.int32())),
        ("n_prompt_tokens", pa.int32()), ("n_completion_tokens", pa.int32()),
        ("finish_reason", pa.string()), ("stop_reason", pa.string()),
        ("t_start", pa.float64()), ("t_end", pa.float64()), ("latency_s", pa.float64()),
        ("run_id", pa.string()),
    ])
    def _v(r, c):
        v = r.get(c)
        if isinstance(v, float) and v != v:
            return None
        if c in _STR_COLS and v is not None and not isinstance(v, str):
            return str(v)
        return v
    cols = {c: [_v(r, c) for r in uniq] for c in RAW_COLUMNS}
    table = pa.table(cols, schema=schema)
    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, parquet_path, compression="zstd")
    return len(uniq)


def parquet_to_jsonl(parquet_path, jsonl_path):
    import pyarrow.parquet as pq
    t = pq.read_table(parquet_path).to_pylist()
    Path(jsonl_path).parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "a", encoding="utf-8") as f:
        for r in t:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return len(t)


# ===========================================================================
# 6. Derived statistics (computed from raw only; re-runnable any time)
# ===========================================================================
REFLECTION_PATTERNS = {
    "wait": r"\bwait\b",
    "but_wait": r"\bbut wait\b",
    "hmm": r"\bhmm+\b",
    "alternatively": r"\balternatively\b",
    "actually": r"\bactually\b",
    "let_me_check": r"\blet me (?:double[- ]?check|re-?check|check|verify|confirm|re-?examine|reconsider)\b",
    "mistake": r"\b(?:mistake|miscalculat\w*|made an error)\b",
    "i_think": r"\bi think\b",
}
_REFL = {k: re.compile(v, re.I) for k, v in REFLECTION_PATTERNS.items()}
_CJK = re.compile(r"[㐀-鿿豈-﫿]")


def split_thinking(text):
    """Qwen thinking output: '<reasoning> </think> <answer>' (the '<think>' is in
    the prompt). Returns (thinking, answer_part, tag_found)."""
    i = text.find("</think>")
    if i < 0:
        return text, "", False
    return text[:i], text[i + len("</think>"):], True


def repetition_stats(ids, tail=4096, n=32):
    ids = list(ids or [])
    L = len(ids)
    distinct4 = (len(set(zip(ids, ids[1:], ids[2:], ids[3:]))) / (L - 3)) if L > 3 else 1.0
    t = ids[-tail:]
    seen, rep, tot = set(), 0, 0
    for i in range(len(t) - n + 1):
        g = tuple(t[i:i + n])
        tot += 1
        if g in seen:
            rep += 1
        else:
            seen.add(g)
    return distinct4, (rep / tot if tot else 0.0)


def max_line_repeats(text, min_len=20):
    c = Counter(l.strip() for l in text.splitlines() if len(l.strip()) >= min_len)
    return max(c.values()) if c else 0


def lenient_correct(text_for_answer, gold):
    """Default math_verify extraction on the answer part (not just the last
    \\boxed) -- used to find cases the grader's parser misses."""
    try:
        g = parse(f"${gold}$")
        if not g:
            g = parse(gold)
        p = parse(text_for_answer)
        return bool(g and p and verify(g, p))
    except Exception:
        return False


def derive_row(rec, gold, think_end_id=None, loop_threshold=0.5):
    text = rec.get("completion") or ""
    ids = rec.get("completion_token_ids") or []
    n_tok = rec.get("n_completion_tokens") or len(ids)
    thinking, answer_part, tag = split_thinking(text)
    closed_ids = bool(think_end_id is not None and ids and think_end_id in ids)
    think_closed = tag or closed_ids
    if closed_ids:
        n_think = ids.index(think_end_id)
        n_answer = len(ids) - n_think - 1
    elif not think_closed:
        n_think, n_answer = n_tok, 0
    else:
        n_think, n_answer = None, None

    # --- grader view (content.strip(), last \boxed anywhere, grader's matcher)
    # If a server ever split the thinking into a separate field, rebuild the full
    # text the graders' vLLM 0.22 (no reasoning parser) would have returned.
    rc = rec.get("reasoning_content")
    if isinstance(rc, str) and rc.strip() and "</think>" not in text:
        text = rc + "</think>" + text
        thinking, answer_part, tag = split_thinking(text)
        think_closed = True
    graded_text = text.strip()
    pred = extract_boxed_answer(graded_text)
    correct_grader, pred_norm = False, None
    try:
        correct_grader, pred_norm, _, _, _ = answers_match(pred, gold)
    except Exception:
        correct_grader = False
    last_box = graded_text.rfind(r"\boxed")
    tpos = graded_text.find("</think>")
    pred_from_thinking = bool(pred is not None and (tpos < 0 or last_box < tpos))

    # --- lenient view
    lenient_src = answer_part if (tag and answer_part.strip()) else text
    correct_len = lenient_correct(lenient_src, gold)

    refl = {f"n_{k}": len(r.findall(thinking)) for k, r in _REFL.items()}
    per1k = 1000.0 / max(n_tok, 1)
    d4, tail_rep = repetition_stats(ids)
    out = dict(
        uid=rec["uid"], module=rec["module"], source=rec["source"],
        sample_idx=rec["sample_idx"], seed=rec.get("seed"),
        gold=gold, n_prompt_tokens=rec.get("n_prompt_tokens"),
        n_completion_tokens=n_tok, finish_reason=rec.get("finish_reason"),
        truncated=(rec.get("finish_reason") == "length"),
        think_closed=think_closed, n_think_tokens=n_think, n_answer_tokens=n_answer,
        n_boxed_total=text.count(r"\boxed"), n_boxed_after_think=answer_part.count(r"\boxed"),
        grader_pred=pred, grader_pred_norm=pred_norm, parsed=pred is not None,
        correct_grader=bool(correct_grader), correct_lenient=bool(correct_len),
        pred_from_thinking=pred_from_thinking,
        wait_per_1k=refl["n_wait"] * per1k,
        reflect_per_1k=sum(refl.values()) * per1k,
        distinct4_ratio=round(d4, 4), tail_repeat_frac=round(tail_rep, 4),
        looping=tail_rep >= loop_threshold, max_line_repeats=max_line_repeats(text),
        n_cjk_chars=len(_CJK.findall(text)), n_chars=len(text),
        latency_s=rec.get("latency_s"),
    )
    out.update(refl)
    return out


def _derive_worker(args):
    rec, gold, tid = args
    try:
        return derive_row(rec, gold, tid)
    except Exception as exc:
        return dict(uid=rec.get("uid"), module=rec.get("module"), source=rec.get("source"),
                    sample_idx=rec.get("sample_idx"), derive_error=f"{type(exc).__name__}: {exc}")


def derive_cli(argv):
    """python nb2lib.py derive <pool.parquet> <out.parquet> <think_end_id|-1> <workers> <raw files...>"""
    import multiprocessing as mp
    import pandas as pd
    pool_path, out_path, tid, workers, *raw_files = argv
    tid = int(tid)
    tid = None if tid < 0 else tid
    pool = pd.read_parquet(pool_path)
    gold = dict(zip(pool.uid, pool.answer))
    recs = []
    for f in raw_files:
        recs += read_jsonl(f) if f.endswith(".jsonl") else pd.read_parquet(f).to_dict("records")
    seen, jobs = set(), []
    for r in recs:
        key = (r["uid"], r["sample_idx"])
        if key in seen or r["uid"] not in gold:
            continue
        seen.add(key)
        if r.get("completion_token_ids") is not None:
            r["completion_token_ids"] = list(r["completion_token_ids"])
        jobs.append((r, gold[r["uid"]], tid))
    t0 = time.time()
    with mp.get_context("fork").Pool(int(workers)) as p:
        rows = p.map(_derive_worker, jobs, chunksize=4)
    df = pd.DataFrame(rows)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"derived {len(df)} samples in {time.time() - t0:.1f}s -> {out_path}")


# ===========================================================================
# 7. Summaries
# ===========================================================================
def summarize(df, by):
    """pass@1 with a problem-clustered 95% CI (mean over problems of the
    per-problem solve rate), plus failure-mode rates."""
    import numpy as np
    import pandas as pd
    rows = []
    for key, g in df.groupby(by, dropna=False):
        per_prob = g.groupby("uid")["correct_grader"].mean()
        n_p = len(per_prob)
        acc = per_prob.mean()
        se = per_prob.std(ddof=1) / np.sqrt(n_p) if n_p > 1 else float("nan")
        key = key if isinstance(key, tuple) else (key,)
        rows.append(dict(
            **dict(zip(by, key)), problems=n_p, samples=len(g),
            acc_grader=round(acc, 4), ci95=round(1.96 * se, 4),
            acc_lenient=round(g.groupby("uid")["correct_lenient"].mean().mean(), 4),
            pass_at_k=round(g.groupby("uid")["correct_grader"].max().mean(), 4),
            never_solved=int((per_prob == 0).sum()), always_solved=int((per_prob == 1).sum()),
            parse_rate=round(g["parsed"].mean(), 4),
            trunc_rate=round(g["truncated"].mean(), 4),
            ans_from_thinking=round(g["pred_from_thinking"].mean(), 4),
            loop_rate=round(g["looping"].mean(), 4),
            mean_tokens=int(g["n_completion_tokens"].mean()),
            med_tokens_correct=(int(g.loc[g.correct_grader, "n_completion_tokens"].median())
                                if g.correct_grader.any() else None),
            wait_per_1k=round(g["wait_per_1k"].mean(), 2),
        ))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "derive":
        derive_cli(sys.argv[2:])
    else:
        print(__doc__ or LIB_VERSION)
'''
    (WORK / "nb2lib.py").write_text(LIB_SRC, encoding="utf-8")
    if str(WORK) not in sys.path:
        sys.path.insert(0, str(WORK))


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # CS6013 · Notebook 2 — base-model training traces (Qwen3.5-4B) · v1.1

    Step 1 of the fine-tune-then-compress plan (plan A). Runs the **base** Qwen3.5-4B over a
    **training pool** with the **graders' exact settings** and keeps every trace. Later we keep
    the correct, finished, shortest traces per problem and fine-tune on them (self-distillation
    towards answering inside the 32K budget).

    | module | contents | samples / problem |
    |---|---|---|
    | **T** | HMMT February + November archive 1999-2024 (Easy2Hard-Bench E2H-AMC, both splits; MathArena HMMT Feb 2024), plus the AIME archive from E2H | 2, topped up to 4 where needed |

    Problems that already have base traces in nb1's repo (module A: AIME 2022-26; module B:
    HMMT Feb 25/26, HMMT Nov 25, BRUMO/CMIMC/SMT 25) are **not** re-run here; they are reused
    when the training set is assembled. Dropped from T: HMMT estimation problems (last guts
    set and "an estimate of E earns ... points" wording), guts problems that use another
    problem's answer, multi-value / ordered-pair answers (the grader's normaliser would merge
    "98,118" into one number), images, unparseable gold answers.

    **Grader settings (fixed, not knobs):** evalkit prompt template, chat template with
    thinking on, temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5,
    32,000 new tokens, max-model-len 33,000, `--language-model-only`, no reasoning parser.
    Sample *i* uses seed 42 + i. Only the server's memory share and parallel sequences are
    raised (speed only).

    **Order:** ① inputs → ② install → ③ model → ④ pool (check the table!) →
    ⑤ start server → ⑥ pilot (HMMT Feb 2024, ~30 × 4) → ⑦a first pass (2 / problem) →
    ⑦b preview top-up → ⑦c top-up to 4 (no usable sample, or shortest usable > 16K) → ⑧ analysis.
    Everything is resume-safe: a restarted session pulls finished samples back from the HF repo.
    Use a **new** HF dataset repo (not nb1's base-traces repo).
    """)
    return


@app.cell(hide_code=True)
def _():
    hf_token = mo.ui.text(kind="password", label="HF token (needs write access)", full_width=True)
    hf_repo = mo.ui.text(placeholder="grey-cat/cs6013-qwen35-4b-train-traces",
                         label="HF dataset repo (created private if missing)", full_width=True)
    model_id = mo.ui.text(value="Qwen/Qwen3.5-4B", label="Base model")
    vllm_spec = mo.ui.text(value="vllm>=0.30.0",
                           label="vLLM pip spec (0.22.0 crashes on molab: flash_attn bug)")
    gpu_util = mo.ui.number(start=0.30, stop=0.95, step=0.05, value=0.90,
                            label="--gpu-memory-utilization (grader 0.35)")
    max_seqs = mo.ui.number(start=16, stop=512, step=16, value=256,
                            label="--max-num-seqs = client concurrency (grader 50)")
    push_every = mo.ui.number(start=5, stop=180, step=5, value=30,
                              label="checkpoint push to HF every N minutes")
    mo.vstack([
        mo.md("## ① Inputs"),
        hf_token, hf_repo,
        mo.hstack([model_id, vllm_spec], justify="start"),
        mo.hstack([gpu_util, max_seqs, push_every], justify="start"),
        mo.md("Nothing here is saved to disk; the token is only passed to `huggingface_hub`."),
    ])
    return gpu_util, hf_repo, hf_token, max_seqs, model_id, push_every, vllm_spec


@app.cell(hide_code=True)
def _(hf_repo, hf_token):
    # ---------------- HF repo helpers (push / pull) ----------------
    def _api():
        from huggingface_hub import HfApi
        return HfApi(token=hf_token.value.strip() or None)

    def repo_ready():
        if hf_repo.value.strip().endswith("-base-traces"):
            raise ValueError("that is nb1's base-traces repo (local leaderboard source) - "
                             "use a new repo, e.g. grey-cat/cs6013-qwen35-4b-train-traces")
        return bool(hf_repo.value.strip()) and bool(hf_token.value.strip())

    def ensure_repo():
        if not repo_ready():
            print("  [hf] no repo/token given -> results stay local only")
            return False
        _api().create_repo(hf_repo.value.strip(), repo_type="dataset", private=True,
                           exist_ok=True)
        return True

    def repo_files():
        if not repo_ready():
            return []
        try:
            return _api().list_repo_files(hf_repo.value.strip(), repo_type="dataset")
        except Exception as exc:
            print(f"  [hf] cannot list repo: {type(exc).__name__}: {exc}")
            return []

    def hf_download(path_in_repo):
        from huggingface_hub import hf_hub_download
        return hf_hub_download(hf_repo.value.strip(), path_in_repo, repo_type="dataset",
                               token=hf_token.value.strip() or None,
                               local_dir=str(WORK / "hf_pull"))

    def push(patterns, message):
        """Upload files under WORK/stage matching patterns (paths mirror the repo)."""
        if not ensure_repo():
            return
        write_readme()
        _t = time.time()
        for _attempt in range(3):
            try:
                _api().upload_folder(repo_id=hf_repo.value.strip(), repo_type="dataset",
                                     folder_path=str(WORK / "stage"),
                                     allow_patterns=list(patterns) + ["README.md", "meta/*"],
                                     commit_message=message)
                print(f"  [hf] pushed {patterns} ({time.time() - _t:.0f}s)")
                return
            except Exception as exc:
                print(f"  [hf] push failed (attempt {_attempt + 1}): {type(exc).__name__}: {exc}")
                time.sleep(20)
        print("  [hf] giving up on this push; data is still safe on local disk")

    def stage_raw(module):
        """raw/<module>/*.jsonl -> stage/raw/<module>/*.parquet"""
        L = STATE["L"]
        _n = 0
        for _f in sorted((WORK / "raw" / module).glob("*.jsonl")):
            _n += L.jsonl_to_parquet(_f, WORK / "stage" / "raw" / module / (_f.stem + ".parquet"))
        return _n

    def sync_down(module):
        """Pull raw/<module>/*.parquet from HF and append any samples missing locally."""
        L = STATE["L"]
        _added = 0
        for _p in repo_files():
            if not (_p.startswith(f"raw/{module}/") and _p.endswith(".parquet")):
                continue
            _local = WORK / "raw" / module / (Path(_p).stem + ".jsonl")
            _have = {(r["uid"], r["sample_idx"]) for r in L.read_jsonl(_local)}
            _pd = importlib.import_module("pandas")
            _df = _pd.read_parquet(hf_download(_p))
            _new = _df[[(u, s) not in _have for u, s in zip(_df.uid, _df.sample_idx)]]
            if len(_new):
                _local.parent.mkdir(parents=True, exist_ok=True)
                with open(_local, "a", encoding="utf-8") as _fh:
                    for _r in _new.to_dict("records"):
                        for _c in ("prompt_token_ids", "completion_token_ids"):
                            if _r.get(_c) is not None:
                                _r[_c] = [int(x) for x in _r[_c]]
                        for _c, _v in list(_r.items()):
                            if isinstance(_v, float) and _v != _v:   # NaN -> None
                                _r[_c] = None
                        _fh.write(json.dumps(_r, ensure_ascii=False, default=str) + "\n")
                _added += len(_new)
        if _added:
            print(f"  [hf] restored {_added} finished samples of module {module} from the repo")
        return _added

    def write_readme():
        _m = STATE.get("meta") or {}
        _txt = f"""---
    license: cc-by-nc-sa-4.0
    tags: [math, reasoning, traces, qwen3.5]
    ---
    # Qwen3.5-4B base traces — CS6013 training pool (nb2)

    Generated by `{NB_VERSION}`. Private course material.

    ## Layout
    - `pool/pool.parquet` — every problem: uid, module, source, year, problem, gold answer, exact prompt, k.
    - `raw/<module>/<source>.parquet` — **one row per sample, never edited**: exact prompt,
      prompt + completion **token IDs**, full completion text (thinking included), finish_reason,
      seed, timing, run_id.
    - `derived/<module>.parquet` — statistics recomputed from `raw/` (correctness under the grader's
      parser and a lenient checker, truncation, thinking/answer tokens, reflection-word counts,
      repetition/loop scores, ...). Safe to regenerate.
    - `meta/run_meta_<run_id>.json` — model revision, vLLM/torch versions, GPU, server command,
      sampling settings.

    ## Generation settings (identical to the CS6013 grader)
    temperature 1.0, top_p 0.95, top_k 20, min_p 0, presence_penalty 1.5, repetition_penalty 1.0,
    max 32000 new tokens, thinking enabled, evalkit prompt template. Seed = 42 + sample_idx.
    Model: `{_m.get('model_id', '?')}` @ `{_m.get('model_revision', '?')}`, vLLM {_m.get('vllm_version', '?')}.

    Per-token log-probabilities were deliberately **not** stored: they can be recomputed exactly
    with one prefill pass over `prompt_token_ids + completion_token_ids`.

    Sources: Easy2Hard-Bench E2H-AMC (HMMT and AIME 1999-2024, CC BY-SA 4.0),
    MathArena HMMT February 2024 (CC BY-NC-SA 4.0). Used to fine-tune the base model.
    """
        _txt = "\n".join(l[4:] if l.startswith("    ") else l for l in _txt.splitlines())
        (WORK / "stage" / "README.md").write_text(_txt, encoding="utf-8")
        (WORK / "stage" / "meta").mkdir(parents=True, exist_ok=True)
        for _f in (WORK / "meta").glob("*.json"):
            shutil.copy(_f, WORK / "stage" / "meta" / _f.name)

    return hf_download, push, repo_files, stage_raw, sync_down


@app.cell(hide_code=True)
def _():
    install_btn = mo.ui.run_button(label="② Install packages")
    force_install = mo.ui.checkbox(label="force pip install even if already satisfied")
    mo.vstack([mo.md("## ② Install\nvLLM (+ matching torch), datasets, math-verify 0.9.0 "
                     "(grader's version), huggingface_hub, installed with **uv** (pip as fallback). "
                     "A few minutes the first time; "
                     "**skipped in seconds** if everything is already installed in this "
                     "environment (e.g. after re-uploading a new notebook version)."),
               mo.hstack([install_btn, force_install], justify="start")])
    return force_install, install_btn


@app.cell(hide_code=True)
def _(force_install, install_btn, vllm_spec):
    mo.stop(not install_btn.value, mo.md("*press ② to install*"))
    _pkgs = [p for p in [vllm_spec.value.strip() or "vllm", "datasets", "openai",
                         "math-verify==0.9.0", "huggingface_hub", "pyarrow", "pandas",
                         "matplotlib", "pyyaml", "tabulate"] if p]
    # Which requirements are not yet satisfied in this environment?
    _check = subprocess.run([PY, "-c", (
        "import sys, json, importlib.metadata as m\n"
        "try:\n"
        "    from packaging.requirements import Requirement\n"
        "except Exception:\n"
        "    print(json.dumps(sys.argv[1:])); raise SystemExit\n"
        "miss = []\n"
        "for r in sys.argv[1:]:\n"
        "    q = Requirement(r)\n"
        "    try:\n"
        "        v = m.version(q.name)\n"
        "    except m.PackageNotFoundError:\n"
        "        miss.append(r); continue\n"
        "    if q.specifier and not q.specifier.contains(v, prereleases=True):\n"
        "        miss.append(r)\n"
        "print(json.dumps(miss))"), *_pkgs], capture_output=True, text=True)
    try:
        _missing = json.loads(_check.stdout.strip().splitlines()[-1])
    except Exception:
        _missing = _pkgs
    if force_install.value:
        _missing = _pkgs
    if _missing:
        # uv (parallel downloads, fast resolver) into this exact interpreter;
        # plain pip is the automatic fallback if anything about uv fails.
        _t0 = time.time()
        _done = False
        _u = subprocess.run([PY, "-m", "pip", "install", "-q", "uv"],
                            capture_output=True, text=True)
        if _u.returncode == 0:
            print("uv pip install", " ".join(_missing), flush=True)
            _r = subprocess.run([PY, "-m", "uv", "pip", "install", "--python", PY, *_missing],
                                capture_output=True, text=True)
            print(_r.stdout[-3000:], _r.stderr[-3000:])
            _done = _r.returncode == 0
            if not _done:
                print("uv failed - falling back to pip", flush=True)
        else:
            print("could not install uv - using pip", _u.stderr[-500:], flush=True)
        if not _done:
            print("pip install", " ".join(_missing), flush=True)
            _r = subprocess.run([PY, "-m", "pip", "install", "-q", *_missing],
                                capture_output=True, text=True)
            print(_r.stdout[-3000:], _r.stderr[-3000:])
            if _r.returncode != 0:
                raise RuntimeError("pip install failed - see above")
        print(f"install took {(time.time() - _t0) / 60:.1f} min", flush=True)
    else:
        print("all requirements already satisfied - skipping pip", flush=True)
    _nb2lib = importlib.reload(importlib.import_module("nb2lib"))
    STATE["L"] = _nb2lib
    _v = subprocess.run([PY, "-c", "import vllm, torch; print('vllm', vllm.__version__, '| torch', "
                         "torch.__version__, '| cuda', torch.version.cuda)"],
                        capture_output=True, text=True)
    print(_v.stdout or _v.stderr[-2000:])
    mo.md(f"**Installed.** nb2lib `{_nb2lib.LIB_VERSION}` loaded. {_v.stdout.strip()}")
    return


@app.cell(hide_code=True)
def _():
    model_btn = mo.ui.run_button(label="③ Check GPU + download model")
    mo.vstack([mo.md("## ③ Model\nDownloads the base model (~9 GB) and records its exact "
                     "revision, the GPU, and library versions in `meta/run_meta_<run_id>.json`."),
               model_btn])
    return (model_btn,)


@app.cell(hide_code=True)
def _(hf_token, model_btn, model_id):
    mo.stop(not model_btn.value, mo.md("*press ③*"))
    mo.stop(STATE["L"] is None, mo.md("**Run ② first.**"))
    from huggingface_hub import HfApi as _HfApi, snapshot_download as _snap
    _gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                           "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    print("GPU:", _gpu or "NONE FOUND", flush=True)
    _tok = hf_token.value.strip() or None
    _rev = _HfApi(token=_tok).model_info(model_id.value.strip()).sha
    _path = _snap(model_id.value.strip(), revision=_rev, token=_tok)
    print("model at", _path, "revision", _rev, flush=True)
    _probe = subprocess.run([PY, "-c", (
        "import json, vllm, torch, transformers as t\n"
        f"tok = t.AutoTokenizer.from_pretrained({_path!r})\n"
        "tid = tok.convert_tokens_to_ids('</think>')\n"
        "print(json.dumps(dict(vllm=vllm.__version__, torch=torch.__version__, "
        "transformers=t.__version__, think_end_id=tid, "
        "think_end_special='</think>' in tok.all_special_tokens)))")],
        capture_output=True, text=True)
    try:
        _info = json.loads(_probe.stdout.strip().splitlines()[-1])
    except Exception:
        print(_probe.stderr[-3000:])
        raise
    _L = STATE["L"]
    STATE["model_path"] = _path
    STATE["meta"] = dict(
        run_id=RUN_ID, notebook=NB_VERSION, lib=_L.LIB_VERSION, model_id=model_id.value.strip(),
        model_revision=_rev, gpu=_gpu, vllm_version=_info["vllm"], torch_version=_info["torch"],
        transformers_version=_info["transformers"], think_end_id=_info["think_end_id"],
        think_end_is_special=_info["think_end_special"], sampling=_L.SAMPLING,
        seed_rule="seed = 42 + sample_idx", max_model_len=_L.MAX_MODEL_LEN,
        math_instruction=_L.MATH_INSTRUCTION, module_k=_L.MODULE_K,
        created=datetime.now(timezone.utc).isoformat(),
    )
    (WORK / "meta" / f"run_meta_{RUN_ID}.json").write_text(json.dumps(STATE["meta"], indent=2))
    mo.md(f"**GPU:** {_gpu}  \n**Model:** `{model_id.value}` @ `{_rev[:12]}`  \n"
          f"vLLM {_info['vllm']}, torch {_info['torch']}, `</think>` id = {_info['think_end_id']}"
          f"{' (special token!)' if _info['think_end_special'] else ''}")
    return


@app.cell(hide_code=True)
def _():
    pool_btn = mo.ui.run_button(label="④ Build / load pool")
    force_rebuild = mo.ui.checkbox(label="force rebuild (only before any inference has run!)")
    mo.vstack([mo.md("## ④ Pool\nLoads the T sources (and nb1's A/B sources, for "
                     "de-duplication only), drops estimation / cross-reference / multi-value / "
                     "image / unscorable problems and freezes the result as `pool/pool.parquet`. "
                     "**Once it is in the HF repo it is reused as-is**, so every later run sees "
                     "the identical problem set. CPU only. **Check the load report and spot-check "
                     "the browse table** (a few problems + gold answers per source) before ⑤."),
               force_rebuild, pool_btn])
    return force_rebuild, pool_btn


@app.cell(hide_code=True)
def _(force_rebuild, hf_download, pool_btn, push, repo_files):
    mo.stop(not pool_btn.value, mo.md("*press ④*"))
    mo.stop(STATE["L"] is None, mo.md("**Run ② first.**"))
    _pd = importlib.import_module("pandas")
    _L = STATE["L"]
    _local = WORK / "pool" / "pool.parquet"
    _origin = None
    if not force_rebuild.value and _local.exists():
        _origin = "local disk"
    elif not force_rebuild.value and "pool/pool.parquet" in repo_files():
        shutil.copy(hf_download("pool/pool.parquet"), _local)
        _origin = "HF repo"
    if _origin is None:
        from datasets import load_dataset as _load_dataset
        _rows, _report = _L.build_train_pool(_load_dataset)
        _pd.DataFrame(_rows).to_parquet(_local, index=False)
        _pd.DataFrame(_report).to_csv(WORK / "pool" / "pool_report.csv", index=False)
        _origin = "freshly built"
    _pool = _pd.read_parquet(_local)
    STATE["pool"] = _pool
    (WORK / "stage" / "pool").mkdir(parents=True, exist_ok=True)
    for _f in (WORK / "pool").iterdir():
        shutil.copy(_f, WORK / "stage" / "pool" / _f.name)
    if _origin == "freshly built":
        push(["pool/*"], "pool: build")

    _by = (_pool.groupby(["module", "source"], sort=False)
           .agg(problems=("uid", "size"), k=("k", "first"),
                gold_parse_ok=("gold_parse_ok", "mean")).reset_index())
    _by["samples"] = _by.problems * _by.k
    _tot = _by.groupby("module", sort=False)[["problems", "samples"]].sum().reset_index()
    _report_md = ""
    if (WORK / "pool" / "pool_report.csv").exists():
        _rep = _pd.read_csv(WORK / "pool" / "pool_report.csv").fillna("")
        _report_md = "### Load report (drops per source)\n" + _rep.to_markdown(index=False)
    mo.vstack([
        mo.md(f"**Pool ({_origin}):** {len(_pool)} problems, {int(_by.samples.sum())} samples."),
        mo.md("### Per module\n" + _tot.to_markdown(index=False)),
        mo.md("### Per source\n" + _by.to_markdown(index=False)),
        mo.md(_report_md),
        mo.md("### Browse (spot-check a few problems and gold answers per source)"),
        mo.md("`level` = Easy2Hard difficulty rating (0 easy - 1 hard); empty for MathArena."),
        mo.ui.table(_pool[["uid", "source", "year", "topic", "level", "answer", "problem"]],
                    page_size=10),
    ])
    return


@app.cell(hide_code=True)
def _():
    start_btn = mo.ui.run_button(label="⑤ Start vLLM server")
    stop_btn = mo.ui.run_button(label="stop server")
    log_btn = mo.ui.run_button(label="show server log tail")
    mo.vstack([mo.md("## ⑤ vLLM server\nThe graders' command (bf16, max-model-len 33000, "
                     "`--language-model-only`, no reasoning parser) with only memory and "
                     "parallel sequences raised. First start compiles kernels: 3-10 min."),
               mo.hstack([start_btn, stop_btn, log_btn], justify="start")])
    return log_btn, start_btn, stop_btn


@app.cell(hide_code=True)
def _(gpu_util, max_seqs, start_btn):
    mo.stop(not start_btn.value, mo.md("*press ⑤*"))
    mo.stop(STATE["model_path"] is None, mo.md("**Run ③ first.**"))
    _L = STATE["L"]
    if STATE["vllm_proc"] is not None and STATE["vllm_proc"].poll() is None:
        print("server already running on port", STATE["port"])
    else:
        _s = socket.socket(); _s.bind(("", 0)); _port = _s.getsockname()[1]; _s.close()
        _cmd = [PY, "-m", "vllm.entrypoints.openai.api_server",
                "--model", STATE["model_path"],
                "--served-model-name", _L.SERVED_MODEL_NAME,
                "--tensor-parallel-size", "1",
                "--dtype", "bfloat16",
                "--max-model-len", str(_L.MAX_MODEL_LEN),
                "--port", str(_port),
                "--gpu-memory-utilization", f"{float(gpu_util.value):.2f}",
                "--max-num-seqs", str(int(max_seqs.value))]
        _help = subprocess.run([PY, "-m", "vllm.entrypoints.openai.api_server", "--help"],
                               capture_output=True, text=True)
        if "--language-model-only" in (_help.stdout + _help.stderr):
            _cmd.append("--language-model-only")
        else:
            print("note: this vLLM has no --language-model-only; omitting it")
        # molab has no nvcc -> FlashInfer's JIT sampler would fail (same as evalkit)
        _env = {**os.environ, "VLLM_USE_FLASHINFER_SAMPLER": "0"}
        print(" ".join(_cmd), flush=True)
        _log = open(WORK / "vllm.log", "w")
        _proc = subprocess.Popen(_cmd, stdout=_log, stderr=subprocess.STDOUT, env=_env,
                                 start_new_session=True)
        STATE.update(vllm_proc=_proc, port=_port)
        if STATE["meta"] is not None:
            STATE["meta"]["server_cmd"] = " ".join(_cmd[1:])
            (WORK / "meta" / f"run_meta_{RUN_ID}.json").write_text(json.dumps(STATE["meta"], indent=2))
        _deadline = time.time() + 1800
        while True:
            if _proc.poll() is not None:
                _txt = (WORK / "vllm.log").read_text()
                _err = [l for l in _txt.splitlines()
                        if any(k in l for k in ("Error", "ERROR", "error:", "CUDA", "OOM",
                                                "memory", "Memory"))]
                print("---- error lines from vllm.log (root cause is usually the last "
                      "'...Error: ...' line) ----")
                print("\n".join(_err[-60:]))
                print("---- last 40 log lines ----")
                print("\n".join(_txt.splitlines()[-40:]))
                raise RuntimeError("vLLM exited before becoming ready - see root cause above")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{_port}/v1/models", timeout=5)
                break
            except Exception:
                if time.time() > _deadline:
                    raise RuntimeError("vLLM not ready after 30 min")
                time.sleep(10)
        print("server ready on port", _port)
    mo.md(f"**Server up** on port {STATE['port']}.")
    return


@app.cell(hide_code=True)
def _(stop_btn):
    mo.stop(not stop_btn.value)
    _p = STATE.get("vllm_proc")
    if _p is not None and _p.poll() is None:
        os.killpg(os.getpgid(_p.pid), 15)
        try:
            _p.wait(timeout=60)
        except Exception:
            os.killpg(os.getpgid(_p.pid), 9)
    STATE.update(vllm_proc=None, port=None)
    mo.md("server stopped")
    return


@app.cell(hide_code=True)
def _(log_btn):
    mo.stop(not log_btn.value)
    _f = WORK / "vllm.log"
    mo.md("```\n" + (_f.read_text()[-6000:] if _f.exists() else "no log yet") + "\n```")
    return


@app.cell(hide_code=True)
def _(max_seqs, push, push_every, stage_raw, sync_down):
    # ---------------- the inference loop ----------------
    def run_jobs(module, sources=None, uids=None, max_k=None):
        """Generate every missing (problem, sample) of `module` (optionally only
        `sources`). Each finished sample is appended to raw/<module>/<source>.jsonl
        immediately; the module is pushed to HF every `push_every` minutes."""
        L, pool = STATE["L"], STATE["pool"]
        assert pool is not None, "run ④ first"
        assert STATE["vllm_proc"] is not None and STATE["vllm_proc"].poll() is None, \
            "server not running - run ⑤"
        from openai import OpenAI
        sync_down(module)
        sub = pool[pool.module == module]
        if sources:
            sub = sub[sub.source.isin(sources)]
        if uids is not None:
            sub = sub[sub.uid.isin(set(uids))]
        rawdir = WORK / "raw" / module
        rawdir.mkdir(parents=True, exist_ok=True)
        done = set()
        for f in rawdir.glob("*.jsonl"):
            done |= {(r["uid"], r["sample_idx"]) for r in L.read_jsonl(f)}
        # sample 0 of every problem first, then sample 1, ... so a partial run
        # still covers every problem
        rows = list(sub.itertuples(index=False))
        # max_k caps the samples per problem for this pass (adaptive sampling);
        # seeds stay 42 + sample_idx, so a later top-up continues the same series
        cap = lambda r: r.k if max_k is None else min(r.k, int(max_k))
        jobs = [(r, s) for s in range(int(sub.k.max()) if len(sub) else 0)
                for r in rows if s < cap(r) and (r.uid, s) not in done]
        total = len(jobs)
        print(f"module {module}{' ' + str(sources) if sources else ''}"
              f"{'' if max_k is None else f' (samples 0..{int(max_k) - 1})'}: "
              f"{len(sub)} problems, {total} samples to generate "
              f"({len(done)} already done)", flush=True)
        if not total:
            return dict(module=module, generated=0, failed=0, seconds=0, tokens=0)

        client = OpenAI(base_url=f"http://127.0.0.1:{STATE['port']}/v1", api_key="EMPTY",
                        timeout=7200, max_retries=0)

        def call(r, s):
            for attempt in range(3):
                try:
                    return L.request_one(client, r.prompt, L.GRADER_SEED + s)
                except Exception as exc:
                    if attempt == 2 or STATE["vllm_proc"].poll() is not None:
                        raise
                    time.sleep(15 * (attempt + 1))

        handles = {}
        def fh(source):
            if source not in handles:
                handles[source] = open(rawdir / f"{source}.jsonl", "a", encoding="utf-8")
            return handles[source]

        t0 = last_print = last_push = time.time()
        n_ok = n_fail = consec_fail = tok = n_trunc = 0
        ex = ThreadPoolExecutor(max_workers=int(max_seqs.value))
        try:
            if True:
                futs = {ex.submit(call, r, s): (r, s) for r, s in jobs}
                for fut in as_completed(futs):
                    r, s = futs[fut]
                    try:
                        out = fut.result()
                    except Exception as exc:
                        n_fail += 1
                        consec_fail += 1
                        print(f"  FAILED {r.uid} #{s}: {type(exc).__name__}: {str(exc)[:200]}",
                              flush=True)
                        if consec_fail >= 20:
                            print("  20 consecutive failures - aborting (server dead?)", flush=True)
                            break
                        continue
                    consec_fail = 0
                    rec = dict(uid=r.uid, module=module, source=r.source, sample_idx=s,
                               seed=L.GRADER_SEED + s, prompt=r.prompt, run_id=RUN_ID, **out)
                    f = fh(r.source)
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f.flush()
                    n_ok += 1
                    tok += out["n_completion_tokens"] or 0
                    n_trunc += out["finish_reason"] == "length"
                    now = time.time()
                    if now - last_print > 60 or n_ok == total:
                        el = now - t0
                        rate = tok / el
                        eta = (total - n_ok - n_fail) * (tok / max(n_ok, 1)) / max(rate, 1e-9)
                        print(f"  [{time.strftime('%H:%M:%S')}] {n_ok}/{total} done, "
                              f"{n_fail} failed | {rate:,.0f} tok/s | mean "
                              f"{tok / n_ok:,.0f} tok | truncated {n_trunc / n_ok:.1%} | "
                              f"elapsed {el / 60:.0f} min | ETA ~{eta / 3600:.1f} h "
                              f"(early ETAs run low: short answers finish first)", flush=True)
                        last_print = now
                    if now - last_push > 60 * float(push_every.value):
                        for h in handles.values():
                            h.flush()
                        stage_raw(module)
                        push([f"raw/{module}/*"], f"raw/{module}: checkpoint {n_ok}/{total}")
                        last_push = time.time()
        finally:
            # On interrupt: drop queued jobs and close the HTTP client, which makes
            # vLLM abort the in-flight requests (their samples are simply re-run
            # next time). Never wait here - in-flight requests can take an hour.
            ex.shutdown(wait=False, cancel_futures=True)
            try:
                client.close()
            except Exception:
                pass
            for h in handles.values():
                h.close()
        el = time.time() - t0
        stage_raw(module)
        push([f"raw/{module}/*"], f"raw/{module}: {n_ok} new samples (run {RUN_ID})")
        print(f"module {module}: {n_ok} generated, {n_fail} failed, {el / 60:.1f} min, "
              f"{tok / max(el, 1):,.0f} tok/s", flush=True)
        return dict(module=module, generated=n_ok, failed=n_fail, seconds=el, tokens=tok)

    def derive(module):
        """raw/<module>/*.jsonl -> derived/<module>.parquet (subprocess), then push."""
        L, meta = STATE["L"], STATE["meta"] or {}
        files = sorted(str(f) for f in (WORK / "raw" / module).glob("*.jsonl"))
        if not files:
            print(f"module {module}: no raw data yet")
            return None
        out = WORK / "derived" / f"{module}.parquet"
        tid = meta.get("think_end_id")
        tid = -1 if tid is None else int(tid)
        workers = max(2, min(16, (os.cpu_count() or 4) - 1))
        r = subprocess.run([PY, str(WORK / "nb2lib.py"), "derive",
                            str(WORK / "pool" / "pool.parquet"), str(out), str(tid),
                            str(workers), *files], capture_output=True, text=True)
        print(r.stdout[-2000:], r.stderr[-2000:] if r.returncode else "")
        if r.returncode:
            raise RuntimeError("derive failed")
        (WORK / "stage" / "derived").mkdir(parents=True, exist_ok=True)
        shutil.copy(out, WORK / "stage" / "derived" / out.name)
        push([f"derived/{module}.parquet"], f"derived/{module}")
        import pandas as pd
        return pd.read_parquet(out)

    return derive, run_jobs


@app.cell(hide_code=True)
def _():
    pilot_btn = mo.ui.run_button(label="⑥ Run pilot (HMMT Feb 2024, ~30 × 4)")
    mo.vstack([mo.md("## ⑥ Pilot\nMeasures throughput, truncation at 32K and a time estimate "
                     "for module T before we commit hours of GPU. The pilot samples are stored "
                     "as part of module T and will not be re-run. Only ~120 requests are in "
                     "flight, so full-module throughput should be higher. If MathArena's "
                     "HMMT Feb 2024 is not in the pool, 30 random E2H HMMT-Feb problems are "
                     "used instead."),
               pilot_btn])
    return (pilot_btn,)


@app.cell(hide_code=True)
def _(derive, pilot_btn, run_jobs):
    mo.stop(not pilot_btn.value, mo.md("*press ⑥*"))
    _pd = importlib.import_module("pandas")
    _L = STATE["L"]
    _pool = STATE["pool"]
    mo.stop(_pool is None, mo.md("**Run ④ first.**"))
    _sub = _pool[_pool.source == "hmmt_feb_2024"]
    if len(_sub) < 10:
        _sub = _pool[_pool.source == "e2h_hmmt_feb"].sample(n=min(30, int((_pool.source == "e2h_hmmt_feb").sum())),
                                                          random_state=0)
    _uids = list(_sub.uid)
    _res = run_jobs("T", uids=_uids)
    _d = derive("T")
    mo.stop(_d is None or not _d.uid.isin(_uids).any(),
            mo.md("**No pilot samples were generated** - check the server log (⑤)."))
    _d = _d[_d.uid.isin(_uids)]
    _sum = _L.summarize(_d, ["source"])
    _pp = (_d.groupby("uid").agg(solve=("correct_grader", "mean"),
                                 mean_tokens=("n_completion_tokens", "mean"),
                                 trunc=("truncated", "mean")).reset_index())
    _tps = _res["tokens"] / max(_res["seconds"], 1) if _res["generated"] else float("nan")
    _mean = _d.n_completion_tokens.mean()
    # adaptive plan: first pass 2 / problem, then top-up to 4 for the selected share.
    # The top-up share is what the 16K rule would pick on the pilot's first 2 samples;
    # the pilot is harder than the archive, so it is an upper bound.
    _sel = _L.topup_select(_d[_d.sample_idx < 2], first_pass_k=2, max_k=4, long_tokens=16000)
    _share = float(_sel.selected.mean()) if len(_sel) else float("nan")
    _n_prob = len(_pool)
    _first_left = 2 * _n_prob - int((_d.sample_idx < 2).sum())
    _rows = []
    for _label, _f, _sh in (("upper (pilot-like)", 1.0, _share),
                            ("lower (archive 0.6x length, half the top-up)", 0.6, _share / 2)):
        _samples = _first_left + _sh * _n_prob * 2
        _toks = _samples * _mean * _f
        _rows.append(dict(guess=_label, assumed_mean_tokens=int(_mean * _f),
                          first_pass_samples=_first_left, topup_share=round(_sh, 3),
                          total_samples=int(_samples), total_Mtok=round(_toks / 1e6, 1),
                          hours_at_pilot_rate=round(_toks / _tps / 3600, 1) if _tps == _tps else None))
    mo.vstack([
        mo.md(f"### Pilot result\nThroughput during pilot: **{_tps:,.0f} tok/s** "
              f"({_res['generated']} new samples in {_res['seconds'] / 60:.1f} min)."),
        mo.md(_sum.T.to_markdown()),
        mo.md("### Per problem\n" + _pp.round(3).to_markdown(index=False)),
        mo.md(f"### Rough time for the rest of module T: first pass 2 / problem + top-up to 4\n"
              f"On the pilot's first 2 samples the 16K rule would top up **{_share:.0%}** of "
              f"problems (upper bound for the archive).\n\n"
              + _pd.DataFrame(_rows).to_markdown(index=False)
              + "\n\nFull-module throughput is usually higher than the pilot's (more requests "
                "in flight), so the real time tends to be lower."),
    ])
    return


@app.cell(hide_code=True)
def _():
    pass_k = mo.ui.number(start=1, stop=4, step=1, value=2,
                          label="samples per problem in this pass")
    run_btn = mo.ui.run_button(label="⑦a Run first pass")
    mo.vstack([mo.md("## ⑦ Module T — adaptive sampling\n"
                     "**⑦a first pass:** samples 0..(n-1) for every problem (default 2). Sample 0 "
                     "of every problem first, then sample 1, so a partial run still covers every "
                     "problem. Safe to stop and re-press: finished samples are skipped, a "
                     "checkpoint is pushed to HF every N minutes (①), and at the end the module "
                     "is pushed and its stats derived. **Keep the tab active** - molab stops "
                     "idle sessions.\n\n"
                     "**⑦b/⑦c top-up** (after ⑦a has finished): samples up to 4 only for problems "
                     "with no usable sample (correct and finished < 32K) or whose shortest usable "
                     "sample is long."),
               mo.hstack([pass_k, run_btn], justify="start")])
    return pass_k, run_btn


@app.cell(hide_code=True)
def _(derive, pass_k, run_btn, run_jobs):
    mo.stop(not run_btn.value, mo.md("*press ⑦a*"))
    _res = run_jobs("T", max_k=int(pass_k.value))
    _d = derive("T")
    _L = STATE["L"]
    mo.vstack([
        mo.md(f"### Module T first pass ({int(pass_k.value)} / problem)\n{_res['generated']} new "
              f"samples, {_res['failed']} failed, {_res['seconds'] / 3600:.2f} h, "
              f"{_res['tokens'] / max(_res['seconds'], 1):,.0f} tok/s."),
        mo.md(_L.summarize(_d, ["module", "source"]).to_markdown(index=False)
              if _d is not None else "no data"),
    ])
    return


@app.cell(hide_code=True)
def _():
    long_tok = mo.ui.number(start=4000, stop=32000, step=1000, value=16000,
                            label="'long' = shortest usable sample above (tokens)")
    topup_k = mo.ui.number(start=2, stop=4, step=1, value=4, label="top up to samples / problem")
    preview_btn = mo.ui.run_button(label="⑦b Preview top-up set (CPU)")
    topup_btn = mo.ui.run_button(label="⑦c Run top-up (GPU)")
    mo.vstack([mo.md("### Top-up\n⑦b syncs and re-derives module T, then shows which problems "
                     "would be topped up and how many samples that is. ⑦c runs exactly the set "
                     "⑦b showed (press ⑦b again after changing the settings)."),
               mo.hstack([long_tok, topup_k], justify="start"),
               mo.hstack([preview_btn, topup_btn], justify="start")])
    return long_tok, preview_btn, topup_btn, topup_k


@app.cell(hide_code=True)
def _(derive, long_tok, pass_k, preview_btn, sync_down, topup_k):
    mo.stop(not preview_btn.value, mo.md("*press ⑦b after ⑦a has finished*"))
    mo.stop(STATE["pool"] is None, mo.md("**Run ② and ④ first.**"))
    _pd = importlib.import_module("pandas")
    _L = STATE["L"]
    sync_down("T")
    _d = derive("T")
    mo.stop(_d is None, mo.md("no module-T samples yet"))
    _sel = _L.topup_select(_d, first_pass_k=int(pass_k.value), max_k=int(topup_k.value),
                           long_tokens=int(long_tok.value))
    _n_pool = len(STATE["pool"])
    _missing = _n_pool - len(_sel)
    STATE["topup"] = dict(uids=list(_sel[_sel.selected].uid), max_k=int(topup_k.value),
                          settings=(int(pass_k.value), int(topup_k.value), int(long_tok.value)))
    _by = (_sel.groupby(["source", "reason"]).size().unstack(fill_value=0).reset_index())
    _tot = dict(problems_in_pool=_n_pool, with_samples=len(_sel),
                not_started=_missing,
                first_pass_incomplete=int((_sel.reason == "first_pass_incomplete").sum()),
                selected=int(_sel.selected.sum()), new_samples=int(_sel.new_samples.sum()),
                share_selected=round(_sel.selected.mean(), 3) if len(_sel) else None)
    _warn = ("" if not (_missing or _tot["first_pass_incomplete"]) else
             f"\n\n**Warning:** {_missing} problems have no samples and "
             f"{_tot['first_pass_incomplete']} have an incomplete first pass - they are not "
             "selected. Finish ⑦a first, then preview again.")
    mo.vstack([
        mo.md("### Top-up preview\n" + _pd.DataFrame([_tot]).to_markdown(index=False) + _warn),
        mo.md("### Per source × reason\n`done` = has a usable sample ≤ threshold; "
              "`no_usable` / `long_only` = selected\n\n" + _by.to_markdown(index=False)),
        mo.ui.table(_sel.sort_values(["selected", "shortest_usable"], ascending=[False, False]),
                    page_size=10),
    ])
    return


@app.cell(hide_code=True)
def _(derive, long_tok, pass_k, run_jobs, topup_btn, topup_k):
    mo.stop(not topup_btn.value, mo.md("*press ⑦c after checking the ⑦b preview*"))
    _t = STATE.get("topup")
    mo.stop(_t is None, mo.md("**Press ⑦b first.**"))
    mo.stop(_t["settings"] != (int(pass_k.value), int(topup_k.value), int(long_tok.value)),
            mo.md("**Settings changed since the preview - press ⑦b again.**"))
    mo.stop(not _t["uids"], mo.md("Nothing to top up."))
    _res = run_jobs("T", uids=_t["uids"], max_k=_t["max_k"])
    _d = derive("T")
    _L = STATE["L"]
    mo.vstack([
        mo.md(f"### Top-up done\n{len(_t['uids'])} problems, {_res['generated']} new samples, "
              f"{_res['failed']} failed, {_res['seconds'] / 3600:.2f} h, "
              f"{_res['tokens'] / max(_res['seconds'], 1):,.0f} tok/s."),
        mo.md(_L.summarize(_d, ["module", "source"]).to_markdown(index=False)
              if _d is not None else "no data"),
    ])
    return


@app.cell(hide_code=True)
def _():
    derive_btn = mo.ui.run_button(label="⑧a Re-derive stats for all modules")
    analyse_btn = mo.ui.run_button(label="⑧b Show analysis")
    mo.vstack([mo.md("## ⑧ Analysis\n⑧a recomputes `derived/` from the raw traces (use it "
                     "after changing nb2lib's stats). ⑧b reads `derived/` and shows the tables."),
               mo.hstack([derive_btn, analyse_btn], justify="start")])
    return analyse_btn, derive_btn


@app.cell(hide_code=True)
def _(derive, derive_btn, sync_down):
    mo.stop(not derive_btn.value)
    mo.stop(STATE["pool"] is None, mo.md("**Run ② and ④ first.**"))
    for _m in STATE["L"].MODULES:
        sync_down(_m)
        derive(_m)
    mo.md("re-derived all modules")
    return


@app.cell(hide_code=True)
def _(analyse_btn, hf_download, repo_files):
    mo.stop(not analyse_btn.value)
    mo.stop(STATE["pool"] is None, mo.md("**Run ② and ④ first.**"))
    import matplotlib as _matplotlib
    _matplotlib.use("Agg")
    import matplotlib.pyplot as _plt
    _pd = importlib.import_module("pandas")
    _L = STATE["L"]
    _pool = STATE["pool"]
    _parts = []
    _files = set(repo_files())
    for _m in _L.MODULES:
        _f = WORK / "derived" / f"{_m}.parquet"
        if not _f.exists() and f"derived/{_m}.parquet" in _files:
            shutil.copy(hf_download(f"derived/{_m}.parquet"), _f)
        if _f.exists():
            _parts.append(_pd.read_parquet(_f))
    mo.stop(not _parts, mo.md("no derived data yet"))
    _der = _pd.concat(_parts, ignore_index=True)
    _der = _der.merge(_pool[["uid", "year"]], on="uid", how="left")

    _by_mod = _L.summarize(_der, ["module"])
    _by_src = _L.summarize(_der, ["module", "source"])
    _per_problem = (_der.groupby(["module", "source", "uid"])
                   .agg(k=("sample_idx", "size"), solve=("correct_grader", "mean"),
                        solve_lenient=("correct_lenient", "mean"),
                        mean_tokens=("n_completion_tokens", "mean"),
                        trunc=("truncated", "mean"), wait_per_1k=("wait_per_1k", "mean"))
                   .reset_index())
    _band = (_per_problem.assign(band=_pd.cut(_per_problem.solve, [-0.01, 0.0, 0.29, 0.8, 0.99, 1.0],
                                            labels=["never", "<30%", "30-80%", "80-99%", "always"]))
             .groupby(["module", "source", "band"], observed=False).size().unstack(fill_value=0)
             .reset_index())

    _BLUE, _ORANGE = "#2a78d6", "#eb6834"
    _mods = [m for m in _L.MODULES if m in set(_der.module)]
    _fig, _ax = _plt.subplots(2, len(_mods), figsize=(3.2 * len(_mods), 5.6), squeeze=False)
    for _i, _m in enumerate(_mods):
        _pp = _per_problem[_per_problem.module == _m]
        _a = _ax[0][_i]
        _a.hist(_pp.solve, bins=[-0.001, 0.125, 0.375, 0.625, 0.875, 1.001], color=_BLUE,
                edgecolor="white", linewidth=2)
        _a.set_title(f"{_m}: per-problem solve rate", fontsize=9)
        _a.set_xlabel("solve rate", fontsize=8)
        _a.set_ylabel("problems" if _i == 0 else "", fontsize=8)
        _dd = _der[_der.module == _m]
        _b = _ax[1][_i]
        _bins = list(range(0, 33001, 2000))
        _b.hist(_dd[_dd.correct_grader].n_completion_tokens, bins=_bins, color=_BLUE, alpha=0.85,
                label="correct", edgecolor="white", linewidth=1)
        _b.hist(_dd[~_dd.correct_grader].n_completion_tokens, bins=_bins, color=_ORANGE, alpha=0.6,
                label="wrong", edgecolor="white", linewidth=1)
        _b.set_title(f"{_m}: output tokens", fontsize=9)
        _b.set_xlabel("tokens (32K = cut off)", fontsize=8)
        if _i == 0:
            _b.legend(fontsize=8, frameon=False)
    for _a in _ax.flat:
        _a.spines[["top", "right"]].set_visible(False)
        _a.tick_params(labelsize=7)
        _a.grid(axis="y", alpha=0.25)
    _fig.tight_layout()

    # training yield: what self-distillation can use (correct under the grader's
    # parser AND finished before the cap), and how short the shortest one is
    _ok = _der[_der.correct_grader & ~_der.truncated]
    _short = (_ok.groupby(["source", "uid"]).n_completion_tokens.min()
              .rename("shortest").reset_index())
    _allp = _per_problem[["source", "uid", "mean_tokens"]].merge(_short, on=["source", "uid"],
                                                                 how="left")
    _yield = (_allp.groupby("source")
              .agg(problems=("uid", "size"),
                   with_correct=("shortest", lambda x: int(x.notna().sum())),
                   shortest_median=("shortest", "median"),
                   all_samples_mean_tokens=("mean_tokens", "mean"))
              .reset_index())
    _yield["with_correct_share"] = (_yield.with_correct / _yield.problems).round(3)
    _tot_y = dict(source="ALL", problems=len(_allp), with_correct=int(_allp.shortest.notna().sum()),
                  shortest_median=_allp.shortest.median(),
                  all_samples_mean_tokens=_allp.mean_tokens.mean())
    _tot_y["with_correct_share"] = round(_tot_y["with_correct"] / max(_tot_y["problems"], 1), 3)
    _yield = _pd.concat([_yield, _pd.DataFrame([_tot_y])], ignore_index=True)
    for _c in ("shortest_median", "all_samples_mean_tokens"):
        _yield[_c] = _yield[_c].round(0)
    _aime_md = _yield.to_markdown(index=False)
    STATE["yield"] = _yield
    _dis = _der[_der.correct_grader != _der.correct_lenient][
        ["uid", "sample_idx", "gold", "grader_pred", "correct_grader", "correct_lenient",
         "truncated", "pred_from_thinking"]]
    _refl = _der.groupby(["module", "correct_grader"])[
        [c for c in _der.columns if c.startswith("n_") and c not in
         ("n_prompt_tokens", "n_completion_tokens", "n_think_tokens", "n_answer_tokens",
          "n_boxed_total", "n_boxed_after_think", "n_chars", "n_cjk_chars")]].mean().round(1)
    STATE["per_problem"] = _per_problem
    STATE["derived"] = _der
    mo.vstack([
        mo.md("### Accuracy per module (pass@1 = mean over problems; ±95% CI clustered by problem)"),
        mo.md(_by_mod.to_markdown(index=False)),
        mo.md("### Per source"), mo.md(_by_src.to_markdown(index=False)),
        mo.md("### Training yield per source\n`with_correct` = problems with at least one "
              "sample that is correct (grader's parser) and finished before 32K; "
              "`shortest_median` = median over those problems of their shortest such sample "
              "(tokens) - the lengths self-distillation would train on."),
        mo.md(_aime_md),
        mo.md("### How many problems fall in each solve-rate band"),
        mo.md(_band.to_markdown(index=False)),
        _fig,
        mo.md("### Reflection words per trace, correct vs wrong (thinking part)"),
        mo.md(_refl.reset_index().to_markdown(index=False)),
        mo.md(f"### Grader vs lenient checker disagreements: {len(_dis)} samples "
              "(grader-wrong / lenient-right = parser or format losses)"),
        mo.ui.table(_dis, page_size=10),
        mo.md("### Per problem (sortable)"), mo.ui.table(_per_problem, page_size=15),
    ])
    return


if __name__ == "__main__":
    app.run()
