# CS6013 — Compression 10 — fine-tuned model + P3-lite + decision-token-weighted Fisher

Source model: Qwen3.5-4B after self-distillation fine-tuning (LoRA on its own shortest
correct solutions, merged to bf16; `local_eval/nb3_sft.py`), compressed with codec E +
decision-token-weighted Fisher allocation (x5) + GPTQ + rANS + MLP neuron pruning +
split-precision embedding. Optional size-neutral recovery: `recovery/README.md`.

## Precision

| | dtype |
|---|---|
| Base model | `torch.bfloat16` |
| Compressed model | weights ≥ 1M elements: integer codes on a uniform grid (step = k · group RMS, `float16` step per 512 columns, no clipping), chosen by GPTQ and entropy-coded with rANS; 0.1% of weights stored exactly as `float16`. **12.8% of MLP neurons removed** (OBD saliency from the Fisher pass; the removed `down_proj` columns are folded into the kept ones by least squares before GPTQ); their gate/up rows and down columns are not stored (packed bit mask) and are restored as zeros. **Embedding split** into common-token rows (from token counts over the base model's maths traces) and the remaining rows, each coded with its own step (rare rows ≥ 1 bit/weight). Smaller tensors kept `bfloat16`. Vision tower and MTP head stored as shape only (restored as zeros). |
| Restored model | `torch.bfloat16`, same tensor names and shapes as the base checkpoint |

## Setup

```bash
uv sync                       # decompress.py: torch + numpy + safetensors only
uv sync --extra compress      # compress.py additionally needs transformers
```

## Compression

```bash
python compress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/base/Qwen3.5-4B \
    --output_path /path/to/compressed
```

Needs a GPU. Constants: `compression/config.py`. Calibration data:
`compression/calib_data/aime_calib.npz` (built once with
`python -m compression.build_calib --train_set <hf dataset repo>:<path to train_set.parquet> --tokenizer <model dir>`
-- the SFT training set: one correct, finished base-model trace per problem, HMMT
Feb/Nov archive + AIME 2022-24 + HMMT/BRUMO/CMIMC/SMT 2025-26; or, as in week 8,
`--traces_repo <repo>` for the AIME 2022-26 traces; the file name is kept either way and
`aime_calib.json` records the source) and
`compression/calib_data/token_freq.npz` (built once with
`python -m compression.build_vocab_freq --tokenizer <base dir> --traces_repo <repo> [--traces_repo <repo2>]`).
Format `cs6013-codecE-rans-v2`: a tensor may be stored as parts (row or column
subsets selected by a packed bit mask); `decompress.py` scatters them back and
fills uncovered positions with zeros.
Side outputs go to `<output_path>_artifacts/`. `CS6013_SMOKE=1` runs a fast
structural test whose output is not a submission.

## Decompression

```bash
python decompress.py \
    --model_name Qwen/Qwen3.5-4B \
    --checkpoint_path /path/to/compressed \
    --output_path /path/to/restored
```

CPU only; a few minutes.
