# CS6013 — Compression 20 — Submission02

## Precision

| | dtype |
|---|---|
| Base model (input of `compress.py`) | `torch.bfloat16` — Qwen3.5-4B after fine-tuning: `grey-cat/cs6013-qwen35-4b-sft-v1` |
| Compressed checkpoint (size_frac 0.198) | integer codes entropy-coded with rANS (variable length) + `float16` per-group step sizes + `float16` outlier values; tensors below 1M elements `torch.bfloat16`; vision tower and MTP head not stored (restored as zeros) |
| Decompressed model | `torch.bfloat16`, same tensor names and shapes as Qwen3.5-4B |

## Setup

```bash
uv sync                       # decompress.py: torch + numpy + safetensors
uv sync --extra compress      # compress.py and recovery/ also need transformers
```

## Decompression

```bash
python decompress.py \
    --model_name Qwen-3.5-4B \
    --checkpoint_path /path/to/downloaded/compressed/checkpoint \
    --output_path /path/to/decompressed
```

CPU only, a few minutes, no downloads. `--model_name` is recorded only.

## Reproducing the checkpoint

### 1. Fine-tuned base model

Download `grey-cat/cs6013-qwen35-4b-sft-v1` (public). To rebuild it instead: `notebooks/nb2_train_pool_run.py`
(sample solutions from Qwen3.5-4B) then `notebooks/nb3_sft.py` (LoRA fine-tune, merged into a
`torch.bfloat16` checkpoint); both are marimo notebooks that run on a GPU (molab) and read/write
the Hugging Face repos set in their first cell.

### 2. Compression

```bash
uv sync --extra compress
python compress.py \
    --model_name Qwen-3.5-4B \
    --checkpoint_path /path/to/cs6013-qwen35-4b-sft-v1 \
    --output_path /path/to/compressed
```

Needs a GPU. Constants: `compression/config.py` (defaults are the values used for this checkpoint). Calibration data (committed): `compression/calib_data/aime_calib.npz` (+ `.json`; built from base-model AIME 2022–26 traces).
Side outputs go to `<output_path>_artifacts/`. `CS6013_SMOKE=1` runs a quick structural test
whose output is not a submission. molab driver: `notebooks/run_compress_ec_gptq20_dtw.py`.

### 3. Recovery

Not used for this checkpoint: the published checkpoint is the output of step 2.
