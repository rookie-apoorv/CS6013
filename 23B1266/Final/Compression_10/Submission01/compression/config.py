"""All tuning constants (final_sft_p3dtw_10: P3-lite 10% + decision-token-weighted Fisher, for the SFT model).

Method: codec E grid + Fisher-weighted bit allocation + Fisher outliers + GPTQ
on that grid + rANS entropy coding. Calibration: base-model AIME 2022-2026
traces (grader sampling settings), bundled as token ids.

Codec E for one weight tensor W, per row and group of GROUP_SIZE columns:
    step  = fp16(k_t * RMS(group))                (k_t chosen per tensor)
    code  = integer chosen by GPTQ on the grid {..., -2, -1, 0, 1, 2, ...} * step
            (no clipping: any integer is allowed)
    w_hat = code * step
The integer codes are entropy-coded with rANS. OUTLIER_FRAC of the weights
(largest Fisher x RMS^2) are stored exactly as fp16.

``CS6013_SMOKE=1`` shrinks only the Fisher pass and the GPTQ calibration set;
planning, GPTQ over every layer, encoding and the size check all run for real.
Never submit a smoke output.
"""

from __future__ import annotations

import os
from pathlib import Path

SMOKE = os.environ.get("CS6013_SMOKE", "0") == "1"

# ---- size budget ---------------------------------------------------------------
TARGET_RATIO = 0.098          # planned size_frac
HARD_LIMIT = 0.10             # compress.py refuses to finish above this
GRADER_TEXT_GIB = 8.0585      # graders' denominator

# ---- codec E -------------------------------------------------------------------
GROUP_SIZE = 512
MIN_NUMEL = 1 << 20           # smaller tensors stay bf16
K_MIN, K_MAX, K_STEPS = 0.05, 8.0, 128  # step multipliers searched by the planner
# (at ~1.6 bits/weight many tensors need steps of 2-6 x group RMS)
OUTLIER_FRAC = 0.001          # 0.25% would cost ~4% of a 1.6-bit budget

# ---- calibration data (built once by compression/build_calib.py) -----------------
CALIB_FILE = Path(__file__).parent / "calib_data" / "aime_calib.npz"

# ---- Fisher --------------------------------------------------------------------
FISHER_SAMPLES = 256          # loss on the completion tokens only
FISHER_MAX_TOKENS = 4096

# ---- planner -------------------------------------------------------------------
TABLE_MAX_ROWS = 2048

# ---- GPTQ ----------------------------------------------------------------------
METHOD = "gptq"               # "gptq" | "rtn"
GPTQ_MAX_TOKENS = 2_000_000   # calibration tokens pushed through every layer
PERCDAMP = 0.01
ACT_ORDER = True              # static groups: steps fixed in original column order
BLOCK_SIZE = 128
HESSIAN_BUDGET_GB = 8.0
CACHE_DEVICE = "cuda"         # per-sample hidden states (about 5 KiB per token)
# Re-plan the remaining layers when the bits actually used by GPTQ exceed the
# plan by more than this fraction.
RATE_TOLERANCE = 0.002

# ---- pruning -------------------------------------------------------------------
DROP_COMPONENTS = ("visual", "vision_tower", "vision_model", "video_tower",
                   "image_newline", "multi_modal_projector", "mtp")

# ---- output --------------------------------------------------------------------
# --------------------------------------------------------------------------
# P3-lite (no training): structured MLP neuron pruning + split-precision embedding
# --------------------------------------------------------------------------
# Neuron j of an MLP is removed by dropping row j of gate/up and column j of
# down. Saliency = OBD estimate of the loss increase, 0.5 * sum_i F[i,j] W_down[i,j]^2,
# from the same diagonal Fisher the planner uses. Neurons are removed in granules
# of PRUNE_GRANULE per layer (keeps the column count a multiple of GROUP_SIZE),
# picked greedily across layers until PRUNE_FRAC of all MLP neurons are gone,
# never more than PRUNE_LAYER_CAP of one layer. Before GPTQ, the removed
# columns of down_proj are folded into the kept ones by least squares on the
# layer's calibration Hessian. Pruned rows/columns are not stored; decompress
# restores them as zeros (exactly equivalent to the pruned model).
PRUNE_FRAC = 0.25          # value used for this submission's checkpoint
PRUNE_LAYER_CAP = 0.30
PRUNE_GRANULE = 512
# Embedding (tied with the LM head) is split into two row groups coded with
# their own step multiplier: "common" tokens (smallest set covering
# EMB_COVERAGE of the token counts in TOKEN_FREQ_FILE, clamped to
# [EMB_MIN_COMMON, EMB_MAX_COMMON], plus the file's forced ids) and the rest.
# The planner allocates both; the rare group is never given fewer than
# RARE_MIN_BITS bits/weight, so every token keeps a usable row.
EMB_SPLIT = True
EMB_COVERAGE = 0.9999
EMB_MIN_COMMON = 8192
EMB_MAX_COMMON = 32768
RARE_MIN_BITS = 1.0
TOKEN_FREQ_FILE = Path(__file__).parent / "calib_data" / "token_freq.npz"

SHARD_PREFIX = "compressed_model"
MAX_SHARD_BYTES = 4_000_000_000
SEED = 0

# Decision-token-weighted Fisher (as in week8_ec_gptq_20_dtw): positions where the
# model decides whether to second-guess itself count DECISION_WEIGHT times more in the
# per-sample Fisher loss. A position is a decision position if its target token is a
# decision word (Wait, But, Alternatively, Hmm, Actually, However, Hold - first
# sub-token, with/without leading space, lower-case - or </think>) or the previous token
# ends a sentence/paragraph (newline or . ? ! :). 1.0 = plain Fisher.
DECISION_WEIGHT = 5.0

# Gated DeltaNet delta-rule kernel: flash-linear-attention's Triton kernel when it is
# installed AND agrees with the PyTorch reference on a probe sequence (checked at
# start-up, falls back automatically); conv1d and the gated norm always stay PyTorch.
# Speed only - results match the PyTorch path to bf16 rounding.
USE_FLA = os.environ.get("CS6013_USE_FLA", "1") == "1"

# Notebook overrides for trying variants (the values used are recorded in
# compression_config.json of every output; the defaults above are the submission's).
if os.environ.get("CS6013_PRUNE_FRAC"):
    PRUNE_FRAC = float(os.environ["CS6013_PRUNE_FRAC"])
if os.environ.get("CS6013_DECISION_WEIGHT"):
    DECISION_WEIGHT = float(os.environ["CS6013_DECISION_WEIGHT"])
if os.environ.get("CS6013_EMB_SPLIT"):
    EMB_SPLIT = os.environ["CS6013_EMB_SPLIT"] == "1"

if SMOKE:
    FISHER_SAMPLES = 4
    FISHER_MAX_TOKENS = 512
    GPTQ_MAX_TOKENS = 8192
