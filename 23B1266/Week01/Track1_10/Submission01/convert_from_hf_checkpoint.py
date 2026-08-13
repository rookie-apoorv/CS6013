"""Sample submission: compress / restore HF checkpoints by sparsifying last dims."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def _chop_last_dim(tensor: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Keep the first ``(1 - sparsity)`` fraction of ``tensor`` along its last dim."""
    if tensor.ndim == 0 or tensor.shape[-1] < 2:
        return tensor.detach().cpu().contiguous()
    keep = round(tensor.shape[-1] * (1.0 - sparsity))
    keep = max(0, min(keep, tensor.shape[-1]))
    return tensor[..., :keep].detach().cpu().contiguous()




def convert_from_hf_checkpoint(
    model_name: str,
    output_path: str,
    sparsity: float = 0.5,
) -> None:
    """Load an HF model, drop a fraction of each weight's last dim, and save.

    Parameters
    ----------
    model_name:
        Hugging Face model id or local path (e.g. ``Qwen/Qwen3-4B-Instruct-2507``).
    output_path:
        Where to write the compressed ``state_dict`` (``torch.save`` file).
    sparsity:
        Fraction of last-dimension columns to discard. The kept slice is the
        first ``round((1 - sparsity) * last_dim)`` columns (e.g. ``0.5`` keeps
        the first half). Only that slice is stored.

    Remark: This is where your logic of compression goes. This sample code just
        keeps the leading ``(1 - sparsity)`` fraction of each weight's last
        dimension.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    state_dict = model.state_dict()

    compressed: dict[str, torch.Tensor] = {}
    for name, tensor in state_dict.items():
        compressed[name] = _chop_last_dim(tensor, sparsity)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_name": model_name,
            "sparsity": sparsity,
            "state_dict": compressed,
        },
        out,
    )

    # Free memory before returning.
    del model, state_dict, compressed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

