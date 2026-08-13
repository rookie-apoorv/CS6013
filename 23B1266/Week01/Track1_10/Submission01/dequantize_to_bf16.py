from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM


def _pad_last_dim(tensor: torch.Tensor, target_last_dim: int) -> torch.Tensor:
    """Pad zeros on the last dimension so shape[-1] == ``target_last_dim``."""
    if tensor.shape[-1] == target_last_dim:
        return tensor
    if tensor.shape[-1] > target_last_dim:
        raise ValueError(
            f"Compressed last dim {tensor.shape[-1]} is larger than "
            f"target {target_last_dim}"
        )
    pad = target_last_dim - tensor.shape[-1]
    # F.pad pad spec is from last dim backwards: (pad_left, pad_right)
    return torch.nn.functional.pad(tensor, (0, pad))


def convert_to_hf_checkpoint(
    model_name: str,
    checkpoint_path: str,
    output_path: str,
) -> None:
    """Restore a sparsified checkpoint into a full HF model directory.

    Loads ``model_name`` for the original parameter shapes, reads the compressed
    checkpoint produced by :func:`convert_from_hf_checkpoint`, pads the missing
    portion of each last dimension with zeros, and writes a standard HF
    checkpoint under ``output_path`` via ``save_pretrained``.

    Parameters
    ----------
    model_name:
        Hugging Face model id or local path used to recover original shapes.
    checkpoint_path:
        Path to the compressed checkpoint from :func:`convert_from_hf_checkpoint`.
    output_path:
        Directory where the restored HF model will be saved.
    Remark: Depending on your compression logic, you should implement this function accordingly.
    This function coverts your "compressed" checkpoint into a full HF model so that we can use
    it for evaluation.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, dict) and "state_dict" in payload:
        compressed = payload["state_dict"]
    else:
        # Allow a bare state_dict for flexibility.
        compressed = payload

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    original = model.state_dict()

    restored: dict[str, torch.Tensor] = {}
    for name, orig_tensor in original.items():
        if name not in compressed:
            # Parameter missing from compressed ckpt: keep zeros of original shape.
            restored[name] = torch.zeros_like(orig_tensor)
            continue
        chunk = compressed[name].to(dtype=orig_tensor.dtype)
        if chunk.shape == orig_tensor.shape:
            restored[name] = chunk
        else:
            # All leading dims must match; only last dim was reduced.
            if chunk.shape[:-1] != orig_tensor.shape[:-1]:
                raise ValueError(
                    f"Shape mismatch for '{name}': compressed {tuple(chunk.shape)} "
                    f"vs original {tuple(orig_tensor.shape)}"
                )
            restored[name] = _pad_last_dim(chunk, orig_tensor.shape[-1])

    model.load_state_dict(restored, strict=True)

    out = Path(output_path)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    del model, original, restored, compressed
    if torch.cuda.is_available():
        torch.cuda.empty_cache()