"""Random unstructured static sparsity masks (the setting of SµPar)."""

from __future__ import annotations

from typing import Optional, Sequence, Union

import torch

Shape = Union[Sequence[int], torch.Size]


def random_unstructured_mask(
    shape: Shape,
    density: float,
    generator: Optional[torch.Generator] = None,
    device: Union[str, torch.device] = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Bernoulli mask with expected density ``density`` (1 = dense, 0 = all pruned)."""
    if not 0.0 < density <= 1.0:
        raise ValueError(f"density must be in (0, 1], got {density}")
    keep = torch.rand(tuple(shape), generator=generator, device=device) < density
    return keep.to(dtype)


def apply_mask(weight: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Elementwise masking; the mask is not stored, so callers keep it as a buffer."""
    return weight * mask


def mask_nonzero_per_row(mask: torch.Tensor) -> torch.Tensor:
    """Number of surviving weights per output row -- i.e. weights-per-neuron (WPN)."""
    return mask.sum(dim=1)


__all__ = ["random_unstructured_mask", "apply_mask", "mask_nonzero_per_row"]
