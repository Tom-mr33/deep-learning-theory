"""Sparse linear layer with SµPar-aware initialisation.

The mask is static and stored as a buffer; the forward pass is a dense matmul
with the masked weight (functionally identical to sparse, hardware aside).
"""

from __future__ import annotations

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smupar.mask import random_unstructured_mask
from src.smupar.scaling import Parameterization, SparseScaling, hidden_init_std


class SparseLinear(nn.Module):
    """``y = x (W . M)^T + b`` with a fixed random unstructured mask ``M``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        density: float = 1.0,
        param: Union[str, Parameterization] = Parameterization.SUPAR,
        scaling: Optional[SparseScaling] = None,
        base_init_std: float = 0.02,
        bias: bool = True,
        generator: Optional[torch.Generator] = None,
        device: Union[str, torch.device] = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.density = density
        self.param = Parameterization.parse(param)
        self.scaling = scaling or SparseScaling()

        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(out_features, device=device, dtype=dtype)) if bias else None

        mask = random_unstructured_mask(
            (out_features, in_features), density, generator=generator, device=device, dtype=dtype
        )
        self.register_buffer("mask", mask, persistent=True)

        std = hidden_init_std(base_init_std, self.scaling, self.param)
        nn.init.normal_(self.weight, mean=0.0, std=std)
        with torch.no_grad():
            self.weight.mul_(self.mask)

    @property
    def masked_weight(self) -> torch.Tensor:
        return self.weight * self.mask

    @property
    def actual_density(self) -> float:
        return float(self.mask.mean().item())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.masked_weight, self.bias)

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, density={self.density:g}, "
            f"param={self.param.value}, init_std={self.weight.detach().std().item():.4g}"
        )


__all__ = ["SparseLinear"]
