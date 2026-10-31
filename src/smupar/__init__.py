"""Sparse maximal update parameterization (SµPar, arXiv:2405.15743)."""

from src.smupar.layers import SparseLinear
from src.smupar.mask import apply_mask, random_unstructured_mask
from src.smupar.scaling import (
    Parameterization,
    SparseScaling,
    hidden_init_std,
    hidden_lr,
)

__all__ = [
    "SparseLinear",
    "apply_mask",
    "random_unstructured_mask",
    "Parameterization",
    "SparseScaling",
    "hidden_init_std",
    "hidden_lr",
]
