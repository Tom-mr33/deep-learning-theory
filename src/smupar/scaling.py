"""SµPar scaling rules (Table 1 / App. D of arXiv:2405.15743).

Three parameterizations are supported:

===========  ==========================  ================================
name         hidden init variance        hidden learning rate (Adam)
===========  ==========================  ================================
``sp``       sigma_base^2                eta_base
``mup``      sigma_base^2 / m_d          eta_base / m_d
``supar``    sigma_base^2 / (m_d m_rho)  eta_base / (m_d m_rho)
===========  ==========================  ================================

with ``m_d = d_model / d_model_base`` (width multiplier) and
``m_rho = rho / rho_base`` (density multiplier).  For SGD the update term is
already normalised by ``d_in``, so only ``eta = eta_base / m_rho`` is needed
(App. D.3.1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


class Parameterization(str, Enum):
    SP = "sp"
    MUP = "mup"
    SUPAR = "supar"

    @classmethod
    def parse(cls, value: "str | Parameterization") -> "Parameterization":
        if isinstance(value, Parameterization):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:  # pragma: no cover - defensive
            raise ValueError(
                f"unknown parameterization {value!r}; expected one of "
                f"{[p.value for p in cls]}"
            ) from exc


@dataclass(frozen=True)
class SparseScaling:
    """Multipliers describing how far a layer is from the dense base model."""

    width_mult: float = 1.0     # m_d
    density_mult: float = 1.0   # m_rho

    def __post_init__(self) -> None:
        if self.width_mult <= 0 or self.density_mult <= 0:
            raise ValueError("multipliers must be positive")

    @classmethod
    def from_dims(
        cls,
        d_model: int,
        density: float,
        d_model_base: int | None = None,
        density_base: float = 1.0,
    ) -> "SparseScaling":
        base = d_model if d_model_base is None else d_model_base
        return cls(width_mult=d_model / base, density_mult=density / density_base)

    # -- scaling factors -------------------------------------------------- #
    def init_std_factor(self, param: "str | Parameterization") -> float:
        p = Parameterization.parse(param)
        if p is Parameterization.SP:
            return 1.0
        if p is Parameterization.MUP:
            return 1.0 / math.sqrt(self.width_mult)
        return 1.0 / math.sqrt(self.width_mult * self.density_mult)

    def adam_lr_factor(self, param: "str | Parameterization") -> float:
        p = Parameterization.parse(param)
        if p is Parameterization.SP:
            return 1.0
        if p is Parameterization.MUP:
            return 1.0 / self.width_mult
        return 1.0 / (self.width_mult * self.density_mult)

    def sgd_lr_factor(self, param: "str | Parameterization") -> float:
        # App. D.3.1: SGD's update already carries a 1/d_in normalisation, so the
        # update term is width-invariant -- only the density needs correcting.
        # For muP (m_rho = 1) this reduces to eta_base, matching the paper's note
        # that the SGD correction "is equivalent to muP when rho = 1".
        p = Parameterization.parse(param)
        if p is Parameterization.SP:
            return 1.0
        if p is Parameterization.MUP:
            return 1.0
        return 1.0 / self.density_mult


def hidden_init_std(
    base_std: float,
    scaling: SparseScaling,
    param: "str | Parameterization" = Parameterization.SUPAR,
) -> float:
    """Standard deviation for hidden sparse weight initialisation."""
    return base_std * scaling.init_std_factor(param)


def hidden_lr(
    base_lr: float,
    scaling: SparseScaling,
    param: "str | Parameterization" = Parameterization.SUPAR,
    optimizer: str = "adam",
) -> float:
    """Learning rate for hidden sparse weights."""
    factor = (
        scaling.sgd_lr_factor(param) if optimizer.lower() == "sgd"
        else scaling.adam_lr_factor(param)
    )
    return base_lr * factor


def weights_per_neuron(d_model: int, density: float) -> float:
    """WPN = d_model * rho.  When WPN is O(1), µP and SµPar coincide (Sec. 4.2)."""
    return d_model * density


__all__ = [
    "Parameterization",
    "SparseScaling",
    "hidden_init_std",
    "hidden_lr",
    "weights_per_neuron",
]
