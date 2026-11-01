"""Signal-propagation probe: qualitatively reproduces Figure 5 of SµPar.

We train a small residual stack of sparsified feed-forward blocks at several
densities and record the mean absolute pre-activation of each layer, at
initialisation and during training.  The claim under test (Finding 1):

* ``sp`` and ``mup``  -> activations shrink as density decreases;
* ``supar``           -> activation scale is roughly independent of density.

Scope note: this is a *toy* probe (no attention, a synthetic linear-regression
task, ~1e5-1e6 params).  It tests the mechanism, not the paper's LLM results.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.smupar.layers import SparseLinear
from src.smupar.scaling import Parameterization, SparseScaling, hidden_lr


@dataclass
class ProbeConfig:
    d_model: int = 256
    n_layer: int = 4
    densities: List[float] = field(default_factory=lambda: [1.0, 0.5, 0.25, 0.125, 0.0625])
    params: List[str] = field(default_factory=lambda: ["sp", "mup", "supar"])
    base_init_std: float = 0.02
    base_lr: float = 1e-3
    batch_size: int = 64
    steps: int = 200
    out_dim: int = 16
    seed: int = 0
    record_every: int = 50
    device: str = "cpu"


class _SparseBlock(nn.Module):
    """h <- h + W2 gelu(W1 h), both projections sparsified."""

    def __init__(self, cfg: ProbeConfig, density: float, param: Parameterization,
                 scaling: SparseScaling, generator: torch.Generator) -> None:
        super().__init__()
        kw = dict(density=density, param=param, scaling=scaling,
                  base_init_std=cfg.base_init_std, generator=generator,
                  bias=False, device=cfg.device)
        self.fc1 = SparseLinear(cfg.d_model, 4 * cfg.d_model, **kw)
        self.fc2 = SparseLinear(4 * cfg.d_model, cfg.d_model, **kw)

    def forward(self, h: torch.Tensor, stats: Optional[List[float]] = None) -> torch.Tensor:
        z = self.fc1(h)
        if stats is not None:
            stats.append(float(z.detach().abs().mean()))
        a = F.gelu(z)
        return h + self.fc2(a)


class SparseStack(nn.Module):
    def __init__(self, cfg: ProbeConfig, density: float, param: Parameterization) -> None:
        super().__init__()
        self.cfg = cfg
        g = torch.Generator(device=cfg.device)
        g.manual_seed(cfg.seed)
        scaling = SparseScaling.from_dims(cfg.d_model, density)
        self.blocks = nn.ModuleList(
            [_SparseBlock(cfg, density, param, scaling, g) for _ in range(cfg.n_layer)]
        )
        self.head = nn.Linear(cfg.d_model, cfg.out_dim, device=cfg.device)

    def forward(self, x: torch.Tensor, collect: bool = False):
        h = x
        stats: List[float] = []
        for blk in self.blocks:
            h = blk(h, stats if collect else None)
        out = self.head(h)
        return (out, stats) if collect else out


def _make_task(cfg: ProbeConfig) -> Dict[str, torch.Tensor]:
    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed + 1)
    w_true = torch.randn(cfg.d_model, cfg.out_dim, generator=g, device=cfg.device)
    w_true = w_true / math.sqrt(cfg.d_model)
    return {"w_true": w_true, "g": g}


@torch.no_grad()
def measure_activations(model: SparseStack, cfg: ProbeConfig, x: torch.Tensor) -> List[float]:
    model.eval()
    _, stats = model(x, collect=True)
    model.train()
    return stats


def run_one(cfg: ProbeConfig, density: float, param: Parameterization) -> List[Dict]:
    """Train a single (density, parameterization) configuration."""
    model = SparseStack(cfg, density, param).to(cfg.device)
    task_w = _make_task(cfg)["w_true"]
    scaling = SparseScaling.from_dims(cfg.d_model, density)
    lr = hidden_lr(cfg.base_lr, scaling, param, optimizer="adam")

    # SµPar/µP scaling applies to *hidden sparse* weights only; the dense output
    # head keeps the base learning rate (otherwise the head becomes a confounder).
    hidden_params = [p for n, p in model.named_parameters() if not n.startswith("head.")]
    head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(
        [
            {"params": hidden_params, "lr": lr},
            {"params": head_params, "lr": cfg.base_lr},
        ],
        weight_decay=0.0,
    )
    g = torch.Generator(device=cfg.device)
    g.manual_seed(cfg.seed + 2)

    records: List[Dict] = []

    def record(step: int, loss: Optional[float], grad_norm: Optional[float]) -> None:
        x = torch.randn(cfg.batch_size, cfg.d_model, generator=g, device=cfg.device)
        stats = measure_activations(model, cfg, x)
        for layer, act in enumerate(stats):
            records.append({
                "param": param.value, "density": density, "step": step, "layer": layer,
                "mean_abs_act": act, "loss": loss, "grad_norm": grad_norm, "lr": lr,
            })

    x0 = torch.randn(cfg.batch_size, cfg.d_model, generator=g, device=cfg.device)
    record(0, None, None)

    for step in range(1, cfg.steps + 1):
        x = torch.randn(cfg.batch_size, cfg.d_model, generator=g, device=cfg.device)
        y = x @ task_w
        loss = F.mse_loss(model(x), y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, 1e9)
        opt.step()
        if step % cfg.record_every == 0 or step == cfg.steps:
            record(step, float(loss.detach()), float(gnorm))

    del x0
    return records


def run_scaling_probe(cfg: Optional[ProbeConfig] = None) -> List[Dict]:
    cfg = cfg or ProbeConfig()
    out: List[Dict] = []
    for pname in cfg.params:
        param = Parameterization.parse(pname)
        for density in cfg.densities:
            recs = run_one(cfg, density, param)
            out.extend(recs)
            last = recs[-1]
            print(
                f"  {pname:6s} rho={density:6.4f}  lr={last['lr']:.2e}  "
                f"loss={last['loss']:.4f}  mean|act|(L0)={recs[0]['mean_abs_act']:.3e}"
            )
    return out


__all__ = ["ProbeConfig", "run_scaling_probe", "run_one", "SparseStack", "asdict"]
