"""Classical solvers used as baselines / sanity checks."""

from __future__ import annotations

import torch


def ols_predict(x_ctx: torch.Tensor, y_ctx: torch.Tensor, x_q: torch.Tensor,
                ridge: float = 0.0) -> torch.Tensor:
    """Ordinary (or ridge) least squares fit on the in-context examples.

    Args:
        x_ctx: ``[B, k, d]`` in-context inputs.
        y_ctx: ``[B, k]`` their labels.
        x_q:   ``[B, d]`` query points.
        ridge: optional L2 stabiliser (0 = plain OLS).

    Returns:
        ``[B]`` predictions at the query points.
    """
    B, k, d = x_ctx.shape
    xtx = torch.einsum("bki,bkj->bij", x_ctx, x_ctx)
    if ridge > 0:
        xtx = xtx + ridge * torch.eye(d, device=x_ctx.device).unsqueeze(0)
    xty = torch.einsum("bki,bk->bi", x_ctx, y_ctx)
    # solve is faster / more stable than an explicit inverse
    w = torch.linalg.solve(xtx, xty.unsqueeze(-1)).squeeze(-1)
    return torch.einsum("bd,bd->b", x_q, w)


def normalized_squared_error(pred: torch.Tensor, target: torch.Tensor,
                             eps: float = 1e-12) -> torch.Tensor:
    """MSE divided by the target variance (Garg et al.'s normalisation)."""
    mse = ((pred - target) ** 2).mean()
    return mse / (target.var(unbiased=False) + eps)


__all__ = ["ols_predict", "normalized_squared_error"]
