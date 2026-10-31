"""Decoder-only transformer with an optional weight-shared loop (Yang et al., ICLR 2024).

Reference: "Looped Transformers are Better at Learning Learning Algorithms",
arXiv:2311.12424.

Architecture of the looped variant (Sec. 4.1 of the paper):

    Y_{t+1} = M_theta(Y_t + P),   Y_0 = 0

where ``P`` is the embedded prompt and ``M_theta`` is *one* transformer block
stack of ``n_layer`` layers that is applied ``t`` times.  The ``+ P`` term is the
"input injection"; without it (weight-tying, ``Y_{t+1} = M(Y_t)``) the model
diverges for t beyond the trained loop count.

The unlooped baseline is the same module with ``looped=False``: a single pass
through ``n_layer`` distinct blocks (``n_layer=12`` in the paper).  Both variants
share the embedding / readout so parameter counts are directly comparable.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError(f"n_embd={n_embd} must be divisible by n_head={n_head}")
        self.n_head = n_head
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=True)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=True)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=2)
        head_dim = C // self.n_head
        q = q.view(B, T, self.n_head, head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, head_dim).transpose(1, 2)
        # is_causal=True builds the lower-triangular mask internally.
        y = F.scaled_dot_product_attention(
            q, k, v,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class MLP(nn.Module):
    def __init__(self, n_embd: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd, bias=True)
        self.c_proj = nn.Linear(4 * n_embd, n_embd, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.c_proj(F.gelu(self.c_fc(x))))


class Block(nn.Module):
    """GPT-2 style pre-LN block."""

    def __init__(self, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.mlp = MLP(n_embd, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class TransformerBody(nn.Module):
    """A stack of ``n_layer`` blocks -- the function ``M_theta`` of the paper."""

    def __init__(self, n_layer: int, n_embd: int, n_head: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [Block(n_embd, n_head, dropout) for _ in range(n_layer)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x)
        return x


class InContextTransformer(nn.Module):
    """Transformer trained to solve a function class in context.

    Input format (following Garg et al. 2022):
        position 0:   x_1          position 1:   f(x_1)
        position 2:   x_2          position 3:   f(x_2)
        ...
        position 2k:  x_query

    ``x`` vectors are embedded with ``emb_x``, scalar labels with ``emb_y``;
    both are summed with a learned positional embedding.  The readout maps every
    position to a scalar prediction.  Because attention is causal, the
    prediction at the even position ``2j`` sees exactly the prefix
    ``(x_1, f(x_1), ..., x_j, f(x_j), x_{j+1})``, i.e. one training signal per
    prompt prefix -- which is what Eq. 1 of the paper averages over.
    """

    def __init__(
        self,
        d: int,
        max_seq: int,
        n_layer: int = 1,
        n_head: int = 8,
        n_embd: int = 256,
        dropout: float = 0.0,
        looped: bool = True,
        tie_readout: bool = False,
    ) -> None:
        super().__init__()
        self.d = d
        self.max_seq = max_seq
        self.looped = looped
        self.n_layer = n_layer

        self.emb_x = nn.Linear(d, n_embd, bias=False)
        self.emb_y = nn.Linear(1, n_embd, bias=False)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_seq, n_embd))
        self.body = TransformerBody(n_layer, n_embd, n_head, dropout)
        self.ln_f = nn.LayerNorm(n_embd)
        self.readout = nn.Linear(n_embd, 1, bias=False)
        if tie_readout:
            # Optional: share the readout with the label-embedding (Garg-style).
            self.readout.weight = self.emb_y.weight

        self.apply(self._init_weights)
        nn.init.normal_(self.pos_emb, std=0.02)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------ #
    def forward(
        self,
        x_seq: torch.Tensor,
        y_seq: torch.Tensor,
        n_loops: int = 1,
        return_all: bool = False,
    ):
        """Run the model and return a prediction per loop iteration.

        Args:
            x_seq: ``[B, T, d]`` inputs, zero-filled at label positions.
            y_seq: ``[B, T]`` labels, zero-filled at input positions.
            n_loops: number of loop iterations (ignored when ``looped=False``).
            return_all: if True return the list of per-iteration predictions.

        Returns:
            ``[B, T, 1]`` prediction of the last iteration, or the full list.
        """
        B, T, _ = x_seq.shape
        if T > self.max_seq:
            raise ValueError(f"sequence length {T} exceeds max_seq {self.max_seq}")

        P = self.emb_x(x_seq) + self.emb_y(y_seq.unsqueeze(-1))
        P = P + self.pos_emb[:, :T]

        preds: List[torch.Tensor] = []
        if not self.looped:
            h = self.body(P)
            preds.append(self.readout(self.ln_f(h)))
        else:
            y = torch.zeros_like(P)
            for _ in range(max(1, n_loops)):
                y = self.body(y + P)          # Y_{t+1} = M(Y_t + P)
                preds.append(self.readout(self.ln_f(y)))

        return preds if return_all else preds[-1]

    # ------------------------------------------------------------------ #
    def param_count(self, trainable_only: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters() if (p.requires_grad or not trainable_only)
        )

    def describe(self) -> str:
        kind = "looped" if self.looped else "unlooped"
        return (
            f"InContextTransformer({kind}, L={self.n_layer}, D={self.emb_x.out_features}, "
            f"params={self.param_count() / 1e6:.2f}M)"
        )


def count_effective_depth(model: InContextTransformer, n_loops: int) -> int:
    """Effective depth = blocks applied per forward pass."""
    return model.n_layer * (n_loops if model.looped else 1)


def sinusoidal_positions(n: int, dim: int) -> torch.Tensor:
    """Utility kept for ablations that replace the learned positional embedding."""
    pos = torch.arange(n).unsqueeze(1).float()
    i = torch.arange(0, dim, 2).float() / dim
    pe = torch.zeros(n, dim)
    pe[:, 0::2] = torch.sin(pos / (10000 ** i))
    pe[:, 1::2] = torch.cos(pos / (10000 ** i))
    return pe.unsqueeze(0)


__all__ = [
    "InContextTransformer",
    "TransformerBody",
    "Block",
    "count_effective_depth",
    "sinusoidal_positions",
    "math",
]
