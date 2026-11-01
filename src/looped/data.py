"""In-context learning task generators (Garg et al. 2022 setting, App. A of the paper).

Every task exposes ``sample(batch_size, k)`` returning a dict with

    x_seq  [B, 2k+1, d]  inputs, zero at label positions
    y_seq  [B, 2k+1]     labels, zero at input positions
    label  [B, 2k+1]     regression target for the prediction at that position
    mask   [B, 2k+1]     1 at the k+1 query positions (even indices), else 0
    y_true [B]           the withheld label of the query point (for reporting)

Layout: ``x_1 y_1 x_2 y_2 ... x_k y_k x_query``.  A causal model predicting at
position ``2j`` sees exactly the prefix with ``j`` in-context examples.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

Tensor = torch.Tensor


# --------------------------------------------------------------------------- #
# Function classes
# --------------------------------------------------------------------------- #
class LinearFunction:
    """f(x) = w^T x, w ~ N(0, I_d)."""

    name = "linear"

    def __init__(self, d: int = 20, generator: Optional[torch.Generator] = None,
                 device: str = "cpu") -> None:
        self.d = d
        self.g = generator
        self.device = device

    def sample_weights(self, batch: int) -> Tensor:
        return torch.randn(batch, self.d, generator=self.g, device=self.device)

    def __call__(self, x: Tensor, w: Tensor) -> Tensor:
        return torch.einsum("bnd,bd->bn", x, w)


class SparseLinearFunction(LinearFunction):
    """f(x) = w^T x with exactly ``s`` non-zero entries in w."""

    name = "sparse_linear"

    def __init__(self, d: int = 20, s: int = 3, generator: Optional[torch.Generator] = None,
                 device: str = "cpu") -> None:
        super().__init__(d, generator, device)
        self.s = s

    def sample_weights(self, batch: int) -> Tensor:
        w = torch.zeros(batch, self.d, device=self.device)
        for b in range(batch):
            idx = torch.randperm(self.d, generator=self.g, device=self.device)[: self.s]
            w[b, idx] = torch.randn(self.s, generator=self.g, device=self.device)
        return w


class DecisionTree:
    """Random binary tree of given depth.

    At each internal node the branch is the sign of a uniformly chosen
    coordinate (threshold 0); leaves hold N(0, 1) values.  Depth 4 -> 31 params.
    """

    name = "decision_tree"

    def __init__(self, d: int = 20, depth: int = 4,
                 generator: Optional[torch.Generator] = None, device: str = "cpu") -> None:
        self.d = d
        self.depth = depth
        self.g = generator
        self.device = device

    def sample_weights(self, batch: int) -> Dict[str, Tensor]:
        n_inner = 2 ** self.depth - 1
        feats = torch.randint(0, self.d, (batch, n_inner), generator=self.g, device=self.device)
        leaves = torch.randn(batch, 2 ** self.depth, generator=self.g, device=self.device)
        return {"feats": feats, "leaves": leaves}

    def __call__(self, x: Tensor, w: Dict[str, Tensor]) -> Tensor:
        B, n, _ = x.shape
        feats, leaves = w["feats"], w["leaves"]
        node = torch.zeros(B, n, dtype=torch.long, device=x.device)
        for _ in range(self.depth):
            f = feats.gather(1, node.reshape(B, n).reshape(B, -1)).reshape(B, n)
            xf = torch.gather(x, 2, f.unsqueeze(-1)).squeeze(-1)
            go_right = (xf > 0).long()
            node = 2 * node + 1 + go_right
        node = node - (2 ** self.depth - 1)
        return leaves.gather(1, node.reshape(B, n)).reshape(B, n)


class TwoLayerNN:
    """f(x) = W2 . ReLU(W1 x + b1), hidden width 100, N(0,1) init."""

    name = "mlp2"

    def __init__(self, d: int = 20, hidden: int = 100,
                 generator: Optional[torch.Generator] = None, device: str = "cpu") -> None:
        self.d = d
        self.hidden = hidden
        self.g = generator
        self.device = device

    def sample_weights(self, batch: int) -> Dict[str, Tensor]:
        return {
            "W1": torch.randn(batch, self.d, self.hidden, generator=self.g, device=self.device),
            "b1": torch.randn(batch, self.hidden, generator=self.g, device=self.device) * 0.1,
            "W2": torch.randn(batch, self.hidden, 1, generator=self.g, device=self.device),
        }

    def __call__(self, x: Tensor, w: Dict[str, Tensor]) -> Tensor:
        h = torch.relu(torch.einsum("bnd,bdh->bnh", x, w["W1"]) + w["b1"].unsqueeze(1))
        return torch.einsum("bnh,bh->bn", h, w["W2"].squeeze(-1))


# --------------------------------------------------------------------------- #
# Wrapper turning a function class into batched prompts
# --------------------------------------------------------------------------- #
class InContextTask:
    def __init__(
        self,
        fn: Callable,
        d: int,
        generator: Optional[torch.Generator] = None,
        device: str = "cpu",
        x_std: float = 1.0,
    ) -> None:
        self.fn = fn
        self.d = d
        self.g = generator
        self.device = device
        self.x_std = x_std
        self.name = getattr(fn, "name", "task")

    def _x(self, batch: int, n: int) -> Tensor:
        return torch.randn(batch, n, self.d, generator=self.g, device=self.device) * self.x_std

    def sample(self, batch_size: int, k: int) -> Dict[str, Tensor]:
        seq = 2 * k + 1
        x = self._x(batch_size, k + 1)                 # k in-context + 1 query
        w = self.fn.sample_weights(batch_size)
        y = self.fn(x, w)                              # [B, k+1]

        x_seq = x.new_zeros(batch_size, seq, self.d)
        y_seq = y.new_zeros(batch_size, seq)
        label = y.new_zeros(batch_size, seq)
        mask = y.new_zeros(batch_size, seq)

        x_seq[:, 0 : 2 * k : 2] = x[:, :k]
        y_seq[:, 1 : 2 * k : 2] = y[:, :k]
        x_seq[:, 2 * k] = x[:, k]

        label[:, 0 : 2 * k : 2] = y[:, :k]
        label[:, 2 * k] = y[:, k]
        mask[:, 0 : 2 * k : 2] = 1.0
        mask[:, 2 * k] = 1.0

        return {
            "x_seq": x_seq,
            "y_seq": y_seq,
            "label": label,
            "mask": mask,
            "y_true": y[:, k],
            "x_query": x[:, k],
        }


def make_task(
    name: str,
    d: int = 20,
    s: int = 3,
    depth: int = 4,
    hidden: int = 100,
    seed: Optional[int] = None,
    device: str = "cpu",
    x_std: float = 1.0,
) -> InContextTask:
    """Factory used by the configs (``task.name``)."""
    g = torch.Generator(device=device)
    if seed is not None:
        g.manual_seed(seed)

    name = name.lower()
    if name == "linear":
        fn = LinearFunction(d, g, device)
    elif name in ("sparse_linear", "sparse"):
        fn = SparseLinearFunction(d, s, g, device)
    elif name in ("decision_tree", "tree"):
        fn = DecisionTree(d, depth, g, device)
    elif name in ("mlp2", "two_layer_nn", "nn"):
        fn = TwoLayerNN(d, hidden, g, device)
    else:
        raise ValueError(f"unknown task {name!r}")
    return InContextTask(fn, d, g, device, x_std=x_std)


__all__ = [
    "LinearFunction",
    "SparseLinearFunction",
    "DecisionTree",
    "TwoLayerNN",
    "InContextTask",
    "make_task",
]
