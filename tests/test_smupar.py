"""Checks that the SµPar rules do what the derivation claims (App. D)."""

import math

import torch

from src.smupar.layers import SparseLinear
from src.smupar.scaling import (
    Parameterization,
    SparseScaling,
    hidden_init_std,
    hidden_lr,
    weights_per_neuron,
)


def test_init_std_matches_derivation():
    base = 0.02
    s = SparseScaling(width_mult=4.0, density_mult=0.25)
    assert math.isclose(hidden_init_std(base, s, "sp"), base)
    assert math.isclose(hidden_init_std(base, s, "mup"), base / 2.0)
    assert math.isclose(hidden_init_std(base, s, "supar"), base / 1.0)


def test_lr_rules():
    base_lr = 1e-3
    s = SparseScaling(width_mult=2.0, density_mult=0.5)
    assert math.isclose(hidden_lr(base_lr, s, "sp"), base_lr)
    assert math.isclose(hidden_lr(base_lr, s, "mup"), base_lr / 2.0)
    assert math.isclose(hidden_lr(base_lr, s, "supar"), base_lr / 1.0)
    # SGD only needs the density correction (App. D.3.1)
    assert math.isclose(
        hidden_lr(base_lr, SparseScaling(1.0, 0.5), "supar", optimizer="sgd"), base_lr / 0.5
    )


def test_supar_keeps_activation_variance_density_invariant():
    """Var(Y) = d * rho * sigma_W^2 * Var(X) should not depend on rho."""
    torch.manual_seed(0)
    d = 1024
    x = torch.randn(256, d)
    ratios = {}
    for rho in (1.0, 0.5, 0.25, 0.0625):
        layer = SparseLinear(
            d, 512, density=rho, param=Parameterization.SUPAR,
            scaling=SparseScaling.from_dims(d, rho), base_init_std=0.02,
        )
        y = layer(x)
        ratios[rho] = float(y.detach().var() / x.var())
    values = list(ratios.values())
    assert max(values) / min(values) < 1.15, ratios


def test_sp_activations_shrink_with_sparsity():
    torch.manual_seed(0)
    d = 1024
    x = torch.randn(256, d)
    variances = []
    for rho in (1.0, 0.25, 0.0625):
        layer = SparseLinear(
            d, 512, density=rho, param=Parameterization.SP,
            scaling=SparseScaling.from_dims(d, rho), base_init_std=0.02,
        )
        variances.append(float(layer(x).var()))
    assert variances[0] > variances[1] > variances[2]


def test_weights_per_neuron_identity():
    assert weights_per_neuron(256, 1.0) == 256.0
    assert weights_per_neuron(2048, 0.125) == 256.0


def test_mask_is_static_and_zeroes_pruned_updates():
    layer = SparseLinear(64, 32, density=0.25, param="supar")
    w0 = layer.masked_weight.detach().clone()
    assert (w0[layer.mask == 0]).abs().sum() == 0
    x = torch.randn(8, 64, requires_grad=False)
    layer(x).sum().backward()
    assert torch.allclose(layer.mask, (layer.mask > 0).float())
    assert layer.weight.grad[layer.mask == 0].abs().sum() == 0
