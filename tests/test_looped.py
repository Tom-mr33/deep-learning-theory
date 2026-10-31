"""Sanity checks for the in-context data pipeline and the looped model."""

import torch

from src.looped.data import make_task
from src.looped.model import InContextTransformer
from src.looped.solvers import normalized_squared_error, ols_predict
from src.looped.train import windowed_loss


def test_prompt_layout():
    task = make_task("linear", d=20, seed=0)
    k = 5
    b = task.sample(4, k)
    seq = 2 * k + 1
    assert b["x_seq"].shape == (4, seq, 20)
    assert b["label"].shape == (4, seq)

    # labels live only at odd positions, inputs only at even positions
    assert torch.allclose(b["y_seq"][:, 1::2][:, :k], b["label"][:, 0:-1:2], atol=0)
    assert b["y_seq"][:, 0::2].abs().sum() == 0
    # k+1 prediction positions
    assert float(b["mask"].sum()) == 4 * (k + 1)
    # the last label is the withheld query label
    assert torch.allclose(b["label"][:, -1], b["y_true"])


def test_ols_solves_noiseless_linear():
    task = make_task("linear", d=20, seed=1)
    k = 40
    b = task.sample(16, k)
    pred = ols_predict(
        b["x_seq"][:, 0 : 2 * k : 2], b["label"][:, 0 : 2 * k : 2], b["x_query"], ridge=1e-6
    )
    assert normalized_squared_error(pred, b["y_true"]).item() < 1e-8


def test_looped_forward_returns_one_prediction_per_iteration():
    model = InContextTransformer(d=8, max_seq=21, n_layer=1, n_head=2, n_embd=16, looped=True)
    task = make_task("linear", d=8, seed=2)
    b = task.sample(3, 10)
    preds = model(b["x_seq"], b["y_seq"], n_loops=5, return_all=True)
    assert len(preds) == 5
    assert preds[0].shape == (3, 21, 1)
    assert model(b["x_seq"], b["y_seq"], n_loops=5).shape == (3, 21, 1)


def test_looped_uses_far_fewer_parameters():
    looped = InContextTransformer(d=20, max_seq=81, n_layer=1, n_head=8, n_embd=256, looped=True)
    standard = InContextTransformer(d=20, max_seq=81, n_layer=12, n_head=8, n_embd=256, looped=False)
    # paper reports 0.79M vs 9.48M, i.e. the looped model is <10% of the baseline
    assert looped.param_count() < 0.15 * standard.param_count()


def test_unlooped_ignores_loop_argument():
    model = InContextTransformer(d=8, max_seq=11, n_layer=2, n_head=2, n_embd=16, looped=False)
    task = make_task("linear", d=8, seed=3)
    b = task.sample(2, 5)
    preds = model(b["x_seq"], b["y_seq"], n_loops=9, return_all=True)
    assert len(preds) == 1


def test_windowed_loss_is_zero_for_perfect_predictions():
    label = torch.zeros(2, 5)
    mask = torch.tensor([[1.0, 0, 1, 0, 1], [1.0, 0, 1, 0, 1]])
    preds = [label.unsqueeze(-1)] * 4
    assert windowed_loss(preds, label, mask, b0=1, b=4).item() == 0.0
