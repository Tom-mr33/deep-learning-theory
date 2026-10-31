"""Looped-transformer reproduction package (arXiv:2311.12424)."""

from src.looped.data import make_task
from src.looped.model import InContextTransformer
from src.looped.solvers import normalized_squared_error, ols_predict
from src.looped.train import evaluate, train, windowed_loss

__all__ = [
    "make_task",
    "InContextTransformer",
    "ols_predict",
    "normalized_squared_error",
    "evaluate",
    "train",
    "windowed_loss",
]
