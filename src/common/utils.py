"""Small shared helpers: seeding, config loading, lightweight logging."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Any, Dict

import torch
import yaml


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(pref: str = "auto") -> str:
    if pref == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return pref


def load_config(path: str | os.PathLike) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"config {path} did not parse into a mapping")
    return cfg


def results_dir(root: str | os.PathLike = "results") -> Path:
    p = Path(root)
    p.mkdir(parents=True, exist_ok=True)
    return p


def append_jsonl(path: str | os.PathLike, record: Dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def markdown_table(rows: list[list[Any]], header: list[str]) -> str:
    """Render a small markdown table (used for the run summary)."""
    widths = [max(len(str(r[i])) for r in [header] + rows) for i in range(len(header))]
    line = "| " + " | ".join(str(h).ljust(w) for h, w in zip(header, widths)) + " |"
    sep = "|" + "|".join("-" * (w + 1) for w in widths) + "|"
    body = [
        "| " + " | ".join(str(c).ljust(w) for c, w in zip(r, widths)) + " |" for r in rows
    ]
    return "\n".join([line, sep, *body])


__all__ = [
    "set_seed",
    "pick_device",
    "load_config",
    "results_dir",
    "append_jsonl",
    "markdown_table",
]
