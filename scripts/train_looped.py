#!/usr/bin/env python
"""Train a looped (or standard) transformer on an in-context task.

Usage
-----
    python scripts/train_looped.py --config configs/looped_linear_debug.yaml
    python scripts/train_looped.py --config configs/looped_linear.yaml --steps 20000 --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.common.utils import load_config  # noqa: E402
from src.looped.train import train  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="path to a YAML config")
    ap.add_argument("--steps", type=int, default=None, help="override train.steps")
    ap.add_argument("--device", default=None, help="cpu | cuda | auto")
    ap.add_argument("--n-loops", type=int, default=None, dest="n_loops", help="override train.n_loops")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg["train"]["steps"] = args.steps
    if args.device is not None:
        cfg["device"] = args.device
    if args.n_loops is not None:
        cfg["train"]["n_loops"] = args.n_loops
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.run_name is not None:
        cfg["run_name"] = args.run_name

    train(cfg, verbose=not args.quiet)


if __name__ == "__main__":
    main()
