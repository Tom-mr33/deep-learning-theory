#!/usr/bin/env python
"""Probe activation scaling vs sparsity for SP / µP / SµPar (Figure 5 analogue).

    python scripts/run_supar_probe.py --steps 200 --d-model 256

Writes ``results/supar_probe.csv`` and ``results/supar_probe.png``.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import replace

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.smupar.probe import ProbeConfig, run_scaling_probe  # noqa: E402


def write_csv(records: list[dict], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fields = list(records[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(records)


def plot(records: list[dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    params = sorted({r["param"] for r in records})
    fig, axes = plt.subplots(1, len(params), figsize=(4.2 * len(params), 3.6), sharey=True)
    if len(params) == 1:
        axes = [axes]

    last_step = max(r["step"] for r in records)
    for ax, p in zip(axes, params):
        for rho in sorted({r["density"] for r in records}):
            pts = sorted(
                (r["density"], r["mean_abs_act"])
                for r in records
                if r["param"] == p and r["step"] == last_step and r["layer"] == 0
                and r["density"] == rho
            )
            if pts:
                ax.scatter([rho], [pts[0][1]], s=18, label=f"rho={rho:g}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"{p.upper()}")
        ax.set_xlabel("density")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("mean |pre-activation| (layer 0)")
    fig.suptitle("SµPar probe: activation scale vs density after training")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-layer", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="results")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    cfg = ProbeConfig(
        d_model=args.d_model, n_layer=args.n_layer, batch_size=args.batch_size,
        steps=args.steps, base_lr=args.base_lr, device=args.device,
    )
    print(f"probe: d_model={cfg.d_model} layers={cfg.n_layer} steps={cfg.steps} device={cfg.device}")
    records = run_scaling_probe(cfg)

    csv_path = os.path.join(args.out_dir, "supar_probe.csv")
    write_csv(records, csv_path)
    print(f"saved {csv_path}")
    if not args.no_plot:
        try:
            plot(records, os.path.join(args.out_dir, "supar_probe.png"))
        except Exception as exc:  # matplotlib optional
            print(f"plot skipped: {exc}")


if __name__ == "__main__":
    main()
