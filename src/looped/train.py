"""Training / evaluation loop for the looped-transformer reproduction.

Implements the truncated windowed objective of Eq. 1 (Yang et al., 2024):

    min_theta E_P [ 1/(b-b0) * sum_{t=b0}^{b} 1/(k+1) * sum_{i=0}^{k} l(Y_t(P^i), f(x_{i+1})) ]

with ``b0 = max(b - T, 0)``.  ``n_loops`` plays the role of ``b`` and
``loss_window`` the role of ``T``; ``b`` is optionally scheduled from
``b_start`` up to ``n_loops`` over training (App. E).
"""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import torch

from src.common.utils import append_jsonl, markdown_table, pick_device, results_dir, set_seed
from src.looped.data import make_task
from src.looped.model import InContextTransformer
from src.looped.solvers import ols_predict


# --------------------------------------------------------------------------- #
# Objective
# --------------------------------------------------------------------------- #
def windowed_loss(
    preds: List[torch.Tensor],
    label: torch.Tensor,
    mask: torch.Tensor,
    b0: int,
    b: int,
) -> torch.Tensor:
    """Average squared error over loop iterations ``b0..b-1`` (0-indexed)."""
    if not 0 <= b0 < b <= len(preds):
        raise ValueError(
            f"invalid loss window: b0={b0}, b={b}, n_predictions={len(preds)}; "
            f"require 0 <= b0 < b <= len(preds)"
        )
    total = label.new_zeros(())
    count = 0
    for t in range(b0, b):
        p = preds[t].squeeze(-1)
        total = total + (((p - label) ** 2) * mask).sum() / mask.sum()
        count += 1
    return total / max(count, 1)


def current_b(step: int, cfg: Dict) -> int:
    """Loop-iteration schedule: linear ramp from ``b_start`` to ``n_loops``."""
    b_max = int(cfg["n_loops"])
    if b_max < 1:
        raise ValueError(f"n_loops must be >= 1, got {b_max}")
    if not cfg.get("schedule_b", False):
        return b_max
    b_start = int(cfg.get("b_start", 1))
    if b_start < 1:
        raise ValueError(f"b_start must be >= 1, got {b_start}")
    warmup = int(cfg.get("b_warmup_steps", max(cfg["steps"] // 2, 1)))
    frac = min(1.0, step / max(warmup, 1))
    return int(round(b_start + frac * (b_max - b_start)))


# --------------------------------------------------------------------------- #
# Model construction
# --------------------------------------------------------------------------- #
def build_model(cfg: Dict, max_seq: int) -> InContextTransformer:
    m = cfg["model"]
    return InContextTransformer(
        d=int(cfg["task"]["d"]),
        max_seq=max_seq,
        n_layer=int(m.get("n_layer", 1)),
        n_head=int(m.get("n_head", 8)),
        n_embd=int(m.get("n_embd", 256)),
        dropout=float(m.get("dropout", 0.0)),
        looped=bool(m.get("looped", True)),
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def evaluate(
    model: InContextTransformer,
    task,
    k: int,
    iterations: List[int],
    batch_size: int = 256,
    device: str = "cpu",
    chunk_size: int = 64,
) -> Dict[str, Dict[str, float]]:
    """Per-loop-iteration normalised error at the query position, + OLS baseline.

    The batch is processed in chunks of ``chunk_size`` prompts so that looped
    evaluation with many iterations (e.g. b=100) does not spike GPU memory.
    Results are identical to a single large batch (same sampled prompts).
    """
    model.eval()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    batch = {key: v.to(device) for key, v in task.sample(batch_size, k).items()}
    n_loops = max(iterations)
    y_true = batch["y_true"]

    sq_err = {t: 0.0 for t in iterations}
    ols_sq = 0.0
    n = 0
    for start in range(0, batch_size, chunk_size):
        sl = slice(start, start + chunk_size)
        preds = model(batch["x_seq"][sl], batch["y_seq"][sl], n_loops=n_loops, return_all=True)
        y_sub = y_true[sl]
        for t in iterations:
            idx = min(t, len(preds)) - 1
            pred_q = preds[idx][:, -1, 0]
            sq_err[t] += float(((pred_q - y_sub) ** 2).sum())
        x_ctx = batch["x_seq"][sl, 0 : 2 * k : 2]
        y_ctx = batch["label"][sl, 0 : 2 * k : 2]
        ols = ols_predict(x_ctx, y_ctx, batch["x_query"][sl], ridge=1e-6)
        ols_sq += float(((ols - y_sub) ** 2).sum())
        n += int(y_sub.shape[0])

    out: Dict[str, Dict[str, float]] = {}
    y_var = float(y_true.var(unbiased=False)) + 1e-12
    for t in iterations:
        out[str(t)] = {"err": sq_err[t] / n / y_var}
    out["ols"] = {"err": ols_sq / n / y_var}
    model.train()
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(cfg: Dict, verbose: bool = True) -> Dict:
    seed = int(cfg.get("seed", 0))
    set_seed(seed)
    device = pick_device(cfg.get("device", "auto"))

    tcfg = cfg["train"]
    ecfg = cfg.get("eval", {})
    k_max = int(tcfg.get("k_max", 40))
    k_eval = int(ecfg.get("k", k_max))
    max_seq = 2 * max(k_max, k_eval) + 1

    task = make_task(
        cfg["task"]["name"],
        d=int(cfg["task"]["d"]),
        s=int(cfg["task"].get("s", 3)),
        depth=int(cfg["task"].get("depth", 4)),
        hidden=int(cfg["task"].get("hidden", 100)),
        seed=seed + 1,
        device=device,
    )

    model = build_model(cfg, max_seq).to(device)
    model.train()
    opt = torch.optim.Adam(
        model.parameters(), lr=float(tcfg.get("lr", 1e-4)),
        weight_decay=float(tcfg.get("weight_decay", 0.0)),
    )

    if verbose:
        print(f"[{cfg.get('run_name', 'run')}] task={task.name} device={device}")
        print(f"  {model.describe()}  effective depth @ b={tcfg['n_loops']}: "
              f"{model.n_layer * (tcfg['n_loops'] if model.looped else 1)}")

    steps = int(tcfg.get("steps", 1000))
    batch_size = int(tcfg.get("batch_size", 64))
    T = int(tcfg.get("loss_window", 15))
    eval_every = int(tcfg.get("eval_every", 0))
    curriculum_k = bool(tcfg.get("curriculum_k", True))
    iterations = [int(i) for i in ecfg.get("iterations", [tcfg["n_loops"]])]

    for name, value in [("steps", steps), ("batch_size", batch_size),
                        ("loss_window", T), ("k_max", k_max)]:
        if value < 1:
            raise ValueError(f"train.{name} must be >= 1, got {value}")
    if any(i < 1 for i in iterations):
        raise ValueError(f"eval.iterations must all be >= 1, got {iterations}")
    if int(tcfg["n_loops"]) < 1:
        raise ValueError(f"train.n_loops must be >= 1, got {tcfg['n_loops']}")
    log_path = results_dir(cfg.get("results_dir", "results")) / f"{cfg.get('run_name', 'run')}.jsonl"

    t0 = time.time()
    history: List[Dict] = []
    for step in range(1, steps + 1):
        k = int(torch.randint(1, k_max + 1, (1,)).item()) if curriculum_k else k_max
        batch = {key: v.to(device) for key, v in task.sample(batch_size, k).items()}

        b = current_b(step, {**tcfg, "steps": steps}) if model.looped else 1
        b0 = max(b - T, 0)
        # truncated BPTT: only the window [b0, b) carries gradients
        preds = model(batch["x_seq"], batch["y_seq"], n_loops=b, return_all=True,
                      detach_before=b0 if model.looped else 0)
        loss = windowed_loss(preds, batch["label"], batch["mask"], b0, b)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=float(tcfg.get("grad_clip", 1e9))
        )
        opt.step()

        if step % 10 == 0 or step == 1:
            rec = {
                "step": step, "k": k, "b": b, "loss": float(loss.detach()),
                "grad_norm": float(grad_norm), "elapsed_s": round(time.time() - t0, 2),
            }
            history.append(rec)
            append_jsonl(log_path, rec)
            if verbose and (step % max(10, steps // 20) == 0 or step == 1):
                print(f"  step {step:6d}  k={k:3d}  b={b:3d}  loss={rec['loss']:.5f}  "
                      f"|g|={rec['grad_norm']:.2f}  {rec['elapsed_s']:.0f}s")

        if eval_every and step % eval_every == 0:
            ev = evaluate(model, task, k_eval, iterations, int(ecfg.get("batch_size", 256)), device)
            if verbose:
                pretty = "  ".join(f"b={t}:{v['err']:.4f}" for t, v in ev.items() if t != "ols")
                print(f"  [eval @ {step}] {pretty}  ols={ev['ols']['err']:.2e}")

    final = evaluate(model, task, k_eval, iterations, int(ecfg.get("batch_size", 256)), device)
    result = {
        "run_name": cfg.get("run_name", "run"),
        "task": task.name,
        "looped": bool(cfg["model"].get("looped", True)),
        "n_layer": int(cfg["model"].get("n_layer", 1)),
        "params": model.param_count(),
        "steps": steps,
        "k_eval": k_eval,
        "final": final,
        "wall_time_s": round(time.time() - t0, 1),
    }

    if verbose:
        rows = [[t, f"{v['err']:.5f}"] for t, v in final.items()]
        print("\n" + markdown_table(rows, ["loop iterations / baseline", "normalised error"]))

    out_path = results_dir(cfg.get("results_dir", "results")) / f"{cfg.get('run_name', 'run')}_summary.json"
    import json

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"\nsaved {out_path}")
    return result


def train_from_config(path: str, verbose: bool = True) -> Dict:
    from src.common.utils import load_config

    return train(load_config(path), verbose=verbose)


__all__ = ["train", "train_from_config", "evaluate", "windowed_loss", "current_b", "build_model"]
