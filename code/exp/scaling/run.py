"""Scaling experiment orchestrator."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Dict, Any

import torch
import torch.nn.functional as F

from models.mlp import ModAddMLP, build_for_params
from scripts.make_grids import load_and_expand
from tasks.mod_add import get_dataloaders
from train.loop import train, evaluate
from train.optim import make_optimizer
from train.sched import cosine_with_warmup
from utils.ckpt import save_ckpt
from utils.log import RunLogger


def _build_model(P: int, depth_candidates: list[int]) -> ModAddMLP:
    width, depth = build_for_params(P, depth_candidates)
    return ModAddMLP(input_dim=97, output_dim=97, width=width, depth=depth)


def _metrics() -> Dict[str, Callable[[torch.Tensor, torch.Tensor], torch.Tensor]]:
    return {
        "loss": lambda logits, targets: F.cross_entropy(logits, targets, reduction="mean"),
        "acc": lambda logits, targets: (logits.argmax(dim=-1) == targets).float().mean(),
    }


def main() -> None:
    small = os.environ.get("AE_SMALL") == "1"
    cfg_name = "grid_small.yml" if small else "grid.yml"
    cfg_path = Path(__file__).with_name(cfg_name)

    configs = load_and_expand(cfg_path)
    results: list[Dict[str, Any]] = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for cfg in configs:
        P = int(cfg["P"])
        T = int(cfg["T"])
        S = int(cfg["S"])
        seed = int(cfg["seed"])
        batch_size = int(cfg.get("batch_size", 32))
        depth_candidates = [int(d) for d in cfg.get("depth_candidates", [1, 2, 3, 4])]
        lr = float(cfg.get("lr", 1e-3))
        weight_decay = float(cfg.get("weight_decay", 0.0))
        opt_name = cfg.get("optimizer", "adamw")

        model = _build_model(P, depth_candidates)

        train_dl, val_dl, test_dl, _ = get_dataloaders(T, batch_size, seed)

        optimizer = make_optimizer(model, opt_name, lr, weight_decay)
        scheduler = cosine_with_warmup(optimizer, S)

        run_dir = Path("runs") / f"P{P}_T{T}_S{S}_seed{seed}"
        logger = RunLogger(run_dir, **cfg)
        log_fn = lambda step, metrics: logger.log(step, **metrics)

        train(model, train_dl, S, optimizer, scheduler, device, seed, log_fn)
        save_ckpt(run_dir / "model.pt", model.state_dict(), cfg)

        eval_metrics = evaluate(model, test_dl, _metrics(), device)
        logger.log(S, **{f"test_{k}": v for k, v in eval_metrics.items()})
        results.append({"P": P, "T": T, "S": S, "seed": seed, **eval_metrics})

    # Aggregate seed-averaged metrics
    from collections import defaultdict
    import csv

    grouped: Dict[tuple[int, int, int], list[Dict[str, Any]]] = defaultdict(list)
    for r in results:
        key = (r["P"], r["T"], r["S"])
        grouped[key].append(r)

    metric_names = [k for k in results[0] if k not in {"P", "T", "S", "seed"}]
    rows: list[Dict[str, Any]] = []
    for (P, T, S), items in grouped.items():
        row: Dict[str, Any] = {"P": P, "T": T, "S": S}
        for m in metric_names:
            row[m] = sum(it[m] for it in items) / len(items)
        rows.append(row)

    out_path = Path("analysis") / "results_scaling.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["P", "T", "S", *metric_names])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


if __name__ == "__main__":
    main()
