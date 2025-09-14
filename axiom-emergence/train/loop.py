"""Training and evaluation loops."""

from __future__ import annotations

from itertools import cycle
from typing import Callable, Dict, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast

from utils import set_seeds

MetricFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def train(
    model: torch.nn.Module,
    dataloader: Iterable,
    steps: int,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler,
    device: torch.device,
    seed: int,
    log_fn: Callable[[int, Dict[str, float]], None],
    grad_clip: float | None = None,
    amp: bool = False,
) -> None:
    """Train ``model`` for a fixed number of steps.

    Parameters
    ----------
    model:
        The neural network to train.
    dataloader:
        Iterable yielding ``(inputs, targets)`` batches.
    steps:
        Number of optimisation steps to perform.
    optimizer:
        Optimiser updating ``model`` parameters.
    scheduler:
        Learning rate scheduler stepped after each optimisation step.
    device:
        Device on which computation takes place.
    seed:
        Random seed for reproducibility. Passed to :func:`set_seeds`.
    log_fn:
        Callback receiving ``(step, metrics)`` after each evaluation period.
    grad_clip:
        Optional maximum norm for gradient clipping.
    amp:
        Enable automatic mixed precision when ``True``.
    """

    set_seeds(seed)
    model.to(device)
    scaler = GradScaler(enabled=amp)

    metrics: Dict[str, MetricFn] = {
        "loss": lambda logits, targets: F.cross_entropy(logits, targets, reduction="mean"),
    }
    eval_interval = max(1, len(dataloader))

    # Initial evaluation before any training
    eval_metrics = evaluate(model, dataloader, metrics, device)
    log_fn(0, eval_metrics)

    data_iter = cycle(dataloader)
    for step in range(1, steps + 1):
        model.train()
        inputs, targets = next(data_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        if amp:
            scaler.scale(loss).backward()
            if grad_clip is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()

        if step % eval_interval == 0 or step == steps:
            eval_metrics = evaluate(model, dataloader, metrics, device)
            log_fn(step, eval_metrics)


def evaluate(
    model: torch.nn.Module,
    dataloader: Iterable,
    metrics: Dict[str, MetricFn],
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate ``model`` on ``dataloader``.

    Saves predictions and labels from the dataloader to ``probe_predictions.npz``
    and returns average metric values.
    """

    model.eval()
    totals = {name: 0.0 for name in metrics}
    n = 0
    preds: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            preds.append(logits.cpu())
            labels.append(targets.cpu())
            batch_size = targets.size(0)
            for name, fn in metrics.items():
                val = fn(logits, targets).item()
                totals[name] += val * batch_size
            n += batch_size

    results = {name: total / max(1, n) for name, total in totals.items()}

    if preds:
        np.savez(
            "probe_predictions.npz",
            logits=torch.cat(preds).numpy(),
            labels=torch.cat(labels).numpy(),
        )

    return results
