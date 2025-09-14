"""Learning rate schedulers for training utilities."""

from __future__ import annotations

from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    LRScheduler,
)


def cosine_with_warmup(
    optimizer: Optimizer, total_steps: int, warmup_frac: float = 0.02
) -> LRScheduler:
    """Cosine annealing schedule with linear warmup."""
    warmup_steps = max(int(total_steps * warmup_frac), 0)
    cosine_steps = max(total_steps - warmup_steps, 1)

    cosine_sched = CosineAnnealingLR(optimizer, T_max=cosine_steps)

    if warmup_steps > 0:
        warmup_sched = LinearLR(
            optimizer, start_factor=1e-8, end_factor=1.0, total_iters=warmup_steps
        )
        return SequentialLR(
            optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_steps]
        )
    return cosine_sched
