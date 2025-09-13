"""Utility functions for controlling random number generators."""

from __future__ import annotations

import random

import numpy as np

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore


def set_seeds(seed: int, deterministic: bool = True) -> None:
    """Seed random number generators for reproducibility.

    Parameters
    ----------
    seed:
        Seed value used for ``random``, ``numpy`` and, if available,
        ``torch``.
    deterministic:
        When ``True`` and PyTorch is installed, sets CuDNN to
        deterministic mode and disables benchmarking to reduce
        nondeterminism.
    """

    random.seed(seed)
    np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

