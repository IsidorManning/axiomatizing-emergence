"""Utility functions for controlling random number generators."""

from __future__ import annotations

import random

try:  # optional dependency
    import numpy as np
except Exception:  # pragma: no cover - handled gracefully
    np = None  # type: ignore

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
    if np is not None:  # pragma: no branch - numpy optional
        np.random.seed(seed)

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
