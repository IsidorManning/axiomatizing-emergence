from __future__ import annotations

import torch
from torch import nn

__all__ = ["CLSHead", "get_head_for"]


class CLSHead(nn.Module):
    """A simple linear classification head.

    Parameters
    ----------
    d_in: int
        Dimension of the input features.
    n_classes: int
        Number of output classes.
    """

    def __init__(self, d_in: int, n_classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_in, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute logits for each class given ``x``."""
        return self.linear(x)


_HEAD_REGISTRY: dict[str, type[nn.Module]] = {
    "classification": CLSHead,
    "cls": CLSHead,
}


def get_head_for(task: str | object) -> type[nn.Module]:
    """Return the head class appropriate for ``task``.

    Parameters
    ----------
    task: str | object
        Task identifier or object with ``head_type`` attribute. If an object
        is provided and it lacks ``head_type``, ``"classification"`` is used
        by default.

    Returns
    -------
    type[nn.Module]
        The head class corresponding to ``task``.

    Raises
    ------
    ValueError
        If ``task`` specifies an unknown head type.
    """
    if isinstance(task, str):
        head_type = task
    else:
        head_type = getattr(task, "head_type", "classification")

    try:
        return _HEAD_REGISTRY[head_type]
    except KeyError as exc:  # pragma: no cover - exceptional path
        raise ValueError(f"Unknown head type: {head_type}") from exc
