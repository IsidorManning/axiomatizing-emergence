"""Projection heads for downstream tasks.

This module implements a simple linear head used by various models in the
project. The head is intentionally minimal: it applies a linear projection to
transform hidden states into logits over the desired output dimension.
"""

from __future__ import annotations

import torch
from torch import nn

from typing import Iterable

__all__ = ["CLSHead", "LinearHead", "get_head_for"]


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
    "linear": LinearHead,
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


class LinearHead(nn.Module):
    """A minimal linear projection head.

    Parameters
    ----------
    d_model:
        Dimensionality of the incoming hidden states.
    n_out:
        Size of the output vocabulary or number of prediction targets.
    bias:
        Whether to include a bias term in the projection.
    """

    def __init__(self, d_model: int, n_out: int, bias: bool = True) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, n_out, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - trivial
        """Project hidden states ``x`` to logits."""
        return self.proj(x)

    # --- parameter targeting -------------------------------------------------
    def parameter_groups(self) -> list[dict[str, Iterable[nn.Parameter]]]:
        """Return parameter groups for optimizers or targeted updates.

        The interface mirrors that of the MLP model used elsewhere in the
        project. Each group is represented as a dictionary with a ``params``
        entry containing an iterable of the underlying parameters.
        """

        return [{"params": self.proj.parameters(), "name": "head"}]
