"""Projection heads for downstream tasks.

This module implements a simple linear head used by various models in the
project. The head is intentionally minimal: it applies a linear projection to
transform hidden states into logits over the desired output dimension.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch import nn


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
