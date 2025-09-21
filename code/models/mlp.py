"""Simple MLP for modular addition task.

This module provides a small configurable MLP used in tests.  The model
consists of an embedding layer that maps integer inputs to vectors of a
specified ``width``.  The embeddings are summed over the sequence dimension
(two numbers for modular addition) and passed through ``depth`` fully
connected layers with ReLU activations.  A final linear layer produces logits
over the output vocabulary.

Utility functions are provided to count parameters and to build a model with a
parameter budget.
"""

from __future__ import annotations

import math
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.nn import functional as F


def param_count(width: int, depth: int, input_dim: int, output_dim: int) -> int:
    """Return the number of trainable parameters for an MLP configuration.

    Parameters
    ----------
    width:
        Hidden dimension of the network and embedding size.
    depth:
        Number of hidden ``Linear`` layers (each followed by ``ReLU``).
    input_dim:
        Size of the token vocabulary for the embedding layer.
    output_dim:
        Number of output classes for the final linear layer.
    """

    # Embedding layer has ``input_dim * width`` parameters (no bias).
    total = input_dim * width

    # Each hidden ``Linear`` layer has ``width * width`` weights and ``width`` bias terms.
    total += depth * (width * width + width)

    # Output linear layer mapping ``width`` to ``output_dim`` plus bias.
    total += width * output_dim + output_dim

    return total


def build_for_params(
    target_P: int,
    depth_candidates: Iterable[int],
    input_dim: int = 97,
    output_dim: int = 97,
) -> Tuple[int, int]:
    """Choose ``(width, depth)`` approximating ``target_P`` parameters.

    The function searches over ``depth_candidates`` and picks the combination of
    width and depth that yields a parameter count closest to ``target_P``.

    Parameters
    ----------
    target_P:
        Desired total number of parameters.
    depth_candidates:
        Iterable of candidate depths to consider.
    input_dim, output_dim:
        Dimensions of the embedding vocabulary and output logits.

    Returns
    -------
    (width, depth):
        The chosen hidden dimension and depth.
    """

    best_cfg: Tuple[int, int] | None = None
    best_diff = float("inf")

    for depth in depth_candidates:
        if depth <= 0:
            continue
        # Solve quadratic equation for width approximating target_P
        a = depth
        b = input_dim + output_dim + depth
        c = output_dim - target_P
        disc = b * b - 4 * a * c
        if disc < 0:
            # No real solution; skip this depth
            continue
        width_est = int(max(1, round((-b + math.sqrt(disc)) / (2 * a))))

        # Check nearby widths to minimise difference
        for w in (width_est - 1, width_est, width_est + 1):
            if w < 1:
                continue
            P = param_count(w, depth, input_dim, output_dim)
            diff = abs(P - target_P)
            if diff < best_diff:
                best_diff = diff
                best_cfg = (w, depth)

    if best_cfg is None:
        raise ValueError("No valid configuration found for given target_P and depths")

    return best_cfg


class ModAddMLP(nn.Module):
    """A simple MLP used for modular addition experiments.

    Architecture: ``Embedding`` → ``depth`` × (``Linear`` + ``ReLU``)
    → ``Linear`` producing logits.
    """

    def __init__(self, input_dim: int, output_dim: int, width: int, depth: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.width = width
        self.depth = depth

        self.embedding = nn.Embedding(input_dim, width)

        hidden_layers: list[nn.Module] = []
        for _ in range(depth):
            hidden_layers.append(nn.Linear(width, width))
            hidden_layers.append(nn.ReLU())
        self.hidden = nn.Sequential(*hidden_layers) if hidden_layers else nn.Identity()

        self.output_layer = nn.Linear(width, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Compute logits for input ``x``.

        Parameters
        ----------
        x:
            Tensor of shape ``(batch, seq_len)`` with integer tokens.
        """

        # Embed and sum over sequence dimension (modular addition has two tokens)
        emb = self.embedding(x)  # (batch, seq_len, width)
        h = emb.sum(dim=1)
        h = self.hidden(h)
        logits = self.output_layer(h)
        return logits

    def loss(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Compute cross-entropy loss for a batch ``(inputs, targets)``."""

        inputs, targets = batch
        logits = self.forward(inputs)
        return F.cross_entropy(logits, targets)

