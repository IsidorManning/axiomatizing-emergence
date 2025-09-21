"""A minimal Transformer encoder with sinusoidal positional embeddings."""

from __future__ import annotations

import math
from typing import Iterable

import torch
from torch import nn

from .heads import LinearHead


class TinyTransformer(nn.Module):
    """A tiny Transformer encoder for experiments and tests.

    Parameters
    ----------
    vocab_size:
        Size of the discrete vocabulary to embed.
    d_model:
        Hidden dimension of the model.
    n_layers:
        Number of Transformer encoder layers.
    n_heads:
        Number of attention heads in each layer.
    d_ff:
        Dimensionality of the feedforward network inside each layer.
    dropout:
        Dropout probability applied after token/positional embeddings and
        within the encoder layers.
    max_len:
        Maximum sequence length supported by the positional embeddings.
    n_out:
        Size of the output projection. Defaults to ``vocab_size``.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.0,
        max_len: int = 512,
        n_out: int | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("positional_encoding", self._build_pos_enc(max_len, d_model), persistent=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.head = LinearHead(d_model, n_out or vocab_size)

    # ------------------------------------------------------------------
    def _build_pos_enc(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encodings."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0)  # shape (1, max_len, d_model)

    # ------------------------------------------------------------------
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encode ``tokens`` and return logits from the head."""
        seq_len = tokens.size(1)
        x = self.token_embedding(tokens) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len]
        x = self.dropout(x)
        x = self.encoder(x)
        return self.head(x)

    # --- parameter targeting -------------------------------------------------
    def parameter_groups(self) -> list[dict[str, Iterable[nn.Parameter]]]:
        """Return parameter groups for optimizers or analysis.

        This mirrors the helper provided for the MLP model, exposing separate
        groups for embeddings, the Transformer encoder, and the output head.
        """

        groups = [
            {"params": self.token_embedding.parameters(), "name": "embedding"},
            {"params": self.encoder.parameters(), "name": "encoder"},
        ]
        groups.extend(self.head.parameter_groups())
        return groups
