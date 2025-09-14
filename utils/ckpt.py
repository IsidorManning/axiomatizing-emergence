"""Utility functions for saving and loading model checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore


def _require_torch() -> None:
    """Raise ``RuntimeError`` if PyTorch is not installed."""
    if torch is None:  # pragma: no cover - simple guard
        raise RuntimeError("PyTorch is required for checkpointing")


def save_ckpt(path: str | Path, model_state: Dict[str, Any], meta: Dict[str, Any]) -> None:
    """Save a model checkpoint.

    Parameters
    ----------
    path:
        Destination file. The parent directory will be created if necessary.
    model_state:
        State dictionary to serialize.
    meta:
        Arbitrary metadata stored alongside the model state.
    """
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model_state, "meta": meta}, path)


def load_ckpt(path: str | Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Load a model checkpoint saved with :func:`save_ckpt`.

    Parameters
    ----------
    path:
        Checkpoint file path.

    Returns
    -------
    Tuple[Dict[str, Any], Dict[str, Any]]
        ``(state_dict, meta)`` loaded from the checkpoint.
    """
    _require_torch()
    ckpt = torch.load(Path(path), map_location="cpu")
    return ckpt["model"], ckpt.get("meta", {})
