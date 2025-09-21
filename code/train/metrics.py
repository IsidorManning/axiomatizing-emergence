"""Utility evaluation metrics for training."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import numpy as np

ArrayLike = Sequence[float] | np.ndarray
LabelLike = Sequence[int] | np.ndarray


def accuracy(preds: ArrayLike, labels: LabelLike) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    preds:
        Predicted probabilities or class labels. If ``preds`` has an extra
        dimension relative to ``labels`` it is interpreted as a probability
        distribution over classes and ``argmax`` is used. For binary
        probabilistic predictions a threshold of ``0.5`` is applied.
    labels:
        Ground-truth class labels.

    Returns
    -------
    float
        Fraction of predictions equal to labels.
    """
    p = np.asarray(preds)
    y = np.asarray(labels)

    if p.ndim > y.ndim:
        p = p.argmax(axis=-1)
    elif p.dtype.kind == "f" and not np.array_equal(p, p.astype(int)):
        p = (p >= 0.5).astype(y.dtype)
    return float(np.mean(p == y))


def brier_score(preds: ArrayLike, labels: LabelLike) -> float:
    """Compute the Brier score for probabilistic predictions.

    Parameters
    ----------
    preds:
        Predicted probabilities for the positive class or distributions over
        classes.
    labels:
        Ground-truth class labels.

    Returns
    -------
    float
        Mean squared error between predicted probabilities and one-hot labels.
    """
    p = np.asarray(preds, dtype=float)
    y = np.asarray(labels)

    if p.ndim > 1:
        y_one_hot = np.eye(p.shape[-1])[y]
        diff = p - y_one_hot
        return float(np.mean(np.sum(diff ** 2, axis=-1)))
    y_float = y.astype(float)
    diff = p - y_float
    return float(np.mean(diff ** 2))


def capability_from_preds(preds: ArrayLike, labels: LabelLike) -> dict[str, float]:
    """Return both accuracy and Brier score for ``preds`` and ``labels``."""
    return {"accuracy": accuracy(preds, labels), "brier": brier_score(preds, labels)}


def seed_average(metrics_list: Sequence[Mapping[str, float]]) -> dict[str, float]:
    """Average metrics across seeds.

    Only metrics present in all dictionaries are averaged and returned.
    """
    if not metrics_list:
        return {}
    keys = set(metrics_list[0].keys())
    for m in metrics_list[1:]:
        keys &= m.keys()
    return {k: float(np.mean([m[k] for m in metrics_list])) for k in keys}

