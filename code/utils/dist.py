"""Distance and similarity measures for probability distributions and features."""

from __future__ import annotations

from typing import Any

import numpy as np

try:  # pragma: no cover - optional dependency
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None  # type: ignore

if torch is not None:  # pragma: no cover - typing helper
    ArrayLike = np.ndarray | torch.Tensor  # type: ignore[misc]
else:  # pragma: no cover - torch is unavailable
    ArrayLike = np.ndarray


def _to_numpy(x: Any) -> np.ndarray:
    """Convert ``x`` to a NumPy array, detaching from Torch if necessary."""
    if isinstance(x, np.ndarray):
        return x
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def tv(p: ArrayLike, q: ArrayLike) -> float:
    """Total variation distance between two discrete distributions.

    Parameters
    ----------
    p, q:
        Probability distributions over the same support. They need not be
        normalized; they will be normalized internally.

    Returns
    -------
    float
        ``0.5 * sum(abs(p - q))``.
    """
    p_np = _to_numpy(p).astype(float)
    q_np = _to_numpy(q).astype(float)
    p_np /= p_np.sum()
    q_np /= q_np.sum()
    return 0.5 * float(np.sum(np.abs(p_np - q_np)))


def js_div(p: ArrayLike, q: ArrayLike, eps: float = 1e-12) -> ArrayLike | float:
    """Jensen-Shannon divergence between two discrete distributions.

    Parameters
    ----------
    p, q:
        Probability distributions over the same support. They need not be
        normalized; they will be normalized internally.
    eps:
        Small constant added for numerical stability when taking logarithms.

    Returns
    -------
    float
        Jensen-Shannon divergence in nats.
    """
    p_np = _to_numpy(p).astype(float)
    q_np = _to_numpy(q).astype(float)
    p_np /= p_np.sum()
    q_np /= q_np.sum()
    m = 0.5 * (p_np + q_np)
    p_safe = np.clip(p_np, eps, 1.0)
    q_safe = np.clip(q_np, eps, 1.0)
    m_safe = np.clip(m, eps, 1.0)
    kl_pm = np.sum(p_safe * np.log(p_safe / m_safe))
    kl_qm = np.sum(q_safe * np.log(q_safe / m_safe))
    result = 0.5 * float(kl_pm + kl_qm)
    if torch is not None and (isinstance(p, torch.Tensor) or isinstance(q, torch.Tensor)):
        return torch.tensor(result, dtype=torch.float32)
    return result


def wasserstein_logits(logits_p: ArrayLike, logits_q: ArrayLike) -> float:
    """1D Wasserstein distance between distributions defined by logits.

    The logits are assumed to correspond to equally spaced histogram bins.

    Parameters
    ----------
    logits_p, logits_q:
        Logits for the two distributions. They need not be normalized.

    Returns
    -------
    float
        1-Wasserstein (Earth Mover's) distance assuming unit bin width.
    """
    p = _to_numpy(logits_p).astype(float)
    q = _to_numpy(logits_q).astype(float)
    p = np.exp(p - p.max())
    q = np.exp(q - q.max())
    p /= p.sum()
    q /= q.sum()
    cdf_p = np.cumsum(p)
    cdf_q = np.cumsum(q)
    return float(np.sum(np.abs(cdf_p - cdf_q)))


def cka(X: ArrayLike, Y: ArrayLike) -> float:
    """Centered Kernel Alignment similarity for features.

    Parameters
    ----------
    X, Y:
        Feature matrices with shape ``(n_samples, n_features)``.

    Returns
    -------
    float
        Linear CKA similarity between ``X`` and ``Y``.
    """
    X_np = _to_numpy(X).astype(float)
    Y_np = _to_numpy(Y).astype(float)
    X_np -= X_np.mean(axis=0, keepdims=True)
    Y_np -= Y_np.mean(axis=0, keepdims=True)
    K = X_np.T @ Y_np
    num = np.linalg.norm(K, ord="fro") ** 2
    denom_x = np.linalg.norm(X_np.T @ X_np, ord="fro")
    denom_y = np.linalg.norm(Y_np.T @ Y_np, ord="fro")
    denom = denom_x * denom_y
    if denom == 0:
        return 0.0
    return float(num / denom)


def svcca(X: ArrayLike, Y: ArrayLike, variance_threshold: float = 0.99, eps: float = 1e-12) -> float:
    """Singular Vector CCA similarity between feature sets.

    Parameters
    ----------
    X, Y:
        Feature matrices with shape ``(n_samples, n_features)``.
    variance_threshold:
        Proportion of variance to keep when performing the SVD-based
        dimensionality reduction step.
    eps:
        Small constant for numerical stability.

    Returns
    -------
    float
        Mean canonical correlation between the reduced representations.
    """
    X_np = _to_numpy(X).astype(float)
    Y_np = _to_numpy(Y).astype(float)
    X_np -= X_np.mean(axis=0, keepdims=True)
    Y_np -= Y_np.mean(axis=0, keepdims=True)

    # SVD-based dimensionality reduction
    Ux, Sx, _ = np.linalg.svd(X_np, full_matrices=False)
    Uy, Sy, _ = np.linalg.svd(Y_np, full_matrices=False)
    var_x = np.cumsum(Sx**2) / np.sum(Sx**2)
    var_y = np.cumsum(Sy**2) / np.sum(Sy**2)
    kx = np.searchsorted(var_x, variance_threshold) + 1
    ky = np.searchsorted(var_y, variance_threshold) + 1
    X_red = Ux[:, :kx] * Sx[:kx]
    Y_red = Uy[:, :ky] * Sy[:ky]

    # Covariance matrices
    Sxx = X_red.T @ X_red
    Syy = Y_red.T @ Y_red
    Sxy = X_red.T @ Y_red

    # Inverse square roots
    evals_x, evects_x = np.linalg.eigh(Sxx)
    evals_y, evects_y = np.linalg.eigh(Syy)
    inv_sqrt_x = evects_x @ np.diag(1.0 / np.sqrt(np.maximum(evals_x, eps))) @ evects_x.T
    inv_sqrt_y = evects_y @ np.diag(1.0 / np.sqrt(np.maximum(evals_y, eps))) @ evects_y.T

    T = inv_sqrt_x @ Sxy @ inv_sqrt_y
    corr = np.linalg.svd(T, compute_uv=False)
    return float(np.mean(corr))
