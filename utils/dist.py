"""Distance and divergence utilities."""

from __future__ import annotations

try:  # optional dependency
    import torch
except Exception:  # pragma: no cover - handled gracefully
    torch = None  # type: ignore


def js_div(p: "torch.Tensor", q: "torch.Tensor") -> "torch.Tensor":
    """Compute the Jensen-Shannon divergence between two distributions."""
    if torch is None:  # pragma: no cover - only triggered without torch
        raise ImportError("PyTorch is required for js_div")

    p = p.float()
    q = q.float()
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    eps = 1e-10
    p_log = torch.log(p + eps)
    q_log = torch.log(q + eps)
    m_log = torch.log(m + eps)
    kl_pm = torch.sum(p * (p_log - m_log))
    kl_qm = torch.sum(q * (q_log - m_log))
    return 0.5 * (kl_pm + kl_qm)
