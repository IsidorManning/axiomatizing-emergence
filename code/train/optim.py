import torch

try:
    from transformers.optimization import Adafactor
    _HAS_ADAFACTOR = True
except Exception:  # pragma: no cover - optional dependency
    Adafactor = None
    _HAS_ADAFACTOR = False


def make_optimizer(model: torch.nn.Module, opt_name: str, lr: float, weight_decay: float, betas: tuple[float, float] = (0.9, 0.95)) -> torch.optim.Optimizer:
    """Create an optimizer for ``model``.

    Parameters
    ----------
    model:
        Model whose parameters will be optimized.
    opt_name:
        Name of the optimizer to create ("adamw" or "adafactor").
    lr:
        Learning rate.
    weight_decay:
        Weight decay coefficient.
    betas:
        Beta coefficients for AdamW; ``(0.9, 0.95)`` by default.

    Returns
    -------
    torch.optim.Optimizer
        Instantiated optimizer.

    Notes
    -----
    If ``opt_name`` is ``"adafactor"`` but the optional dependency
    ``transformers`` is not available, this function falls back to
    ``torch.optim.AdamW``.
    """
    name = opt_name.lower()
    params = model.parameters()

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)
    if name == "adafactor":
        if _HAS_ADAFACTOR:
            # ``Adafactor`` does not support the ``betas`` argument directly; we map
            # the first beta to ``beta1`` for consistency and rely on its internal
            # decay for the second moment.
            return Adafactor(params, lr=lr, weight_decay=weight_decay, beta1=betas[0])
        # Fallback to AdamW if Adafactor is unavailable
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas)

    raise ValueError(f"Unknown optimizer '{opt_name}'")
