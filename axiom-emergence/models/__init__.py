"""Model architectures for Axiom Emergence experiments."""

from .heads import LinearHead
from .tiny_transformer import TinyTransformer

__all__ = ["LinearHead", "TinyTransformer"]
