import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for importing `utils`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils import set_seeds

torch = pytest.importorskip("torch")


def test_rng_reproducibility():
    set_seeds(42)
    first = torch.rand(3, 3)
    set_seeds(42)
    second = torch.rand(3, 3)
    assert torch.equal(first, second)
