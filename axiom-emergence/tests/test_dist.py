import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for importing `utils`
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.dist import js_div

torch = pytest.importorskip("torch")


def test_js_div_symmetry():
    p = torch.rand(10)
    q = torch.rand(10)
    assert torch.allclose(js_div(p, q), js_div(q, p))


def test_js_div_non_negative():
    p = torch.rand(10)
    q = torch.rand(10)
    div = js_div(p, q)
    assert div >= 0
