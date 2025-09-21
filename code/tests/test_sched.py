import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path for importing packages
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

torch = pytest.importorskip("torch")

from train.sched import cosine_with_warmup


def test_cosine_with_warmup_schedule():
    model = torch.nn.Linear(1, 1)
    optim = torch.optim.SGD(model.parameters(), lr=1.0)

    sched = cosine_with_warmup(optim, total_steps=10, warmup_frac=0.2)

    lrs = []
    for _ in range(10):
        optim.step()
        sched.step()
        lrs.append(optim.param_groups[0]["lr"])

    assert lrs[1] > lrs[0]
    assert max(lrs) == pytest.approx(1.0, rel=1e-6)
    assert lrs[-1] == pytest.approx(0.0, abs=1e-6)
