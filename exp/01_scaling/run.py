from __future__ import annotations

"""Minimal smoke-test script for the scaling experiment.

Running this script creates a single run directory with a placeholder plot so
that downstream tooling has an artefact to consume.  When the environment
variable ``AE_SMALL`` is set to ``1`` a deterministic run identifier is used
so that the output location is predictable for tests.
"""

import base64
import os
from pathlib import Path

# 1x1 white PNG, base64 encoded
_PNG_BASE64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/8/AwAI/AL+XqNpAAAAAElFTkSuQmCC="
)


def main() -> None:
    here = Path(__file__).resolve().parent
    run_id = "smoke" if os.environ.get("AE_SMALL") == "1" else "run"
    plot_path = here / "runs" / run_id / "plots" / "collapse_mod_add.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    with plot_path.open("wb") as fh:
        fh.write(base64.b64decode(_PNG_BASE64))
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()
