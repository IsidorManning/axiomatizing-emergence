from __future__ import annotations

"""Aggregate experiment run outputs into a single CSV table.

This utility scans ``exp/*/runs/`` for ``.jsonl`` and ``.csv`` files and
concatenates their rows into a unified table.  Two metadata columns are added
for each record:

``experiment``
    Name of the experiment directory under ``exp/``.
``run``
    Name of the immediate parent directory of the source file (typically the
    run identifier).

The resulting table is written to ``analysis/summary.csv`` by default but an
alternative location may be provided via the ``--out`` command line flag.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a JSON Lines file."""
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_csv(path: Path) -> Iterable[dict[str, Any]]:
    """Yield dictionaries from a CSV file."""
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def gather_runs(exp_root: Path) -> list[dict[str, Any]]:
    """Collect rows from all run outputs under ``exp_root``.

    Parameters
    ----------
    exp_root:
        Directory containing experiment subdirectories. Each experiment is
        expected to contain a ``runs`` directory with JSONL or CSV files.

    Returns
    -------
    list of dict
        Records augmented with ``experiment`` and ``run`` metadata.
    """

    rows: list[dict[str, Any]] = []
    for exp_dir in sorted(exp_root.glob("*")):
        runs_dir = exp_dir / "runs"
        if not runs_dir.is_dir():
            continue
        for file in runs_dir.rglob("*"):
            if not file.is_file():
                continue
            loaders: dict[str, Any] = {".jsonl": _load_jsonl, ".csv": _load_csv}
            loader = loaders.get(file.suffix.lower())
            if loader is None:
                continue
            for record in loader(file):
                record = dict(record)
                record["experiment"] = exp_dir.name
                record["run"] = file.parent.name
                record["file"] = file.name
                rows.append(record)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarise run outputs across experiments.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("analysis/summary.csv"),
        help="Output CSV path (default: analysis/summary.csv)",
    )
    args = parser.parse_args()

    records = gather_runs(Path("exp"))
    if not records:
        print("No run files found.")
        return

    fieldnames = sorted({key for rec in records for key in rec})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            serialised = {
                k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in rec.items()
            }
            writer.writerow(serialised)
    print(f"Saved {len(records)} rows to {args.out}")


if __name__ == "__main__":
    main()
