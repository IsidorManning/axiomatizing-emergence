"""Helpers for expanding experiment configuration grids."""

from __future__ import annotations

import itertools
from typing import Any, Iterable, Mapping


def _as_iterable(val: Any) -> Iterable[Any]:
    if isinstance(val, (list, tuple, set)):
        return list(val)
    return [val]


def expand_grid(spec: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expand a sparse factorial grid specification.

    Parameters
    ----------
    spec:
        Mapping where each key maps to either a single value or an iterable of
        values. The Cartesian product of the value lists defines the grid.

    Returns
    -------
    list of dict
        Each dictionary corresponds to one point in the expanded grid.
    """

    keys = list(spec.keys())
    values = [list(_as_iterable(spec[k])) for k in keys]
    combos = []
    for prod in itertools.product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos


def load_and_expand(path: str | bytes | "os.PathLike[str]") -> list[dict[str, Any]]:
    """Load a YAML file and expand it into a list of configurations."""
    import yaml

    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if "grid" in cfg:
        grid_spec = cfg.get("grid", {})
        base = {k: v for k, v in cfg.items() if k != "grid"}
        expanded = expand_grid(grid_spec)
        return [{**base, **g} for g in expanded]
    return expand_grid(cfg)


def main(argv: list[str] | None = None) -> None:
    """Expand a YAML grid file into a JSON plan.

    Parameters
    ----------
    argv:
        Optional sequence of arguments. Uses :data:`sys.argv` when ``None``.
    """
    import argparse
    import json

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", help="YAML grid specification to expand")
    parser.add_argument(
        "--out",
        "-o",
        help="Optional path to save the expanded plan as JSON",
    )
    args = parser.parse_args(argv)

    runs = load_and_expand(args.path)
    plan = json.dumps(runs, indent=2)
    print(plan)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(plan)


if __name__ == "__main__":
    main()
