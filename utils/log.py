import json
import time
from pathlib import Path
from typing import Any, Dict


def append_jsonl(path: str | Path, record_dict: Dict[str, Any]) -> None:
    """Append ``record_dict`` as a JSON line to ``path``.

    Parameters
    ----------
    path:
        Destination file. The parent directory will be created if necessary.
    record_dict:
        Dictionary to serialize as a single line of JSON.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        json.dump(record_dict, fh)
        fh.write("\n")


class RunLogger:
    """Utility for logging experiment runs.

    Parameters
    ----------
    run_dir:
        Directory in which to store log files. ``meta.json`` and
        ``metrics.jsonl`` are created inside this directory.
    **meta:
        Arbitrary metadata stored in ``meta.json`` when the logger is created.

    Examples
    --------
    >>> logger = RunLogger("runs/example", model="gpt")
    >>> logger.log(step=1, loss=0.5)
    """

    def __init__(self, run_dir: str | Path, **meta: Any) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        meta_path = self.run_dir / "meta.json"
        with meta_path.open("w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        self.metrics_path = self.run_dir / "metrics.jsonl"

    def log(self, step: int, **metrics: Any) -> None:
        """Record a set of metrics for a given step.

        Parameters
        ----------
        step:
            Step number associated with the metrics.
        **metrics:
            Arbitrary key-value metrics to log.
        """
        record = {"step": step, "time": time.time()}
        record.update(metrics)
        append_jsonl(self.metrics_path, record)
