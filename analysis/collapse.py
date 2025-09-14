from __future__ import annotations

r"""Collapse capability curves onto a common scaling law.

This script reads run summaries, computes an effective resource
``\tilde{R}`` as a multiplicative combination of resource features with
exponents ``\alpha`` and optimizes these exponents using L-BFGS so that
``capability \approx A * \tilde{R}^{-\gamma}`` where ``A`` and ``\gamma`` are
obtained from ``results_scaling.csv`` produced by ``fit_scaling.py``.

For each run a log-log scatter plot of capability versus ``\tilde{R}`` is
stored under ``<run>/plots/collapse.png``.
"""

from pathlib import Path
import glob

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent


def load_runs(exp_dir: Path) -> dict[Path, pd.DataFrame]:
    """Load all run summaries into memory."""
    pattern = str(exp_dir / "*" / "summary.csv")
    files = glob.glob(pattern)
    runs: dict[Path, pd.DataFrame] = {}
    for file in files:
        run_dir = Path(file).resolve().parent
        df = pd.read_csv(file)
        df["run"] = run_dir.name
        runs[run_dir] = df
    return runs


def optimise_alpha(df: pd.DataFrame, resource_cols: list[str], A: float, gamma: float) -> np.ndarray:
    """Fit exponents ``alpha`` such that ``capability ≈ A * tildeR**(-gamma)``."""
    X = np.log(df[resource_cols].values)
    y = np.log(df["capability"].values)

    def loss(alpha: np.ndarray) -> float:
        tilde_log = X @ alpha
        pred_log = np.log(A) - gamma * tilde_log
        return np.mean((y - pred_log) ** 2)

    res = minimize(loss, x0=np.zeros(len(resource_cols)), method="L-BFGS-B")
    return res.x


def main() -> None:
    exp_dir = Path("exp")
    runs = load_runs(exp_dir)
    if not runs:
        print("No run summaries found.")
        return

    # Combine data across runs to optimise alpha.
    combined = pd.concat(runs.values(), ignore_index=True)
    resource_cols = [c for c in combined.columns if c not in ("capability", "seed", "run")]
    if not resource_cols:
        raise ValueError("Run summaries lack resource columns to define \u03b1 parameters.")

    results_file = SCRIPT_DIR / "results_scaling.csv"
    if not results_file.exists():
        raise FileNotFoundError("results_scaling.csv not found. Run fit_scaling.py first.")
    params = pd.read_csv(results_file)
    params = dict(zip(params["parameter"], params["estimate"]))
    A = float(params["A"])
    gamma = float(params["gamma"])

    alpha = optimise_alpha(combined, resource_cols, A, gamma)
    print("Optimised alphas:", alpha)

    # Produce collapsed plots for each run
    for run_dir, df in runs.items():
        tildeR = np.exp(np.log(df[resource_cols].values) @ alpha)
        plt.figure()
        plt.scatter(tildeR, df["capability"])
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel(r"$\tilde{R}$")
        plt.ylabel("Capability")
        plt.title(f"Collapsed scaling for {run_dir.name}")
        plot_dir = run_dir / "plots"
        plot_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_dir / "collapse.png")
        plt.close()
    print("Saved collapsed plots to each run's plots/ directory.")


if __name__ == "__main__":
    main()
