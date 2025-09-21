from __future__ import annotations

"""Fit scaling law parameters from run summaries.

This script aggregates per-run summaries, averages capabilities across seeds,
fits a simple five-parameter scaling function using nonlinear least squares,
bootstraps confidence intervals, and stores the results in
``analysis/results_scaling.csv``.

The expected layout of the experiment directory is ``exp/<run>/summary.csv``
where each ``summary.csv`` contains at least the columns ``seed``, ``R``
(resource) and ``capability``.
"""

from pathlib import Path
import glob
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class FitResult:
    parameter: str
    estimate: float
    ci_lower: float
    ci_upper: float


def scaling_function(R: np.ndarray, A: float, a: float, b: float, c: float, gamma: float) -> np.ndarray:
    """Simple saturating scaling function.

    The form is chosen to use the five parameters ``A, a, b, c, gamma``. It is
    flexible enough for various empirical scaling curves yet intentionally
    uncomplicated as the repository does not prescribe a particular form.
    """
    return A + a * (R ** b) / ((c + R) ** gamma)


def load_summaries(exp_dir: Path) -> pd.DataFrame:
    """Load all ``summary.csv`` files underneath ``exp_dir``."""
    pattern = str(exp_dir / "*" / "summary.csv")
    files = glob.glob(pattern)
    if not files:
        return pd.DataFrame()

    frames = []
    for file in files:
        run = Path(file).resolve().parent.name
        df = pd.read_csv(file)
        df["run"] = run
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def seed_average(df: pd.DataFrame) -> pd.DataFrame:
    """Average capability across seeds for each run and resource value."""
    if df.empty:
        return df
    group_cols = [c for c in df.columns if c not in ("capability", "seed")]
    return df.groupby(group_cols, as_index=False)["capability"].mean()


def fit_parameters(R: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit the scaling function parameters using nonlinear least squares."""
    p0 = [float(np.max(y)), 1.0, 1.0, 1.0, 1.0]
    params, _ = curve_fit(scaling_function, R, y, p0=p0, maxfev=10000)
    return params


def bootstrap_parameters(R: np.ndarray, y: np.ndarray, popt: np.ndarray, *, n_boot: int = 500) -> np.ndarray:
    """Bootstrap confidence intervals for the parameters."""
    rng = np.random.default_rng()
    samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(R), len(R))
        R_s = R[idx]
        y_s = y[idx]
        try:
            params, _ = curve_fit(scaling_function, R_s, y_s, p0=popt, maxfev=10000)
            samples.append(params)
        except RuntimeError:
            # Occasionally the fit may fail for a bootstrap sample.
            continue
    return np.array(samples)


def main() -> None:
    exp_dir = Path("exp")
    df = load_summaries(exp_dir)
    if df.empty:
        print("No run summaries found.")
        return

    avg = seed_average(df)
    R = avg["R"].to_numpy()
    y = avg["capability"].to_numpy()

    popt = fit_parameters(R, y)
    samples = bootstrap_parameters(R, y, popt, n_boot=500)
    if samples.size:
        ci_lower = np.percentile(samples, 2.5, axis=0)
        ci_upper = np.percentile(samples, 97.5, axis=0)
    else:
        ci_lower = [np.nan] * len(popt)
        ci_upper = [np.nan] * len(popt)

    parameters = ["A", "a", "b", "c", "gamma"]
    results = [
        FitResult(param, est, low, high)
        for param, est, low, high in zip(parameters, popt, ci_lower, ci_upper)
    ]
    out_df = pd.DataFrame(results)

    SCRIPT_DIR.mkdir(exist_ok=True, parents=True)
    out_file = SCRIPT_DIR / "results_scaling.csv"
    out_df.to_csv(out_file, index=False)
    print(f"Saved results to {out_file}")


if __name__ == "__main__":
    main()
