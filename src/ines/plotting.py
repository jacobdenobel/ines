"""Shared optimization-history data and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunHistory:
    """Best-so-far objective and coordinate-wise delta values from one run."""

    evaluations: np.ndarray
    best_values: np.ndarray
    deltas: np.ndarray
    optimum: np.ndarray
    final_x: np.ndarray


def plot_delta_history(history: RunHistory, output: str | Path | None = None):
    """Plot every coordinate's expected absolute step size over evaluations."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history.evaluations, history.deltas, linewidth=0.8)
    axis.set(xlabel="objective evaluations", ylabel=r"$\delta_i$")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180)
    return figure, axis


def plot_objective_history(history: RunHistory, output: str | Path | None = None):
    """Plot best-so-far objective value over evaluations."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history.evaluations, np.maximum(history.best_values, 1e-12))
    axis.set(xlabel="objective evaluations", ylabel="best objective value")
    axis.set_yscale("log")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180)
    return figure, axis

