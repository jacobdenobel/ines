"""Shared optimization-history data and plotting helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class RunHistory:
    """Best-so-far objective and coordinate-wise delta values from one run."""

    evaluations: np.ndarray
    function_values: np.ndarray
    best_values: np.ndarray
    l1_distances: np.ndarray
    deltas: np.ndarray
    optimum: np.ndarray
    final_x: np.ndarray


def plot_delta_history(history: RunHistory, output: str | Path | None = None):
    """Plot every coordinate's expected absolute step size over evaluations."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes

    figure, axis = plt.subplots(figsize=(8, 4.5))
    coordinate_count = history.deltas.shape[1]
    colormap = plt.get_cmap("viridis")
    normalization = plt.Normalize(0, coordinate_count - 1)
    for index, values in enumerate(history.deltas.T):
        axis.plot(
            history.evaluations,
            values,
            color=colormap(normalization(index)),
            linewidth=0.8,
        )

    scalar_map = mpl.cm.ScalarMappable(cmap=colormap, norm=normalization)
    scalar_map.set_array([])
    colorbar_axis = inset_axes(
        axis,
        width="20%",
        height="4%",
        loc="lower left",
        borderpad=1,
    )
    colorbar = figure.colorbar(scalar_map, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_ticks([0, coordinate_count - 1])
    colorbar.set_ticklabels([r"$\delta_1$", r"$\delta_n$"])
    colorbar.ax.xaxis.set_ticks_position("top")
    colorbar.ax.xaxis.set_label_position("top")

    axis.set(xlabel="objective evaluations", ylabel=r"$\delta_i$")
    axis.set_yscale("log")
    axis.grid()
    figure.subplots_adjust(left=0.12, right=0.97, bottom=0.15, top=0.97)
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180)
    return figure, axis


def plot_objective_history(history: RunHistory, output: str | Path | None = None):
    """Plot selected and best-so-far objective values, including exact zero."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        history.evaluations,
        history.function_values,
        color="red",
        alpha=0.5,
        label="selected offspring",
    )
    axis.plot(
        history.evaluations,
        history.best_values,
        color="black",
        linewidth=1.2,
        label="best so far",
    )
    axis.set(xlabel="objective evaluations", ylabel=r"$f$")
    axis.set_yscale("symlog", base=10, linthresh=1.0)
    axis.set_ylim(bottom=0)
    axis.grid()
    axis.legend()
    figure.tight_layout()
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180)
    return figure, axis


def plot_l1_distance_history(history: RunHistory, output: str | Path | None = None):
    """Plot the center's integer L1 distance to the known optimum."""
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(history.evaluations, history.l1_distances, color="green")
    axis.set(
        xlabel="objective evaluations",
        ylabel=r"$|x - x^*|_1$",
    )
    axis.set_yscale("symlog", base=10, linthresh=1.0)
    axis.set_ylim(bottom=0)
    axis.grid()
    figure.tight_layout()
    if output is not None:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(target, dpi=180)
    return figure, axis

