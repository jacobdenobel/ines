"""Single-file implementation and paper-style runner for original INES.

The optimizer mirrors the reference implementation in ``integer-es``:
best-offspring selection, a Fisher-normalized coordinate-wise signal, an
evolution path, and a multiplicative update of ``E[|Z_i|]``. Running this file
directly evaluates it on one of the four quadratic functions from the paper
and saves objective- and delta-history plots.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

PAPER_FUNCTIONS = ("sphere", "ellipse", "discus", "cigar")


def sample_double_geometric(
    rng: np.random.Generator, delta: np.ndarray, population_size: int
) -> np.ndarray:
    """Sample coordinate-wise DG mutations as differences of geometrics."""
    q = delta / (np.sqrt(1.0 + delta**2) + 1.0)
    p = np.clip(1.0 - q, 1e-12, 1.0 - 1e-12)
    log_q = np.log1p(-p)
    shape = (len(delta), population_size)
    g1 = np.floor(np.log1p(-rng.random(shape)) / log_q).astype(int)
    g2 = np.floor(np.log1p(-rng.random(shape)) / log_q).astype(int)
    return g1 - g2


class BarebonesINES:
    """Original coordinate-wise ``(1, lambda)`` INES in ask/tell form."""

    def __init__(
        self,
        x0: np.ndarray,
        delta0: float,
        population_size: int = 10,
        c: float | None = None,
        eta: float | None = None,
        seed: int | None = None,
        binary: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.x = np.asarray(x0, dtype=int).reshape(-1, 1).copy()
        self.n = len(self.x)
        self.population_size = population_size
        self.c = 1.0 - 1.5 / self.n if c is None else c
        self.eta = (2.0 / self.n) ** (1.0 / 3.0) if eta is None else eta
        self.path_scale = np.sqrt(self.c * (2.0 - self.c))
        self.delta = np.full((self.n, 1), delta0, dtype=float)
        self.path = np.zeros((self.n, 1))
        self.binary = binary

    def ask(self) -> np.ndarray:
        self.steps = sample_double_geometric(self.rng, self.delta, self.population_size)
        candidates = self.x + self.steps
        return candidates & 1 if self.binary else candidates

    def tell(self, values: np.ndarray) -> None:
        """Update from objective values in the same column order as ``ask``."""
        best = int(np.argmin(values))
        selected_step = self.steps[:, best, None]
        self.x += selected_step

        variance_abs_step = self.delta * np.sqrt(1.0 + self.delta**2)
        gradient = (np.abs(selected_step) - self.delta) / np.maximum(
            variance_abs_step, 1.0
        )
        self.path = (1.0 - self.c) * self.path + self.path_scale * gradient
        self.delta *= np.exp(self.eta * self.path)


def make_paper_objective(
    kind: str, dimension: int, instance: int = 1
) -> tuple[Callable[[np.ndarray], np.ndarray], np.ndarray]:
    """Return a vectorized paper quadratic and its shifted integer optimum."""
    if kind not in PAPER_FUNCTIONS:
        raise ValueError(f"kind must be one of {PAPER_FUNCTIONS}")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")

    optimum = np.random.default_rng(instance).integers(-50, 51, dimension)
    if kind == "sphere":
        weights = np.ones(dimension)
    elif kind == "ellipse":
        weights = np.logspace(0.0, 4.0, num=dimension)
    elif kind == "discus":
        weights = np.ones(dimension)
        weights[0] = 10_000.0
    else:
        weights = np.ones(dimension)
        weights[1:] = 10_000.0

    def objective(points: np.ndarray) -> np.ndarray:
        points = np.asarray(points, dtype=float)
        if points.ndim == 1:
            points = points[:, None]
        if points.shape[0] != dimension:
            raise ValueError(f"points must have shape ({dimension}, samples)")
        shifted = points - optimum[:, None]
        return np.linalg.norm(shifted * weights[:, None], axis=0)

    return objective, optimum


def corrected_delta0(dimension: int) -> float:
    """Paper scale ``sigma0=100/n`` with the corrected DG conversion."""
    sigma0 = 100.0 / dimension
    variance0 = sigma0**2
    return variance0 / np.sqrt(2.0 * variance0 + 1.0)


@dataclass
class RunHistory:
    evaluations: np.ndarray
    best_values: np.ndarray
    deltas: np.ndarray
    optimum: np.ndarray
    final_x: np.ndarray


def run_paper_benchmark(
    kind: str = "sphere",
    dimension: int = 20,
    instance: int = 1,
    seed: int = 1993,
    budget: int | None = None,
    population_size: int = 10,
    c: float | None = None,
    eta: float | None = None,
) -> RunHistory:
    """Run barebones INES under the paper's quadratic benchmark conditions."""
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if budget is None:
        budget = 10_000 * dimension
    if budget < population_size:
        raise ValueError("budget must allow at least one generation")

    objective, optimum = make_paper_objective(kind, dimension, instance)
    x0 = np.random.default_rng(seed).integers(-50, 51, dimension)
    optimizer = BarebonesINES(
        x0,
        delta0=corrected_delta0(dimension),
        population_size=population_size,
        c=c,
        eta=eta,
        seed=seed,
    )

    evaluations: list[int] = []
    best_values: list[float] = []
    deltas: list[np.ndarray] = []
    best_so_far = np.inf
    evaluated = 0

    while evaluated + population_size <= budget:
        points = optimizer.ask()
        values = objective(points)
        optimizer.tell(values)

        evaluated += population_size
        best_so_far = min(best_so_far, float(values.min()))
        evaluations.append(evaluated)
        best_values.append(best_so_far)
        deltas.append(optimizer.delta.ravel().copy())
        if best_so_far <= 1e-8:
            break

    return RunHistory(
        evaluations=np.asarray(evaluations),
        best_values=np.asarray(best_values),
        deltas=np.asarray(deltas),
        optimum=optimum,
        final_x=optimizer.x.ravel().copy(),
    )


def plot_delta_history(history: RunHistory, output: str | Path | None = None):
    """Plot all coordinate-wise expected absolute step sizes over evaluations."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--function", choices=PAPER_FUNCTIONS, default="sphere")
    parser.add_argument("--dimension", type=int, default=20)
    parser.add_argument("--instance", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--budget", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=10)
    parser.add_argument("--c", type=float, default=None)
    parser.add_argument("--eta", type=float, default=None)
    parser.add_argument("--output", type=Path, default=Path("results/barebones"))
    args = parser.parse_args()

    history = run_paper_benchmark(
        kind=args.function,
        dimension=args.dimension,
        instance=args.instance,
        seed=args.seed,
        budget=args.budget,
        population_size=args.population_size,
        c=args.c,
        eta=args.eta,
    )
    stem = f"original_ines_{args.function}_{args.dimension}d"
    delta_path = args.output / f"{stem}_delta.png"
    objective_path = args.output / f"{stem}_objective.png"
    plot_delta_history(history, delta_path)
    plot_objective_history(history, objective_path)

    print("evaluations =", history.evaluations[-1])
    print("best objective =", history.best_values[-1])
    print("x =", history.final_x)
    print("optimum =", history.optimum)
    print("delta plot =", delta_path)
    print("objective plot =", objective_path)


if __name__ == "__main__":
    main()

