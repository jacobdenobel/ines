"""Minimal original and path-free INES variants with a paper-style runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from .benchmarks import PAPER_QUADRATICS, make_quadratic_benchmark
from .distributions import cwise_double_geometric
from .optimizers import IntegerNaturalEvolutionStrategy
from .plotting import RunHistory, plot_delta_history, plot_objective_history


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

    @property
    def p_effective(self) -> np.ndarray:
        q = self.delta / (np.sqrt(1.0 + self.delta**2) + 1.0)
        return 1.0 - q

    def ask(self) -> np.ndarray:
        self.steps = cwise_double_geometric(
            self.rng, self.p_effective, self.population_size
        )
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


class BarebonesNaturalGradientINES:
    """Path-free DG natural gradient with only the learning rate ``eta``."""

    def __init__(
        self,
        x0: np.ndarray,
        delta0: float,
        eta: float,
        population_size: int = 10,
        seed: int | None = None,
        binary: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.x = np.asarray(x0, dtype=int).reshape(-1, 1).copy()
        self.n = len(self.x)
        self.delta = np.full((self.n, 1), delta0, dtype=float)
        self.eta = eta
        self.population_size = population_size
        self.binary = binary

    @property
    def p_effective(self) -> np.ndarray:
        q = self.delta / (np.sqrt(1.0 + self.delta**2) + 1.0)
        return 1.0 - q

    def ask(self) -> np.ndarray:
        self.steps = cwise_double_geometric(
            self.rng, self.p_effective, self.population_size
        )
        candidates = self.x + self.steps
        return candidates & 1 if self.binary else candidates

    def tell(self, values: np.ndarray) -> None:
        best = int(np.argmin(values))
        selected_step = self.steps[:, best, None]
        self.x += selected_step

        fisher = self.delta * np.sqrt(1.0 + self.delta**2)
        natural_gradient = (np.abs(selected_step) - self.delta) / fisher
        self.delta *= np.exp(self.eta * natural_gradient)


Algorithm = Literal["original", "natural-gradient"]


def run_paper_benchmark(
    algorithm: Algorithm = "original",
    kind: str = "sphere",
    dimension: int = 20,
    instance: int = 1,
    seed: int = 1993,
    budget: int | None = None,
    population_size: int = 10,
    c: float | None = None,
    eta: float | None = None,
) -> RunHistory:
    """Run a barebones variant on a central paper quadratic benchmark."""
    if kind not in PAPER_QUADRATICS:
        raise ValueError(f"kind must be one of {PAPER_QUADRATICS}")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if budget is None:
        budget = 10_000 * dimension
    if budget < population_size:
        raise ValueError("budget must allow at least one generation")

    problem = make_quadratic_benchmark(dimension, kind, seed=instance)
    x0 = np.random.default_rng(seed).integers(-50, 51, dimension)
    delta0 = IntegerNaturalEvolutionStrategy.std_to_delta(100.0 / dimension)

    if algorithm == "original":
        optimizer = BarebonesINES(
            x0,
            delta0=delta0,
            population_size=population_size,
            c=c,
            eta=eta,
            seed=seed,
        )
    elif algorithm == "natural-gradient":
        effective_eta = (2.0 / dimension) ** (1.0 / 3.0) if eta is None else eta
        optimizer = BarebonesNaturalGradientINES(
            x0,
            delta0=delta0,
            eta=effective_eta,
            population_size=population_size,
            seed=seed,
        )
    else:
        raise ValueError("algorithm must be 'original' or 'natural-gradient'")

    evaluations: list[int] = []
    best_values: list[float] = []
    deltas: list[np.ndarray] = []
    best_so_far = np.inf
    evaluated = 0

    while evaluated + population_size <= budget:
        points = optimizer.ask()
        values = np.asarray(problem(points.T), dtype=float)
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
        optimum=np.asarray(problem.optimum.x, dtype=int),
        final_x=optimizer.x.ravel().copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("original", "natural-gradient"), default="original"
    )
    parser.add_argument("--function", choices=PAPER_QUADRATICS, default="sphere")
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
        algorithm=args.algorithm,
        kind=args.function,
        dimension=args.dimension,
        instance=args.instance,
        seed=args.seed,
        budget=args.budget,
        population_size=args.population_size,
        c=args.c,
        eta=args.eta,
    )
    stem = f"{args.algorithm}_{args.function}_{args.dimension}d"
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

