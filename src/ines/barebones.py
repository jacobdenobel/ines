"""Minimal original and path-free INES variants with a paper-style runner."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import numpy as np

from .benchmarks import (
    PAPER_QUADRATICS,
    evaluate_binary_benchmark,
    evaluate_quadratic_benchmark,
    make_binary_benchmark,
    make_quadratic_benchmark,
)
from .distributions import cwise_double_geometric
from .plotting import (
    RunHistory,
    plot_delta_history,
    plot_l1_distance_history,
    plot_objective_history,
)


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
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed) if rng is None else rng
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

    def ask(self, n_samples: int | None = None) -> np.ndarray:
        sample_count = self.population_size if n_samples is None else n_samples
        self.steps = cwise_double_geometric(self.rng, self.p_effective, sample_count)
        candidates = self.x + self.steps
        self.candidates = candidates & 1 if self.binary else candidates
        return self.candidates

    def tell(self, values: np.ndarray) -> None:
        """Update from objective values in the same column order as ``ask``."""
        best = int(np.argmin(values))
        selected_step = self.steps[:, best, None]
        self.x = self.candidates[:, best, None].copy()

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
        rng: np.random.Generator | np.random.RandomState | None = None,
    ) -> None:
        self.rng = np.random.default_rng(seed) if rng is None else rng
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

    def ask(self, n_samples: int | None = None) -> np.ndarray:
        sample_count = self.population_size if n_samples is None else n_samples
        self.steps = cwise_double_geometric(self.rng, self.p_effective, sample_count)
        candidates = self.x + self.steps
        self.candidates = candidates & 1 if self.binary else candidates
        return self.candidates

    def tell(self, values: np.ndarray) -> None:
        best = int(np.argmin(values))
        selected_step = self.steps[:, best, None]
        self.x = self.candidates[:, best, None].copy()

        fisher = self.delta * np.sqrt(1.0 + self.delta**2)
        natural_gradient = (np.abs(selected_step) - self.delta) / fisher
        self.delta *= np.exp(self.eta * natural_gradient)


Algorithm = Literal["original", "natural-gradient"]
PAPER_PBO = ("onemax", "leadingones")


def count_paper_evaluations(steps: np.ndarray) -> int:
    """Count offspring whose raw mutation is not the all-zero vector."""
    return int(np.count_nonzero(np.any(steps != 0, axis=0)))


def _evaluate_candidates(
    problem,
    candidates: np.ndarray,
    steps: np.ndarray,
    parent_value: float,
    reuse_zero_steps: bool,
) -> tuple[np.ndarray, int]:
    """Evaluate candidates, reusing the parent value for zero mutations."""
    if not reuse_zero_steps:
        values = np.asarray(problem(candidates.T), dtype=float).reshape(-1)
        return values, candidates.shape[1]

    changed = np.any(steps != 0, axis=0)
    values = np.full(candidates.shape[1], parent_value, dtype=float)
    if np.any(changed):
        values[changed] = np.asarray(
            problem(candidates[:, changed].T), dtype=float
        ).reshape(-1)
    return values, int(np.count_nonzero(changed))


def _draw_initial_center(
    rng: np.random.Generator | np.random.RandomState,
    low: int,
    high: int,
    dimension: int,
) -> np.ndarray:
    if isinstance(rng, np.random.RandomState):
        return rng.randint(low, high, dimension)
    return rng.integers(low, high, dimension)


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
    random_state: np.random.RandomState | None = None,
    paper_evaluation_counting: bool = True,
) -> RunHistory:
    """Run a barebones variant with the paper's evaluation protocol."""
    if kind not in (*PAPER_QUADRATICS, *PAPER_PBO):
        raise ValueError(f"kind must be one of {(*PAPER_QUADRATICS, *PAPER_PBO)}")
    if dimension < 2:
        raise ValueError("dimension must be at least 2")
    if population_size < 2:
        raise ValueError("population_size must be at least 2")
    if budget is None:
        budget = 10_000 * dimension
    if budget < population_size:
        raise ValueError("budget must allow at least one generation")

    binary = kind in PAPER_PBO
    if binary:
        problem = make_binary_benchmark(dimension, kind, seed=instance)
    else:
        problem = make_quadratic_benchmark(dimension, kind, seed=instance)

    rng = np.random.default_rng(seed) if random_state is None else random_state
    x0 = (
        _draw_initial_center(rng, 0, 2, dimension)
        if binary
        else _draw_initial_center(rng, -50, 51, dimension)
    )
    delta0 = (
        1.0 / dimension
        if binary
        else (100.0 / dimension) ** 2 / np.sqrt(2.0 * (100.0 / dimension) ** 2 + 1.0)
    )
    optimum = np.asarray(problem.optimum.x, dtype=int)
    parent_value = (
        evaluate_binary_benchmark(kind, x0, optimum)
        if binary
        else evaluate_quadratic_benchmark(kind, x0, optimum)
    )

    if algorithm == "original":
        optimizer = BarebonesINES(
            x0,
            delta0=delta0,
            population_size=population_size,
            c=c,
            eta=eta,
            seed=seed,
            binary=binary,
            rng=rng,
        )
    elif algorithm == "natural-gradient":
        effective_eta = (2.0 / dimension) ** (1.0 / 3.0) if eta is None else eta
        optimizer = BarebonesNaturalGradientINES(
            x0,
            delta0=delta0,
            eta=effective_eta,
            population_size=population_size,
            seed=seed,
            binary=binary,
            rng=rng,
        )
    else:
        raise ValueError("algorithm must be 'original' or 'natural-gradient'")

    evaluations: list[int] = []
    function_values: list[float] = []
    best_values: list[float] = []
    l1_distances: list[float] = []
    deltas: list[np.ndarray] = []
    best_so_far = np.inf
    evaluated = 0
    sampled = 0
    early_stop_sampling = dimension < 10

    while sampled < budget:
        generation_size = min(population_size, budget - sampled)
        if not early_stop_sampling and generation_size < population_size:
            break
        if early_stop_sampling:
            generation_steps: list[np.ndarray] = []
            generation_points: list[np.ndarray] = []
            generation_values: list[float] = []
            generation_evaluated = 0
            for _ in range(generation_size):
                point = optimizer.ask(1)
                candidate_values, candidate_evaluated = _evaluate_candidates(
                    problem,
                    point,
                    optimizer.steps,
                    parent_value,
                    paper_evaluation_counting,
                )
                value = float(candidate_values[0])
                generation_steps.append(optimizer.steps.copy())
                generation_points.append(point.copy())
                generation_values.append(value)
                generation_evaluated += candidate_evaluated
                sampled += 1
                if value <= 1e-8:
                    break
            optimizer.steps = np.concatenate(generation_steps, axis=1)
            optimizer.candidates = np.concatenate(generation_points, axis=1)
            values = np.asarray(generation_values, dtype=float)
        else:
            points = optimizer.ask()
            values, generation_evaluated = _evaluate_candidates(
                problem,
                points,
                optimizer.steps,
                parent_value,
                paper_evaluation_counting,
            )
            sampled += population_size

        selected = int(np.argmin(values))
        optimizer.tell(values)
        parent_value = float(values[selected])

        evaluated += generation_evaluated
        selected_value = float(values.min())
        best_so_far = min(best_so_far, selected_value)
        evaluations.append(evaluated)
        function_values.append(selected_value)
        best_values.append(best_so_far)
        l1_distances.append(
            float(
                np.linalg.norm(
                    optimizer.x.ravel() - np.asarray(problem.optimum.x), ord=1
                )
            )
        )
        deltas.append(optimizer.delta.ravel().copy())
        if best_so_far <= 1e-8:
            break

    return RunHistory(
        evaluations=np.asarray(evaluations),
        function_values=np.asarray(function_values),
        best_values=np.asarray(best_values),
        l1_distances=np.asarray(l1_distances),
        deltas=np.asarray(deltas),
        optimum=np.asarray(problem.optimum.x, dtype=int),
        final_x=optimizer.x.ravel().copy(),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--algorithm", choices=("original", "natural-gradient"), default="original"
    )
    parser.add_argument(
        "--function", choices=(*PAPER_QUADRATICS, *PAPER_PBO), default="sphere"
    )
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
    l1_path = args.output / f"{stem}_l1_distance.png"
    plot_delta_history(history, delta_path)
    plot_objective_history(history, objective_path)
    plot_l1_distance_history(history, l1_path)

    print("evaluations =", history.evaluations[-1])
    print("best objective =", history.best_values[-1])
    print("x =", history.final_x)
    print("optimum =", history.optimum)
    print("delta plot =", delta_path)
    print("objective plot =", objective_path)
    print("L1-distance plot =", l1_path)


if __name__ == "__main__":
    main()
