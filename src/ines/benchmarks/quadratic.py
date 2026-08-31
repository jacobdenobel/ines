from __future__ import annotations

from typing import Callable, Literal

import ioh
import numpy as np
from numpy.typing import NDArray

BenchmarkKind = Literal[
    "sphere",
    "ellipse",
    "discus",
    "cigar",
    "rippled_sphere",
    "griewank",
    "himmelblau",
]

PAPER_QUADRATICS = ("sphere", "ellipse", "discus", "cigar")


def sphere(z: NDArray[np.float64]) -> float:
    return float(np.linalg.norm(z))


def ellipse(z: NDArray[np.float64]) -> float:
    n = len(z)
    w = np.logspace(np.log10(1), np.log10(10_000), num=n)
    return float(np.linalg.norm(z * w))


def discus(z: NDArray[np.float64]) -> float:
    w = np.ones_like(z)
    w[0] = 10_000
    return float(np.linalg.norm(z * w))


def cigar(z: NDArray[np.float64]) -> float:
    w = np.ones_like(z)
    w[1:] = 10_000
    return float(np.linalg.norm(z * w))


def rippled_sphere(z: NDArray[np.float64]) -> float:
    """
    Separable multimodal integer benchmark.

    f(z) = sum_i z_i^2 + 10(1 - cos(pi z_i))

    Since z is integer-valued in the integer setting, odd coordinates get
    an additional penalty while even coordinates form local valleys.
    """
    z = np.asarray(z, dtype=np.float64)
    return float(np.sum(z**2 + 10.0 * (1.0 - np.cos(np.pi * z))))


def griewank(z: NDArray[np.float64]) -> float:
    """
    Integer Griewank benchmark.

    f(z) = 1 + sum_i z_i^2 / 4000 - prod_i cos(z_i / sqrt(i)).
    """
    z = np.asarray(z, dtype=np.float64)
    i = np.arange(1, z.size + 1, dtype=np.float64)

    return float(1.0 + np.sum(z**2) / 4000.0 - np.prod(np.cos(z / np.sqrt(i))))


def himmelblau_blocks(z: NDArray[np.float64]) -> float:
    """
    Block-separable generalized Himmelblau benchmark.

    Each pair of coordinates forms a shifted Himmelblau block with an
    exact integer optimum at (0, 0). For odd dimensions, the final
    coordinate receives a simple quadratic penalty.

    h(u, v) =
        ((u + 3)^2 + (v + 2) - 11)^2
        +
        ((u + 3) + (v + 2)^2 - 7)^2
    """
    z = np.asarray(z, dtype=np.float64)

    total = 0.0

    for i in range(0, z.size - 1, 2):
        u = z[i]
        v = z[i + 1]

        total += ((u + 3.0) ** 2 + (v + 2.0) - 11.0) ** 2 + (
            (u + 3.0) + (v + 2.0) ** 2 - 7.0
        ) ** 2

    if z.size % 2 == 1:
        total += z[-1] ** 2

    return float(total)


FUNCTIONS: dict[BenchmarkKind, Callable[[NDArray[np.float64]], float]] = {
    "sphere": sphere,
    "ellipse": ellipse,
    "discus": discus,
    "cigar": cigar,
    "rippled_sphere": rippled_sphere,
    "griewank": griewank,
    "himmelblau": himmelblau_blocks,
}


class ShiftedBenchmark:
    def __init__(
        self,
        func: Callable[[NDArray[np.float64]], float],
        x_opt: NDArray[np.int_],
    ) -> None:
        self.func = func
        self.x_opt = x_opt

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=np.float64)
        z = x - self.x_opt
        return self.func(z)


def make_quadratic_benchmark(
    dim: int,
    kind: BenchmarkKind = "ellipse",
    lb: int = -50,
    ub: int = 50,
    seed: int = 1,
    integer: bool = True,
):
    rng = np.random.default_rng(seed)
    x_opt = rng.integers(lb, ub + 1, dim)

    benchmark = ShiftedBenchmark(
        func=FUNCTIONS[kind],
        x_opt=x_opt,
    )

    name = f"{'integer' if integer else ''}{kind}"

    return ioh.wrap_problem(
        benchmark,
        name=name,
        problem_class=ioh.ProblemClass.INTEGER if integer else ioh.ProblemClass.REAL,
        dimension=dim,
        lb=lb,
        ub=ub,
        calculate_objective=lambda iid, dim: (x_opt, 0.0),
    )

