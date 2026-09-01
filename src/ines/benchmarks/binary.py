from __future__ import annotations

from typing import Callable, Literal

import ioh
import numpy as np
from numpy.typing import NDArray

BinaryKind = Literal["onemax", "leadingones"]


def onemax_loss(matches: NDArray[np.int_]) -> float:
    return float(len(matches) - np.sum(matches))


def leadingones_loss(matches: NDArray[np.int_]) -> float:
    mismatch = matches == 0
    if not np.any(mismatch):
        return 0.0
    return float(len(matches) - np.argmax(mismatch))


FUNCTIONS: dict[BinaryKind, Callable[[NDArray[np.int_]], float]] = {
    "onemax": onemax_loss,
    "leadingones": leadingones_loss,
}


def evaluate_binary_benchmark(
    kind: BinaryKind,
    x: NDArray[np.int_],
    x_opt: NDArray[np.int_],
) -> float:
    """Evaluate a shifted binary benchmark without an IOH logger call."""
    matches = 1 - (np.asarray(x, dtype=int) ^ np.asarray(x_opt, dtype=int))
    return FUNCTIONS[kind](matches)


class ShiftedBinaryBenchmark:
    def __init__(self, func, x_opt: NDArray[np.int_]) -> None:
        self.func = func
        self.x_opt = x_opt

    def __call__(self, x) -> float:
        x = np.asarray(x, dtype=int)
        matches = 1 - (x ^ self.x_opt)
        return self.func(matches)


def make_binary_benchmark(dim: int, kind: BinaryKind, seed: int = 1):
    """Create the minimization-form PBO benchmark used by integer-es."""
    try:
        func = FUNCTIONS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown binary benchmark kind: {kind!r}; choose one of {sorted(FUNCTIONS)}"
        ) from exc

    rng = np.random.default_rng(seed)
    x_opt = rng.integers(0, 2, size=dim, dtype=int)
    benchmark = ShiftedBinaryBenchmark(func, x_opt)
    return ioh.wrap_problem(
        benchmark,
        name=kind,
        problem_class=ioh.ProblemClass.INTEGER,
        dimension=dim,
        lb=0,
        ub=1,
        calculate_objective=lambda iid, n: (x_opt, 0.0),
    )
