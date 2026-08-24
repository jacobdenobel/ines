from typing import Callable, Literal

import ioh
import numpy as np
from numpy.typing import NDArray


TargetKind = Literal["l1", "l0", "target_l1", "target_l0"]


def l1_target_distance(z: NDArray[np.float64]) -> float:
    return float(np.sum(np.abs(z)))


def l0_target_distance(z: NDArray[np.float64]) -> float:
    return float(np.count_nonzero(z))


FUNCTIONS: dict[str, Callable[[NDArray[np.float64]], float]] = {
    "l1": l1_target_distance,
    "target_l1": l1_target_distance,
    "l0": l0_target_distance,
    "target_l0": l0_target_distance,
}


class ShiftedTargetBenchmark:
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
    

def make_target_initial_center(
    problem: ioh.ProblemType,
    rng: np.random.Generator,
    radius: int = 25,
) -> NDArray[np.int_]:
    signs = rng.choice(np.array([-1, 1], dtype=int), size=problem.meta_data.n_variables)
    m0 = problem.optimum.x + signs * radius

    too_low = m0 < problem.bounds.lb
    too_high = m0 > problem.bounds.ub

    m0[too_low] = problem.optimum.x[too_low] + radius
    m0[too_high] = problem.optimum.x[too_high] - radius

    return m0.astype(int)

def make_target_benchmark(
    dim: int,
    kind: TargetKind = "target_l1",
    lb: int = -100,
    ub: int = 100,
    seed: int = 1,
    integer: bool = True,
):
    rng = np.random.default_rng(seed)

    x_opt = rng.integers(-50, 51, size=dim, dtype=int)
    
    try:
        func = FUNCTIONS[kind]
    except KeyError as exc:
        raise ValueError(
            f"Unknown target benchmark kind: {kind!r}. "
            f"Choose one of {sorted(FUNCTIONS)}."
        ) from exc

    benchmark = ShiftedTargetBenchmark(
        func=func,
        x_opt=x_opt,
    )

    name = "integertarget_l1" if kind in {"l1", "target_l1"} else "integertarget_l0"

    problem = ioh.wrap_problem(
        benchmark,
        name=name,
        problem_class=ioh.ProblemClass.INTEGER if integer else ioh.ProblemClass.REAL,
        dimension=dim,
        lb=lb,
        ub=ub,
        calculate_objective=lambda iid, dim: (x_opt, 0.0),
    )
    return problem