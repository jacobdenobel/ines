from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Literal

import ioh
import numpy as np
from numpy.typing import NDArray

from ines import IntegerNaturalEvolutionStrategy
from ines.benchmarks import (
    make_binary_benchmark,
    make_quadratic_benchmark,
    make_target_benchmark,
    make_target_initial_center,
)
from ines.optimizers.recombination import CenterUpdateKind, SufficientStatisticKind

BenchmarkSuite = Literal["quadratic", "pbo", "target"]


PBO_IDS: dict[str, int] = {
    "onemax": 1,
    "leadingones": 2,
}


@dataclass(frozen=True)
class BenchmarkResult:
    algorithm_name: str
    problem_name: str
    problem_id: int
    dimension: int
    values: NDArray[np.float64]
    times: NDArray[np.int_]
    ert: float


def ert(
    times: NDArray[np.integer],
    values: NDArray[np.floating],
    target: float,
) -> float:
    successes = values <= target
    n_successes = int(np.sum(successes))

    if n_successes == 0:
        return float("inf")

    return float(np.sum(times) / n_successes)


def make_problem(
    suite: BenchmarkSuite,
    kind: str,
    dim: int,
    instance: int,
) -> ioh.ProblemType:
    if suite == "quadratic":
        return make_quadratic_benchmark(
            dim=dim,
            kind=kind,
            seed=instance,
            integer=True,
        )

    if suite == "target":
        return make_target_benchmark(
            dim=dim,
            kind=kind,
            seed=instance,
            integer=True,
        )

    if suite == "pbo":
        if kind.lower() in PBO_IDS:
            return make_binary_benchmark(
                dim=dim,
                kind=kind.lower(),
                seed=instance,
            )

        fid = PBO_IDS.get(kind.lower(), None)

        if fid is None:
            try:
                fid = int(kind)
            except ValueError as exc:
                raise ValueError(
                    f"Unknown PBO kind: {kind!r}. "
                    f"Use one of {sorted(PBO_IDS)} or pass a PBO function id."
                ) from exc

        return ioh.get_problem(
            fid,
            instance,
            dim,
            ioh.ProblemClass.PBO,
        )

    raise ValueError(f"Unknown benchmark suite: {suite!r}")


def run_single_es(
    problem: ioh.ProblemType,
    budget: int,
    target: float,
    seed: int,
    lambda_: int,
    mu: int,
    center_update_kind: CenterUpdateKind,
    sufficient_statistic_kind: SufficientStatisticKind,
    save_deltas: bool = False,
    random_state: np.random.RandomState | None = None,
    **es_kwargs,
) -> list[NDArray[np.float64]]:
    if "target" in problem.meta_data.name:
        rng = np.random.default_rng(seed)

        if "l0" in problem.meta_data.name:
            es_kwargs["x0"] = make_target_initial_center(problem, rng, radius=1)
            es_kwargs["delta0"] = 1.0 / problem.meta_data.n_variables
        else:
            es_kwargs["x0"] = make_target_initial_center(problem, rng, radius=25)
            # es.kwargs['delta0'] = 1

    if random_state is not None and "x0" not in es_kwargs:
        es_kwargs["x0"] = random_state.randint(
            problem.bounds.lb,
            problem.bounds.ub + 1,
        )

    es = IntegerNaturalEvolutionStrategy.from_problem(
        problem,
        seed=seed,
        lambda_=lambda_,
        mu=mu,
        center_update_kind=center_update_kind,
        sufficient_statistic_kind=sufficient_statistic_kind,
        **es_kwargs,
    )

    # integer-es used NumPy's process-global RandomState for both the initial
    # center and every mutation, without reseeding between repetitions. Keep
    # that historical protocol opt-in; the public API remains on Generator.
    if random_state is not None:
        es.rng = random_state

    deltas: list[NDArray[np.float64]] = []

    while problem.state.evaluations <= (budget - es.lambda_):
        if problem.state.current_best.y <= target:
            break

        X = es.ask()
        f = np.asarray(problem(X.T), dtype=np.float64)
        es.tell(X, f)

        if save_deltas:
            deltas.append(es.delta.copy().ravel())

    return deltas


def run_benchmark(
    algorithm_name: str,
    suite: BenchmarkSuite,
    kind: str,
    dim: int,
    n_rep: int = 25,
    budget: int | None = None,
    target: float = 1e-8,
    lambda_: int = 10,
    mu: int = 5,
    seed: int = 1993,
    instance: int = 1,
    save_deltas: bool = False,
    output_dir: str | Path = "data",
    center_update_kind: CenterUpdateKind = CenterUpdateKind.WEIGHTED_DISCRETE,
    sufficient_statistic_kind: SufficientStatisticKind = SufficientStatisticKind.WEIGHTED,
    random_state: np.random.RandomState | None = None,
) -> BenchmarkResult:
    problem = make_problem(
        suite=suite,
        kind=kind,
        dim=dim,
        instance=instance,
    )

    if budget is None:
        budget = int(problem.meta_data.n_variables * 1e4)

    values: list[float] = []
    times: list[int] = []
    all_deltas: list[list[NDArray[np.float64]]] = []

    for ridx in range(n_rep):
        run_seed = seed + ridx

        deltas = run_single_es(
            problem=problem,
            budget=budget,
            target=target,
            seed=run_seed,
            lambda_=lambda_,
            mu=mu,
            center_update_kind=center_update_kind,
            sufficient_statistic_kind=sufficient_statistic_kind,
            save_deltas=save_deltas,
            random_state=random_state,
        )

        values.append(float(problem.state.current_best.y))
        times.append(int(problem.state.evaluations))

        if save_deltas:
            all_deltas.append(deltas)

        problem.reset()

    values_array = np.asarray(values, dtype=np.float64)
    times_array = np.asarray(times, dtype=int)

    result = BenchmarkResult(
        algorithm_name=algorithm_name,
        problem_name=problem.meta_data.name,
        problem_id=problem.meta_data.problem_id,
        dimension=problem.meta_data.n_variables,
        values=values_array,
        times=times_array,
        ert=ert(times_array, values_array, target),
    )

    if save_deltas:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        target_file = (
            output_path
            / f"{problem.meta_data.name}_{problem.meta_data.n_variables}_deltas.pkl"
        )

        with target_file.open("wb") as handle:
            pickle.dump(all_deltas, handle)

    return result
