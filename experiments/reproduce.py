#!/usr/bin/env python3
"""Reproduce the INES experiments and figures from the paper."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from ines.runner import run_benchmark
from ines.optimizers.recombination import CenterUpdateKind, SufficientStatisticKind
from ines.runner import ert, make_problem

from cma_ih import run_cma_ih

QUADRATICS = ("sphere", "ellipse", "discus", "cigar")
PBO = ("onemax", "leadingones")
PAPER_DIMS = (2, 3, 5, 10, 20, 40, 100)
PAPER_LAMBDA = 10
PAPER_MU = 1


def benchmark_instance(kind: str, dim: int) -> int:
    """Match integer-es's product(dimensions, functions) enumeration."""
    return PAPER_DIMS.index(dim) * len(QUADRATICS) + QUADRATICS.index(kind) + 1


def run_suite(
    output: Path,
    suite: str,
    kinds: tuple[str, ...],
    dims: tuple[int, ...],
    reps: int,
    budget_multiplier: int,
    seed: int,
    save_deltas: bool,
) -> list[dict[str, object]]:
    delta_dir = output / "deltas"
    rows: list[dict[str, object]] = []

    for kind in kinds:
        for dim in dims:
            result = run_benchmark(
                algorithm_name="INES",
                suite=suite,
                kind=kind,
                dim=dim,
                n_rep=reps,
                budget=budget_multiplier * dim,
                target=1e-8,
                lambda_=PAPER_LAMBDA,
                mu=PAPER_MU,
                seed=seed,
                instance=(benchmark_instance(kind, dim) if suite == "quadratic" else 1),
                save_deltas=save_deltas,
                output_dir=delta_dir,
                center_update_kind=CenterUpdateKind.BEST,
                sufficient_statistic_kind=SufficientStatisticKind.BEST,
            )
            rows.append(
                {
                    "algorithm": "INES",
                    "suite": suite,
                    "function": kind,
                    "dimension": dim,
                    "repetitions": reps,
                    "budget": budget_multiplier * dim,
                    "successes": int(np.sum(result.values <= 1e-8)),
                    "ert": result.ert,
                    "mean_final_value": float(result.values.mean()),
                }
            )
            print(kind, dim, "ERT", result.ert)
    return rows


def run_cma_suite(
    kinds: tuple[str, ...],
    dims: tuple[int, ...],
    reps: int,
    budget_multiplier: int,
    seed: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for kind in kinds:
        for dim in dims:
            for label, separable in (("CMA-IH", False), ("CMA-IH-sep", True)):
                problem = make_problem(
                    "quadratic", kind, dim, benchmark_instance(kind, dim)
                )
                values: list[float] = []
                times: list[int] = []
                for repetition in range(reps):
                    value, evaluations = run_cma_ih(
                        problem,
                        budget=budget_multiplier * dim,
                        target=1e-8,
                        seed=seed + repetition,
                        separable=separable,
                    )
                    values.append(value)
                    times.append(evaluations)
                    problem.reset()
                values_array = np.asarray(values)
                times_array = np.asarray(times)
                rows.append(
                    {
                        "algorithm": label,
                        "suite": "quadratic",
                        "function": kind,
                        "dimension": dim,
                        "repetitions": reps,
                        "budget": budget_multiplier * dim,
                        "successes": int(np.sum(values_array <= 1e-8)),
                        "ert": ert(times_array, values_array, 1e-8),
                        "mean_final_value": float(values_array.mean()),
                    }
                )
                print(label, kind, dim, "ERT", rows[-1]["ert"])
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def resample(run: list[np.ndarray], points: int = 101) -> np.ndarray:
    values = np.asarray(run, dtype=float)
    if values.ndim != 2 or len(values) == 0:
        raise ValueError("A saved delta run must contain at least one generation")
    source = np.linspace(0.0, 1.0, len(values))
    target = np.linspace(0.0, 1.0, points)
    return np.stack(
        [np.interp(target, source, values[:, i]) for i in range(values.shape[1])],
        axis=1,
    )


def load_trajectories(path: Path) -> np.ndarray:
    with path.open("rb") as handle:
        runs = pickle.load(handle)
    nonempty = [run for run in runs if run]
    if not nonempty:
        raise ValueError(f"No trajectories found in {path}")
    return np.stack([resample(run) for run in nonempty])


def find_delta_file(delta_dir: Path, token: str, dim: int) -> Path:
    suffix = f"_{dim}_deltas.pkl"
    matches = sorted(
        path
        for path in delta_dir.glob("*_deltas.pkl")
        if token.lower() in path.name.lower() and path.name.endswith(suffix)
    )
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one delta file for {token}/{dim}, found {matches}"
        )
    return matches[0]


def plot_quadratics(output: Path, dim: int = 20) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(9, 6), sharex=True)
    tau = np.linspace(0, 1, 101)
    for ax, kind in zip(axes.flat, QUADRATICS):
        data = load_trajectories(find_delta_file(output / "deltas", kind, dim))
        median = np.median(data, axis=0)
        q1, q3 = np.quantile(data, [0.25, 0.75], axis=0)
        for coordinate in range(dim):
            ax.plot(tau, median[:, coordinate], linewidth=0.8)
            ax.fill_between(tau, q1[:, coordinate], q3[:, coordinate], alpha=0.08)
        ax.set_title(kind.capitalize())
        ax.set_yscale("log")
        ax.set_ylabel(r"$\delta_i$")
    for ax in axes[-1]:
        ax.set_xlabel("relative run progress")
    figure.tight_layout()
    target = output / "figures" / f"quadratic_step_sizes_{dim}d.pdf"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, bbox_inches="tight")
    plt.close(figure)


def plot_pbo(output: Path, dim: int = 500) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    tau = np.linspace(0, 1, 101)
    for ax, kind in zip(axes, PBO):
        data = load_trajectories(find_delta_file(output / "deltas", kind, dim))
        median = np.median(data, axis=0)
        for coordinate in range(dim):
            ax.plot(tau, median[:, coordinate], linewidth=0.35)
        ax.set_title(kind.capitalize())
        ax.set_xlabel("relative run progress")
        ax.set_yscale("log")
    axes[0].set_ylabel(r"$\delta_i$")
    figure.tight_layout()
    target = output / "figures" / f"pbo_step_sizes_{dim}d.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(target, dpi=200, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("all", "performance", "step-sizes", "pbo"))
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--seed", type=int, default=1993)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    reps = 2 if args.quick else 25
    dims = (2, 5) if args.quick else PAPER_DIMS
    budget_multiplier = 200 if args.quick else 10_000

    if args.stage in {"all", "performance", "step-sizes"}:
        rows = run_suite(
            args.output,
            "quadratic",
            QUADRATICS,
            dims,
            reps,
            budget_multiplier,
            args.seed,
            save_deltas=args.stage in {"all", "step-sizes"},
        )
        if args.stage in {"all", "performance"}:
            rows.extend(
                run_cma_suite(
                    QUADRATICS,
                    dims,
                    reps,
                    budget_multiplier,
                    args.seed,
                )
            )
        write_csv(args.output / "performance" / "ert.csv", rows)
        if args.stage in {"all", "step-sizes"}:
            plot_dim = 5 if args.quick else 20
            if plot_dim not in dims:
                raise ValueError(f"Plot dimension {plot_dim} was not run")
            plot_quadratics(args.output, plot_dim)

    if args.stage in {"all", "pbo"}:
        pbo_dim = 20 if args.quick else 500
        rows = run_suite(
            args.output,
            "pbo",
            PBO,
            (pbo_dim,),
            reps,
            budget_multiplier,
            args.seed,
            save_deltas=True,
        )
        write_csv(args.output / "performance" / "pbo.csv", rows)
        plot_pbo(args.output, pbo_dim)


if __name__ == "__main__":
    main()

