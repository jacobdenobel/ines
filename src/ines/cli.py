import argparse
import math

from ines.runner import run_benchmark
from ines.optimizers.recombination import CenterUpdateKind, SufficientStatisticKind


def parse_dimensions(value: str) -> list[int]:
    dimensions = []

    for item in value.split(","):
        item = item.strip()

        if not item:
            continue

        dim = int(item)

        if dim <= 0:
            raise argparse.ArgumentTypeError("dimensions must be positive")

        dimensions.append(dim)

    if not dimensions:
        raise argparse.ArgumentTypeError("at least one dimension is required")

    return dimensions


def parse_center_update(value: str) -> CenterUpdateKind:
    try:
        return CenterUpdateKind(value)
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in CenterUpdateKind)
        raise argparse.ArgumentTypeError(
            f"invalid center update {value!r}; choose one of: {valid}"
        ) from exc


def parse_sufficient_statistic(value: str) -> SufficientStatisticKind:
    try:
        return SufficientStatisticKind(value)
    except ValueError as exc:
        valid = ", ".join(kind.value for kind in SufficientStatisticKind)
        raise argparse.ArgumentTypeError(
            f"invalid sufficient statistic {value!r}; choose one of: {valid}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ines",
        description="Run INES benchmarks.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser(
        "benchmark",
        help="Run an INES benchmark.",
    )

    benchmark.add_argument(
        "--suite",
        choices=["quadratic", "pbo", "target"],
        default="quadratic",
    )
    benchmark.add_argument(
        "--kind",
        default="sphere",
        help=(
            "Benchmark kind, e.g. sphere, ellipse, discus, cigar, "
            "onemax, leadingones, or a PBO id."
        ),
    )

    dimension_group = benchmark.add_mutually_exclusive_group(required=True)
    dimension_group.add_argument(
        "--dim",
        type=int,
        help="Single dimension.",
    )
    dimension_group.add_argument(
        "--dims",
        type=parse_dimensions,
        help="Comma-separated dimensions, e.g. 2,3,5,10,20,40,100.",
    )

    benchmark.add_argument("--reps", type=int, default=25)
    benchmark.add_argument("--budget", type=int, default=None)
    benchmark.add_argument("--target", type=float, default=1e-8)
    benchmark.add_argument("--lambda", dest="lambda_", type=int, default=None)
    benchmark.add_argument("--mu", type=int, default=None)
    benchmark.add_argument("--seed", type=int, default=1993)
    benchmark.add_argument("--instance", type=int, default=1)
    benchmark.add_argument("--algorithm-name", default="INES")
    benchmark.add_argument("--save-deltas", action="store_true")
    benchmark.add_argument("--output-dir", default="data")

    benchmark.add_argument(
        "--center",
        type=parse_center_update,
        default=CenterUpdateKind.BEST,
        help=(
            "Center recombination operator. Choices: "
            + ", ".join(kind.value for kind in CenterUpdateKind)
        ),
    )

    benchmark.add_argument(
        "--statistic",
        type=parse_sufficient_statistic,
        default=SufficientStatisticKind.BEST,
        help=(
            "Sufficient-statistic recombination operator. Choices: "
            + ", ".join(kind.value for kind in SufficientStatisticKind)
        ),
    )

    return parser.parse_args()


def format_ert(value: float) -> str:
    return "inf" if math.isinf(value) else f"{value:.1f}"


def main() -> None:
    args = parse_args()

    dimensions = [args.dim] if args.dim is not None else args.dims

    for dim in dimensions:
        lambda_ = args.lambda_
        if lambda_ is None:
            lambda_ = 10

        mu = args.mu
        if mu is None:
            mu = 1

        result = run_benchmark(
            algorithm_name=args.algorithm_name,
            suite=args.suite,
            kind=args.kind,
            dim=dim,
            n_rep=args.reps,
            budget=args.budget,
            target=args.target,
            lambda_=lambda_,
            mu=mu,
            seed=args.seed,
            instance=args.instance,
            save_deltas=args.save_deltas,
            output_dir=args.output_dir,
            center_update_kind=args.center,
            sufficient_statistic_kind=args.statistic,
        )

        print(
            result.algorithm_name,
            f"({result.problem_id}, {result.problem_name}, {result.dimension}D)",
            f"mu={mu}",
            f"lambda={lambda_}",
            f"center={args.center.value}",
            f"statistic={args.statistic.value}",
            f"avg f: {result.values.mean():.2e} +- {result.values.std():.2e}",
            f"ERT: {format_ert(result.ert)}",
        )


if __name__ == "__main__":
    main()

