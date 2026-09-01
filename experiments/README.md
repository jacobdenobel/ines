# Reproducing the paper experiments

This directory contains the reproducibility entry point for *Integer Natural
Evolution Strategies*. Commands are run from the repository root after
installing `.[examples]` (or `.[dev,examples]`). Every command is deterministic
for a fixed `--seed`.

The executable driver is [`reproduce.py`](./reproduce.py). It uses the original
paper implementation in [`../src/ines/barebones.py`](../src/ines/barebones.py),
the shared benchmarks in
[`../src/ines/benchmarks/`](../src/ines/benchmarks/), and the shared plotting
helpers in [`../src/ines/plotting.py`](../src/ines/plotting.py).

## Common protocol

- Quadratic optimum: independently sampled, uniformly, from `[-50, 50]^n`.
  Seeds match the original `integer-es` enumeration of the Cartesian product
  of dimensions and functions.
- Initial center: uniformly sampled from the same box by the barebones paper
  runner.
- Dimensions: `2, 3, 5, 10, 20, 40, 100`.
- Repetitions: 25 independent runs.
- Budget: `10,000 n` objective evaluations per run.
- Target: objective value 0 (implemented with tolerance `1e-8`).
- Population: fixed `lambda = 10`, `mu = 1`, matching the INES code used for
  the paper. The dimension-dependent CMA-ES default is used only by CMA-IH.
- Initial scale: `sigma0 = 100/n` and
  `delta0 = sigma0^2 / sqrt(2 sigma0^2 + 1)`.
- Adaptation: `c = 1 - 1.5/n`, `eta = (2/n)^(1/3)`.
- ERT: total evaluations (including the full budgets of failed runs) divided
  by the number of successful runs; infinity means zero successes.
- Implementation: all INES rows are run with the original single-parent
  `BarebonesINES`; the recombination-capable API is not used for paper results.
- Paper evaluation count: all-zero raw mutations reuse the cached parent value
  and are not submitted to IOH. For `n < 10`, offspring are sampled
  sequentially and sampling stops as soon as the optimum is found, so the
  final generation can contain fewer than `lambda` offspring.
- RNG: the reproduction command defaults to `--rng-protocol integer-es`, which
  reuses one legacy NumPy stream across repetitions as the source repository
  did. `--rng-protocol independent` provides isolated `seed + repetition`
  streams as a separate, modern protocol.

The four objectives are `||W(x-x*)||_2`: Sphere uses all unit weights;
Ellipse uses log-spaced weights from 1 to `10^4`; Discus uses `w1=10^4` and
unit remaining weights; Cigar uses `w1=1` and `10^4` elsewhere.

## One-command reproduction

```bash
python experiments/reproduce.py all --output results
```

This creates:

- `results/performance/ert.csv`: per-function, per-dimension ERT for INES,
  CMA-IH, and CMA-IH-sep;
- `results/deltas/*.pkl`: raw per-generation `delta_i` trajectories;
- `results/figures/quadratic_step_sizes_20d.pdf`: median and interquartile
  trajectories on the four quadratic functions;
- `results/performance/pbo.csv`: OneMax and LeadingOnes ERTs for dimensions
  `2, 3, 5, 10, 20, 40, 100, 200, 500`, including the manuscript values as
  separate audit columns;
- `results/figures/pbo_step_sizes_500d.png`: OneMax and LeadingOnes trajectories.

Use `--quick` for two repetitions, dimensions 2 and 5, and reduced budgets.
It validates the pipeline but does not reproduce the reported statistics.

Individual stages are also available:

```bash
python experiments/reproduce.py performance --output results
python experiments/reproduce.py step-sizes --output results
python experiments/reproduce.py pbo --output results
```

## Parameter calibration

The original `integer-es` grid search used dimensions `2, 3, 5, 10`, 15 runs,
a budget of `200n`, and 60 equally spaced values from `1e-4` to `1-1e-4`.
It evaluated `eta` on that interval and `c` on 1.5 times that interval for
Sphere and Ellipse. The fitted laws `c(n)=1-1.5/n` and
`eta(n)=(2/n)^(1/3)` are the paper configuration used by the other stages.

## CMA-IH baselines

The adapter in [`cma_ih.py`](./cma_ih.py) is extracted from `integer-es` commit
`ae0fdb80cb693eddf4ecb959e3319fc6b279a058`. It uses pycma integer variables,
the library's default population size, `sigma0=100/n`, zero function-tolerance
termination, and either full covariance (`CMA-IH`) or `CMA_diagonal=True`
(`CMA-IH-sep`). All methods receive the same optimum and repetition seed.

## Step-size aggregation

Runs terminate independently. Each coordinate trajectory is linearly
interpolated onto 101 equally spaced relative-progress points in `[0,1]`.
Figures show the pointwise median and 25th/75th percentiles across 25 runs,
matching the manuscript's method.

For binary problems, mutations are mapped cyclically with `(x+z) mod 2` and
the initial expected absolute step is `delta_i=1/n`.

## PBO evaluation accounting

For `n < 10`, the paper stops the offspring loop immediately when an optimum is
found. Across every dimension, all-zero raw mutations are omitted from the
reported function-evaluation count. These rules account for ERTs below
`lambda=10`. See [`PBO_DIAGNOSIS.md`](./PBO_DIAGNOSIS.md) for the exact protocol.

