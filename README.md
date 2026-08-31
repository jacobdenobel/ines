# INES: Integer Natural Evolution Strategies

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

INES is a lattice-native `(1, lambda)` evolution strategy for black-box integer
optimization. It samples coordinate-wise mutations from the double geometric
(discrete Laplace) distribution and adapts each coordinate's expected absolute
step size with a Fisher-normalized natural-gradient signal and a fading-memory
evolution path.

This repository is the public reference implementation accompanying the paper
*Integer Natural Evolution Strategies* by Jacob de Nobel et al.

## Installation

INES requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Quick start

```python
import numpy as np
from ines import IntegerNaturalEvolutionStrategy

def objective(x):
    return np.sum((x - np.array([3, -2, 7])) ** 2, axis=0)

es = IntegerNaturalEvolutionStrategy(
    x0=np.zeros(3, dtype=int),
    delta0=2.0,
    lambda_=10,
    seed=1,
)

for _ in range(1_000):
    candidates = es.ask()                 # shape: (dimension, lambda)
    values = objective(candidates)        # one value per column
    es.tell(candidates, values)
    if values.min() == 0:
        break

print(es.m.ravel(), es.delta.ravel())
```

`ask()` returns candidates as columns. Pass objective values in the same order
to `tell()`. For bounded problems, clip or repair candidates before evaluation;
the optimizer intentionally does not impose a constraint-handling policy.

The default update is the deliberately simple reference algorithm from
`integer-es`. An optional `stabilize=True` mode adds log-space clipping, a
`1/n` lower dispersion floor, and anti-windup for exceptionally long runs. It
is an extension and is not used by the paper-reproduction commands.

The two shortest implementations now live together in
[`src/ines/barebones.py`](src/ines/barebones.py): original evolution-path INES
and a path-free natural-gradient variant with only `eta` and no
`max(Var[|Z|], 1)` denominator floor. 

Run either variant on the shifted Sphere, Ellipse, Discus, and Cigar objectives
used in the paper. Each run records the selected and best-so-far objective,
the center's L1 distance to the optimum, and coordinate-wise `delta` histories.
It saves three plots: objective, L1 distance, and `delta`. 

```bash
python -m ines.barebones --algorithm original \
  --function ellipse --dimension 20
python -m ines.barebones --algorithm natural-gradient \
  --function ellipse --dimension 20 --eta 0.1
```

An IOH problem can initialize the domain and binary/integer mode automatically:

```python
from ines import IntegerNaturalEvolutionStrategy, make_quadratic_benchmark

problem = make_quadratic_benchmark(dim=20, kind="ellipse", seed=1)
es = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=1993)
```

## Command line

Run 25 repetitions of the paper's 20-dimensional Ellipse benchmark:

```bash
ines benchmark --suite quadratic --kind ellipse --dim 20 --reps 25 \
  --budget 200000 --save-deltas --output-dir results/deltas
```

Useful options include `--dims 2,3,5,10,20,40,100`, `--lambda`, `--mu`,
`--seed`, and `--target`. Explicit `--lambda` and `--mu` values are preserved.
Use `ines benchmark --help` for the complete interface.

## Reproducing the paper experiments

The full protocol, randomization scheme, output layout, and figure-generation
commands are documented in [experiments/README.md](experiments/README.md).
The bundled driver reproduces the INES calibration, coordinate-wise step-size
trajectories (quadratic and pseudo-Boolean), and quadratic performance runs:

```bash
python experiments/reproduce.py all --output results
```

For a quick test, add `--quick`. Full runs use 25 repetitions and a budget of
`10^4 n` evaluations and can take many hours, especially at `n=500` and
`n=1000`.

## Algorithm defaults

The paper configuration uses

- fixed `lambda = 10` and `mu = 1`; the dimension-dependent CMA-ES population
  rule applies only to the CMA-IH baselines;
- `c(n) = 1 - 1.5/n` (with a valid one-dimensional fallback);
- `eta(n) = (2/n)^(1/3)`;
- `delta0 = sigma0^2 / sqrt(2 sigma0^2 + 1)`, where `sigma0 = 100/n`,
  for the quadratic experiments;
- coordinate-wise double-geometric mutations and best-offspring center and
  sufficient-statistic updates.

## License

MIT. See [LICENSE](LICENSE).
