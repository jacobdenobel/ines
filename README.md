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

## Paper implementation: start here

Readers of the paper or poster should start with
[`src/ines/barebones.py`](src/ines/barebones.py). `BarebonesINES` is the short,
single-parent implementation of the original algorithm, without recombination
or the optional stability extensions. It is also the implementation used by
the reproduction driver.

```python
import numpy as np
from ines.barebones import BarebonesINES

def objective(x):
    return np.sum((x - np.array([3, -2, 7])) ** 2, axis=0)

es = BarebonesINES(
    np.zeros(3, dtype=int),
    delta0=2.0,
    population_size=10,
    seed=1,
)

for _ in range(1_000):
    candidates = es.ask()          # columns are offspring
    values = objective(candidates)
    es.tell(values)
    if values.min() == 0:
        break

print(es.x.ravel(), es.delta.ravel())
```

The file also contains a small runner for the paper's quadratic and
pseudo-Boolean benchmarks and produces objective, L1-distance, and
coordinate-wise step-size plots:

```bash
python -m ines.barebones --algorithm original --function ellipse --dimension 20
python -m ines.barebones --algorithm original --function onemax --dimension 100
```

## Installation

INES requires Python 3.10 or newer.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Paper configuration

- fixed `lambda = 10` and one selected parent;
- `c(n) = 1 - 1.5/n` and `eta(n) = (2/n)^(1/3)`;
- `delta0 = sigma0^2 / sqrt(2 sigma0^2 + 1)`, with `sigma0 = 100/n`,
  on the quadratic benchmarks;
- `delta0 = 1/n` and cyclic `(x+z) mod 2` mapping on OneMax and LeadingOnes;
- the selected raw DG mutation supplies the coordinate-wise sufficient
  statistic.

For the reported evaluation count, all-zero raw mutations are excluded. When
`n < 10`, offspring are sampled sequentially and the loop stops immediately
when an optimum is found instead of completing the population.

## Reproducibility

The complete protocol and output layout are documented in
[`experiments/README.md`](experiments/README.md). Paper INES runs use
`BarebonesINES`; they do not use the extended recombination API.

```bash
python experiments/reproduce.py all --output results
```

Add `--quick` to validate the pipeline with reduced dimensions, repetitions,
and budgets. Full runs use 25 repetitions and can take many hours, especially
for 500-dimensional LeadingOnes.

## Extended optimizer and recombination experiments

[`IntegerNaturalEvolutionStrategy`](src/ines/optimizers/ines.py) is the extended
ask/tell API. It supports multiple parents and alternative center and
sufficient-statistic recombination rules. Those experiments and options are
not part of the paper's INES results.

An IOH problem can initialize its bounds and binary/integer mode automatically:

```python
from ines import IntegerNaturalEvolutionStrategy, make_quadratic_benchmark

problem = make_quadratic_benchmark(dim=20, kind="ellipse", seed=1)
es = IntegerNaturalEvolutionStrategy.from_problem(
    problem,
    seed=1993,
    lambda_=10,
    mu=5,
)
```

The extended command-line benchmark interface is:

```bash
ines benchmark --suite quadratic --kind ellipse --dim 20 --reps 25 \
  --lambda 10 --mu 5
```

The optional `stabilize=True` mode, recombination choices, and multi-parent
updates are extensions. They are deliberately kept out of the barebones paper
implementation and reproduction path.

## Additional path-free variant

`BarebonesNaturalGradientINES` is a separate experimental variant in the same
single file. It removes the evolution path and the `max(Var[|Z|], 1)` floor and
has only the learning rate `eta`:

```bash
python -m ines.barebones --algorithm natural-gradient \
  --function ellipse --dimension 20 --eta 0.1
```

## License

MIT. See [LICENSE](LICENSE).
