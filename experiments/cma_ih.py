"""CMA-IH baselines extracted from integer-es for paper reproduction."""

from __future__ import annotations

import cma
import numpy as np


def run_cma_ih(problem, budget: int, target: float, seed: int, separable: bool):
    """Run the integer-handling pycma baseline used in integer-es."""
    n = problem.meta_data.n_variables
    rng = np.random.default_rng(seed)
    x0 = rng.integers(problem.bounds.lb, problem.bounds.ub + 1)
    sigma0 = float(problem.bounds.ub[0] - problem.bounds.lb[0]) / n
    options = {
        "integer_variables": list(range(n)),
        "tolfun": 0,
        "tolfunhist": 0,
        "tolflatfitness": 60,
        "verbose": -9,
        "bounds": [problem.bounds.lb, problem.bounds.ub],
        "CMA_diagonal": separable,
        "seed": seed,
    }
    constructor = getattr(cma, "CMA", cma.CMAEvolutionStrategy)
    optimizer = constructor(x0, sigma0, options)

    while problem.state.evaluations + optimizer.popsize <= budget:
        if problem.state.current_best.y <= target:
            break
        asked = optimizer.ask()
        points = np.asarray(asked).astype(int)
        values = np.asarray(problem(points), dtype=float)
        optimizer.tell(asked, values)

    return float(problem.state.current_best.y), int(problem.state.evaluations)
