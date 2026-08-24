from dataclasses import dataclass
from typing import Optional

import numpy as np
import ioh

from numpy.typing import NDArray

from ..distributions import cwise_double_geometric
from .recombination import CenterUpdateKind, SufficientStatisticKind


@dataclass
class IntegerNaturalEvolutionStrategy:
    x0: NDArray[np.integer]
    delta0: float
    mu: int = None
    lambda_: int = None
    seed: Optional[int] = None
    c: Optional[float] = None
    eta: Optional[float] = None
    is_binary: bool = False

    center_update_kind: CenterUpdateKind = CenterUpdateKind.BEST
    sufficient_statistic_kind: SufficientStatisticKind = SufficientStatisticKind.BEST

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.m = np.asarray(self.x0, dtype=int).copy().reshape(-1, 1)
        self.n = self.m.size

        if self.lambda_ is None:
            self.lambda_ = 10 
        
        if self.mu is None or self.mu > self.lambda_:
            self.mu = int(np.floor(self.lambda_ / 2))

        if self.eta is None:
            self.eta = pow(2, 1 / 3) / pow(self.n, 1 / 3)
        if self.c is None:
            self.c = 1 - (1.5 / self.n)

        self.c_old = np.sqrt(self.c * (2 - self.c))
        self.delta = np.full((self.n, 1), self.delta0)
        self.pi = np.zeros((self.n, 1))
        self.generation = 0

        self.w = np.log(self.mu + 1 / 2) - np.log(np.arange(1, self.mu + 1))
        self.w /= self.w.sum()

        # w2 = np.maximum(0, np.log(self.lambda_ / 2 + 1) - np.log(np.arange(1, self.lambda_ + 1)))
        # w2 /= w2.sum()
        # self.u = w2 - (1 / self.lambda_)
        # self.eta = (3 + np.log(self.n)) / (5 * np.sqrt(self.n))

        self.center_update = self.center_update_kind.make()
        self.sufficient_statistic = self.sufficient_statistic_kind.make()
        
        self.delta_min =  1.0 / self.n

    @property
    def q(self):
        return self.delta / (np.sqrt(1 + self.delta**2) + 1)

    @property
    def p_effective(self):
        return 1 - self.q

    @property
    def std(self):
        return self.delta_to_std(self.delta)

    @property
    def var(self):
        return self.std**2

    @property
    def expected_absolute_step(self):
        return self.delta

    @property
    def absolute_step_variance(self):
        return IntegerNaturalEvolutionStrategy.delta_to_abs_variance(self.delta)

    def ask(self) -> NDArray[np.integer]:
        self.Z = cwise_double_geometric(self.rng, self.p_effective, self.lambda_)
        X = self.m + self.Z
        if not self.is_binary:
            return X
        return X & 1

    def tell(self, X: NDArray[np.integer], f: NDArray[np.floating]) -> None:
        """Assumes the order of X,y is consistent with ask"""

        self.generation += 1
        f = np.asarray(f)
        idx = np.argsort(f)

        self.m += self.center_update(self.Z, idx, self.w, self.rng)
        statistic = self.sufficient_statistic(self.Z, idx, self.w, self.rng)

        # Fisher-normalized coordinate-wise signal
        dz = statistic - self.expected_absolute_step
        grad = dz / np.maximum(self.absolute_step_variance, 1)

        self.pi = (1 - self.c) * self.pi + (self.c_old * grad)
        self.delta *= np.exp(self.eta * self.pi)

        # # 1. Center update
        # self.m += self.center_update(self.Z, idx, self.w, self.rng)

        # # 2. Sufficient statistic of selected/recombined steps
        # statistic = self.sufficient_statistic(self.Z, idx, self.w, self.rng)

        # # 3. Fisher-normalized coordinate-wise signal
        # dz = statistic - self.expected_absolute_step
        # grad = dz /  np.maximum(self.absolute_step_variance, 1)

        # # 4. If a coordinate was never sampled nonzero in the whole population,
        # #    there is no selection information for that coordinate.
        # inactive = ~np.any(self.Z != 0, axis=1)
        # grad[inactive] = 0.0

        # # 5. Evolution path update
        # self.pi = (1.0 - self.c) * self.pi + self.c_old * grad

        # # 6. Multiplicative natural-gradient step
        # delta_new = self.delta * np.exp(self.eta * self.pi)

        # # 7. Projection to non-degenerate DG family
        # hit_floor = delta_new < self.delta_min
        # self.delta = np.maximum(delta_new, self.delta_min)

        # # 8. Anti-windup: do not keep accumulating negative path while clipped
        # self.pi[hit_floor & (self.pi < 0.0)] = 0.0



    @staticmethod
    def std_to_delta(sigma: float) -> float:
        """Convert desired Std[Z] to delta = E[|Z|]."""
        variance = sigma**2
        return variance / np.sqrt(2 * variance + 1)

    @staticmethod
    def delta_to_variance(delta: float) -> float:
        """Return Var[Z] from delta = E[|Z|]."""
        return delta**2 + delta * np.sqrt(delta**2 + 1)

    @staticmethod
    def delta_to_std(delta: float) -> float:
        """Return Std[Z] from delta = E[|Z|]."""
        variance = IntegerNaturalEvolutionStrategy.delta_to_variance(delta)
        return np.sqrt(variance)

    @staticmethod
    def delta_to_abs_variance(delta: float) -> float:
        """Return Var[|Z|], the Fisher information for theta = log(q)."""
        return delta * np.sqrt(delta**2 + 1)

    @staticmethod
    def from_problem(
        problem: ioh.ProblemType, **kwargs
    ) -> "IntegerNaturalEvolutionStrategy":
        n = problem.meta_data.n_variables

        lb = np.asarray(problem.bounds.lb, dtype=int)
        ub = np.asarray(problem.bounds.ub, dtype=int)
        spans = ub - lb

        if not np.all(spans == spans[0]):
            raise ValueError(
                "Only equal-width box bounds are supported for automatic initialization."
            )

        db = int(spans[0])
        is_binary = db == 1

        initial_scale = db / n

        if not is_binary:
            delta0 = IntegerNaturalEvolutionStrategy.std_to_delta(initial_scale) 
        else:
            delta0 = initial_scale

        rng = np.random.default_rng(kwargs.get("seed"))

        if "x0" not in kwargs:
            kwargs['x0'] = rng.integers(lb, ub + 1)

        if "delta0" not in kwargs:
            kwargs['delta0'] = delta0

        return IntegerNaturalEvolutionStrategy(
            is_binary=is_binary,
            **kwargs,
        )
