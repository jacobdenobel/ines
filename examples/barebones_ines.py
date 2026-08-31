"""A single-file, NumPy-only implementation of the original INES algorithm.

The class intentionally mirrors the reference implementation in integer-es:
best-offspring selection, a Fisher-normalized coordinate-wise signal, an
evolution path, and a multiplicative update of E[|Z_i|].
"""

from __future__ import annotations

import numpy as np


def sample_double_geometric(
    rng: np.random.Generator, delta: np.ndarray, population_size: int
) -> np.ndarray:
    """Sample coordinate-wise DG mutations as differences of geometrics."""
    q = delta / (np.sqrt(1.0 + delta**2) + 1.0)
    p = np.clip(1.0 - q, 1e-12, 1.0 - 1e-12)
    log_q = np.log1p(-p)
    shape = (len(delta), population_size)
    g1 = np.floor(np.log1p(-rng.random(shape)) / log_q).astype(int)
    g2 = np.floor(np.log1p(-rng.random(shape)) / log_q).astype(int)
    return g1 - g2


class BarebonesINES:
    """Original coordinate-wise `(1, lambda)` INES in ask/tell form."""

    def __init__(
        self,
        x0: np.ndarray,
        delta0: float,
        population_size: int = 10,
        c: float | None = None,
        eta: float | None = None,
        seed: int | None = None,
        binary: bool = False,
    ) -> None:
        self.rng = np.random.default_rng(seed)
        self.x = np.asarray(x0, dtype=int).reshape(-1, 1).copy()
        self.n = len(self.x)
        self.population_size = population_size
        self.c = 1.0 - 1.5 / self.n if c is None else c
        self.eta = (2.0 / self.n) ** (1.0 / 3.0) if eta is None else eta
        self.path_scale = np.sqrt(self.c * (2.0 - self.c))
        self.delta = np.full((self.n, 1), delta0, dtype=float)
        self.path = np.zeros((self.n, 1))
        self.binary = binary

    def ask(self) -> np.ndarray:
        self.steps = sample_double_geometric(self.rng, self.delta, self.population_size)
        candidates = self.x + self.steps
        return candidates & 1 if self.binary else candidates

    def tell(self, values: np.ndarray) -> None:
        """Update from objective values in the same column order as `ask()`."""
        best = int(np.argmin(values))
        selected_step = self.steps[:, best, None]
        self.x += selected_step

        variance_abs_step = self.delta * np.sqrt(1.0 + self.delta**2)
        gradient = (np.abs(selected_step) - self.delta) / np.maximum(
            variance_abs_step, 1.0
        )
        self.path = (1.0 - self.c) * self.path + self.path_scale * gradient
        self.delta *= np.exp(self.eta * self.path)


if __name__ == "__main__":
    target = np.array([3, -2, 7])[:, None]
    optimizer = BarebonesINES(np.zeros(3, dtype=int), delta0=2.0, seed=1)
    for _ in range(1_000):
        points = optimizer.ask()
        fitness = np.sum((points - target) ** 2, axis=0)
        optimizer.tell(fitness)
        if fitness.min() == 0:
            break
    print("x =", optimizer.x.ravel())
    print("delta =", optimizer.delta.ravel())
