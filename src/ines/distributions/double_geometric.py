import numpy as np


def double_geometric(rng: np.random.Generator, p: np.ndarray, n: int, zero_mutations: bool = True):
    eps = 1e-10
    p = np.clip(p, eps, 1 - eps)
    log_p = np.log(1 - p)

    z = np.zeros(n)
    while np.sum(np.abs(z)) <= 0.0:
        g1 = np.floor(np.log(1 - rng.random(n)) / log_p).astype(int)
        g2 = np.floor(np.log(1 - rng.random(n)) / log_p).astype(int)
        z = g1 - g2
        # Why are zero mutations good?
        if zero_mutations:
            break
    return z


def cwise_double_geometric(rng: np.random.Generator, p: np.ndarray, n_samples: int,) -> np.ndarray:
    """
    Coordinate-wise double geometric (discrete Laplace) sampler via difference
    of two geometric RVs with parameter p (support {0,1,2,...}).

    Returns: Z with shape (n, n_samples)
    """
    eps = 1e-12
    p = np.asarray(p, dtype=float).reshape(-1, 1)  # (n,1) always
    p = np.clip(p, eps, 1 - eps)

    log1mp = np.log1p(-p)  # (n,1), stable
    n = p.shape[0]

    u1 = rng.random((n, n_samples))  # [0,1)
    u2 = rng.random((n, n_samples))

    g1 = np.floor(np.log1p(-u1) / log1mp).astype(int)
    g2 = np.floor(np.log1p(-u2) / log1mp).astype(int)

    return g1 - g2


