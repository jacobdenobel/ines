import enum
from typing import Callable
from functools import wraps

import numpy as np

from numpy.typing import NDArray


class StrEnum(str, enum.Enum):
    """Python 3.10-compatible subset of enum.StrEnum."""

    def __str__(self) -> str:
        return self.value

RecombinerType = Callable[
    [NDArray[np.integer], NDArray[np.integer], NDArray[np.floating]],
    NDArray[np.integer],
]


def select_best(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    return Z[:, idx[0], None].copy()


def abs_select_best(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    return np.abs(select_best(Z, idx, w, rng))


def selected(
    Z: NDArray[np.integer], idx: NDArray[np.integer], w: NDArray[np.floating]
) -> NDArray[np.integer]:
    mu = len(w)
    return Z[:, idx[:mu]]


def weigthed_abs_avg(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    abs_Z = np.abs(selected(Z, idx, w))
    a_bar = (abs_Z @ w).reshape(-1, 1)
    return a_bar


def abs_avg(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    abs_Z = np.abs(selected(Z, idx, w))
    a_bar_u = np.mean(abs_Z, axis=1).reshape(-1, 1)
    return a_bar_u


def weigthed_avg(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    Zsel = selected(Z, idx, w)
    z_bar = (Zsel @ w).reshape(-1, 1)
    return z_bar


def round_avg(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    z_bar = weigthed_avg(Z, idx, w, rng)
    return np.round(z_bar).astype(int)


def sround_avg(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    z_bar = weigthed_avg(Z, idx, w, rng)
    mask = rng.integers(0, 2, size=z_bar.size).astype(bool)
    z_bar[mask] = np.ceil(z_bar[mask])
    z_bar[~mask] = np.floor(z_bar[~mask])
    return z_bar.astype(int)


def uniform_discrete(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
):
    Zs = selected(Z, idx, w)
    n, mu = Zs.shape
    U = rng.choice(mu, size=n)
    u = Zs[np.arange(n), U].reshape(-1, 1)
    return u


def weighted_discrete(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
):
    Zs = selected(Z, idx, w)
    n, mu = Zs.shape
    A = rng.choice(mu, size=n, p=w.ravel())
    d = Zs[np.arange(n), A].reshape(-1, 1)
    return d


def weighted_median(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    Zs = selected(Z, idx, w)

    order = np.argsort(Zs, axis=1, kind="stable")
    vals = np.take_along_axis(Zs, order, axis=1)

    # Broadcast weights according to the per-coordinate sorting order.
    ws = w[order]
    cumsum = np.cumsum(ws, axis=1)
    cumbefore = cumsum - ws

    # Lower end of the weighted-median interval:
    # smallest value with cumulative weight >= 1/2.
    lo_idx = np.argmax(cumsum >= 0.5, axis=1)
    lo = vals[np.arange(Z.shape[0]), lo_idx]

    # Upper end of the weighted-median interval:
    # largest value whose strict-left weight is <= 1/2.
    valid_hi = cumbefore <= 0.5
    hi_idx = valid_hi.shape[1] - 1 - np.argmax(valid_hi[:, ::-1], axis=1)
    hi = vals[np.arange(Z.shape[0]), hi_idx]

    med = np.where(
        hi < 0,
        hi,
        np.where(lo > 0, lo, 0),
    ).astype(
        Z.dtype
    )[:, None]

    return med


def weighted_sign(
    Z: NDArray[np.integer],
    idx: NDArray[np.integer],
    w: NDArray[np.floating],
    rng: np.random.Generator,
) -> NDArray[np.integer]:
    """
    Conservative one-step DG-native location update.

    For each coordinate i, move +1 only if more than half of the selected
    rank weight lies at positive mutation steps, move -1 only if more than
    half lies at negative mutation steps, and otherwise stay at 0.

    This is the one-lattice-step move toward the weighted median.
    """
    Zs = selected(Z, idx, w)

    w_pos = (Zs > 0) @ w
    w_neg = (Zs < 0) @ w

    step = np.zeros((Z.shape[0], 1), dtype=Z.dtype)

    step[w_pos > 0.5, 0] = 1
    step[w_neg > 0.5, 0] = -1

    return step


class CenterUpdateKind(StrEnum):
    BEST = enum.auto()
    ROUND = enum.auto()
    SROUND = enum.auto()
    DISCRETE = enum.auto()
    WEIGHTED_DISCRETE = enum.auto()
    WEIGHTED_SIGN = enum.auto()
    WEIGHTED_MEDIAN = enum.auto()

    def make(self) -> RecombinerType:
        return {
            CenterUpdateKind.BEST: select_best,
            CenterUpdateKind.ROUND: round_avg,
            CenterUpdateKind.SROUND: sround_avg,
            CenterUpdateKind.DISCRETE: uniform_discrete,
            CenterUpdateKind.WEIGHTED_DISCRETE: weighted_discrete,
            CenterUpdateKind.WEIGHTED_SIGN: weighted_sign,
            CenterUpdateKind.WEIGHTED_MEDIAN: weighted_median,
        }[self]


class SufficientStatisticKind(StrEnum):
    BEST = enum.auto()
    WEIGHTED = enum.auto()
    UNIFORM = enum.auto()

    def make(self) -> RecombinerType:
        return {
            SufficientStatisticKind.BEST: abs_select_best,
            SufficientStatisticKind.WEIGHTED: weigthed_abs_avg,
            SufficientStatisticKind.UNIFORM: abs_avg,
        }[self]

