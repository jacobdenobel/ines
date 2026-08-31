import numpy as np

from examples.barebones_ines import BarebonesINES
from examples.barebones_natural_gradient import BarebonesNaturalGradientINES
from ines import IntegerNaturalEvolutionStrategy


def test_barebones_original_matches_packaged_reference_update():
    x0 = np.array([0, 0, 0], dtype=int)
    kwargs = dict(delta0=2.0, c=0.5, eta=0.2, seed=7)
    packaged = IntegerNaturalEvolutionStrategy(x0=x0, lambda_=10, **kwargs)
    standalone = BarebonesINES(x0, population_size=10, **kwargs)

    packaged_points = packaged.ask()
    standalone_points = standalone.ask()
    np.testing.assert_array_equal(packaged_points, standalone_points)

    target = np.array([3, -2, 7])[:, None]
    values = np.sum((packaged_points - target) ** 2, axis=0)
    packaged.tell(packaged_points, values)
    standalone.tell(values)

    np.testing.assert_array_equal(packaged.m, standalone.x)
    np.testing.assert_allclose(packaged.pi, standalone.path)
    np.testing.assert_allclose(packaged.delta, standalone.delta)


def test_path_free_variant_is_one_unclipped_natural_gradient_step():
    optimizer = BarebonesNaturalGradientINES(
        np.zeros(2, dtype=int), delta0=0.25, eta=0.1, seed=3
    )
    points = optimizer.ask()
    values = np.sum(points**2, axis=0)
    best = int(np.argmin(values))
    selected = optimizer.steps[:, best, None]
    old_delta = optimizer.delta.copy()
    fisher = old_delta * np.sqrt(1.0 + old_delta**2)
    expected = old_delta * np.exp(
        optimizer.eta * (np.abs(selected) - old_delta) / fisher
    )

    optimizer.tell(values)

    np.testing.assert_allclose(optimizer.delta, expected)
    assert not hasattr(optimizer, "path")
