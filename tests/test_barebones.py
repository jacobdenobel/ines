import matplotlib.pyplot as plt
import numpy as np

from examples import barebones_ines as original
from examples import barebones_natural_gradient as path_free
from ines import IntegerNaturalEvolutionStrategy, make_quadratic_benchmark

BarebonesINES = original.BarebonesINES
BarebonesNaturalGradientINES = path_free.BarebonesNaturalGradientINES


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


def test_barebones_objectives_match_packaged_paper_benchmarks():
    dimension = 5
    instance = 7
    points = np.arange(-10, 15).reshape(dimension, -1)

    for kind in original.PAPER_FUNCTIONS:
        objective, optimum = original.make_paper_objective(kind, dimension, instance)
        path_free_objective, path_free_optimum = path_free.make_paper_objective(
            kind, dimension, instance
        )
        problem = make_quadratic_benchmark(dimension, kind, seed=instance)

        np.testing.assert_array_equal(optimum, problem.optimum.x)
        np.testing.assert_array_equal(path_free_optimum, problem.optimum.x)
        expected = np.asarray(problem(points.T), dtype=float)
        np.testing.assert_allclose(objective(points), expected)
        np.testing.assert_allclose(path_free_objective(points), expected)
        problem.reset()


def test_paper_runners_record_objective_and_delta_histories(tmp_path):
    original_history = original.run_paper_benchmark(
        kind="ellipse", dimension=5, budget=100, population_size=10
    )
    path_free_history = path_free.run_paper_benchmark(
        kind="ellipse", dimension=5, budget=100, population_size=10, eta=0.1
    )

    for name, module, history in (
        ("original", original, original_history),
        ("path_free", path_free, path_free_history),
    ):
        assert history.evaluations.shape == (10,)
        assert history.best_values.shape == (10,)
        assert history.deltas.shape == (10, 5)
        assert np.all(np.diff(history.best_values) <= 0)

        delta_path = tmp_path / f"{name}_delta.png"
        objective_path = tmp_path / f"{name}_objective.png"
        delta_figure, _ = module.plot_delta_history(history, delta_path)
        objective_figure, _ = module.plot_objective_history(history, objective_path)
        assert delta_path.stat().st_size > 0
        assert objective_path.stat().st_size > 0
        plt.close(delta_figure)
        plt.close(objective_figure)

