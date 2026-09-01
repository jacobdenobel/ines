import matplotlib.pyplot as plt
import numpy as np
from unittest.mock import patch

from ines import IntegerNaturalEvolutionStrategy
from ines.barebones import (
    BarebonesINES,
    BarebonesNaturalGradientINES,
    _evaluate_candidates,
    count_paper_evaluations,
    run_paper_benchmark,
)
from ines.plotting import (
    plot_delta_history,
    plot_l1_distance_history,
    plot_objective_history,
)


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


def test_paper_runners_record_objective_and_delta_histories(tmp_path):
    original_history = run_paper_benchmark(
        algorithm="original",
        kind="ellipse",
        dimension=5,
        budget=100,
        population_size=10,
    )
    path_free_history = run_paper_benchmark(
        algorithm="natural-gradient",
        kind="ellipse",
        dimension=5,
        budget=100,
        population_size=10,
        eta=0.1,
    )

    for name, history in (
        ("original", original_history),
        ("path_free", path_free_history),
    ):
        assert history.evaluations.shape == (10,)
        assert history.function_values.shape == (10,)
        assert history.best_values.shape == (10,)
        assert history.l1_distances.shape == (10,)
        assert history.deltas.shape == (10, 5)
        assert np.all(np.diff(history.best_values) <= 0)
        assert np.all(history.l1_distances >= 0)

        delta_path = tmp_path / f"{name}_delta.png"
        objective_path = tmp_path / f"{name}_objective.png"
        l1_path = tmp_path / f"{name}_l1.png"
        delta_figure, _ = plot_delta_history(history, delta_path)
        objective_figure, objective_axis = plot_objective_history(
            history, objective_path
        )
        l1_figure, l1_axis = plot_l1_distance_history(history, l1_path)
        assert delta_path.stat().st_size > 0
        assert objective_path.stat().st_size > 0
        assert l1_path.stat().st_size > 0
        assert len(delta_figure.axes) == 2
        assert objective_axis.get_yscale() == "symlog"
        assert l1_axis.get_yscale() == "symlog"
        plt.close(delta_figure)
        plt.close(objective_figure)
        plt.close(l1_figure)


def test_paper_evaluation_count_excludes_only_all_zero_mutations():
    steps = np.array([[0, 1, 0, 2], [0, 0, -1, 0]])
    assert count_paper_evaluations(steps) == 3


def test_zero_mutations_are_not_submitted_to_objective():
    candidates = np.array([[4, 5, 4, 6], [7, 7, 7, 7]])
    steps = np.array([[0, 1, 0, 2], [0, 0, 0, 0]])
    submitted = []

    def objective(points):
        submitted.append(np.asarray(points).copy())
        return np.sum(points, axis=1)

    values, evaluated = _evaluate_candidates(
        objective,
        candidates,
        steps,
        parent_value=11.0,
        reuse_zero_steps=True,
    )

    np.testing.assert_array_equal(submitted[0], np.array([[5, 7], [6, 7]]))
    np.testing.assert_array_equal(values, np.array([11.0, 12.0, 11.0, 13.0]))
    assert evaluated == 2


def test_small_dimension_sampling_stops_at_first_optimum():
    # RandomState(1993) initializes x=[1,1], while instance 1 has x*=[0,1].
    optimum_step = np.array([[1], [0]])
    with patch(
        "ines.barebones.cwise_double_geometric", return_value=optimum_step
    ) as sample:
        history = run_paper_benchmark(
            kind="onemax",
            dimension=2,
            budget=20,
            random_state=np.random.RandomState(1993),
        )

    assert sample.call_count == 1
    assert history.evaluations.tolist() == [1]
    assert history.best_values.tolist() == [0.0]
    assert history.final_x.tolist() == [0, 1]
