import unittest
from unittest.mock import patch

import ioh
import numpy as np

from ines import (
    IntegerNaturalEvolutionStrategy,
    make_binary_benchmark,
    make_quadratic_benchmark,
)
from ines.cli import main
from ines.optimizers.recombination import CenterUpdateKind, SufficientStatisticKind
from ines.runner import run_single_es


class TestAskTellINES(unittest.TestCase):
    def test_binary_benchmarks_are_minimization_problems(self):
        for kind in ("onemax", "leadingones"):
            problem = make_binary_benchmark(20, kind, seed=3)
            self.assertEqual(problem(problem.optimum.x), 0.0)
            self.assertGreater(problem(1 - problem.optimum.x), 0.0)
            problem.reset()

    def test_ask_sphere(self):
        problem = make_quadratic_benchmark(2, "sphere")
        es = IntegerNaturalEvolutionStrategy.from_problem(
            problem, seed=10, lambda_=100_000
        )
        X = es.ask()
        sample_std = np.std(es.Z, axis=1)
        sample_mean = np.mean(X, axis=1)
        self.assertTrue(np.isclose(es.m.ravel(), sample_mean, 1).all())
        self.assertTrue(np.isclose(es.std.ravel(), sample_std, 1).all())

    def test_ask_onemax(self):
        problem = ioh.get_problem("OneMax", 1, 20, ioh.ProblemClass.PBO)
        es = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=10)
        self.assertAlmostEqual(es.delta0, 1 / problem.meta_data.n_variables)
        X = es.ask()
        self.assertEqual(X.min(), 0)
        self.assertEqual(X.max(), 1)
        self.assertEqual(es.m.min(), 0)
        self.assertEqual(es.m.max(), 1)

    def test_tell_best_candidate(self):
        problem = make_quadratic_benchmark(2, "sphere")
        es = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=10)

        X = es.ask()
        f = problem(X.T)

        idx = np.argmin(f)
        expected_m = X[:, idx, None].copy()
        expected_z_prime = es.Z[:, idx, None].copy()

        es.tell(X, f)

        np.testing.assert_array_equal(es.m, expected_m)
        np.testing.assert_array_equal(
            expected_z_prime, X[:, idx, None] - (X[:, idx, None] - expected_z_prime)
        )

    def test_tell_updates_path_and_delta_by_formula(self):
        problem = make_quadratic_benchmark(2, "sphere")
        es = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=10)

        X = es.ask()
        Z = es.Z.copy()
        f = problem(X.T)

        idx = np.argmin(f)
        z_prime = Z[:, idx, None].copy()

        old_delta = es.delta.copy()
        old_pi = es.pi.copy()

        dz = np.abs(z_prime) - old_delta
        grad = dz / np.maximum(es.delta_to_abs_variance(old_delta), 1)
        expected_pi = (1 - es.c) * old_pi + es.c_old * grad
        expected_delta = old_delta * np.exp(es.eta * expected_pi)

        es.tell(X, f)

        np.testing.assert_allclose(es.pi, expected_pi)
        np.testing.assert_allclose(es.delta, expected_delta)

    def test_rejects_invalid_hyperparameters(self):
        with self.assertRaisesRegex(ValueError, "lambda_"):
            IntegerNaturalEvolutionStrategy(np.zeros(2, dtype=int), 1.0, lambda_=1)
        with self.assertRaisesRegex(ValueError, "mu"):
            IntegerNaturalEvolutionStrategy(
                np.zeros(2, dtype=int), 1.0, lambda_=4, mu=5
            )
        with self.assertRaisesRegex(ValueError, "delta0"):
            IntegerNaturalEvolutionStrategy(np.zeros(2, dtype=int), 0.0)

    def test_corrected_standard_deviation_conversion(self):
        self.assertAlmostEqual(
            IntegerNaturalEvolutionStrategy.std_to_delta(1.0),
            1.0 / np.sqrt(3.0),
        )

    def test_stabilization_is_opt_in(self):
        problem = make_quadratic_benchmark(2, "sphere")
        reference = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=10)
        stabilized = IntegerNaturalEvolutionStrategy.from_problem(
            problem, seed=10, stabilize=True
        )

        X = reference.ask()
        X_stable = stabilized.ask()
        np.testing.assert_array_equal(X, X_stable)
        f = problem(X.T)

        reference.tell(X, f)
        stabilized.tell(X_stable, f)
        np.testing.assert_allclose(reference.delta, stabilized.delta)

    def test_runner_uses_last_complete_population_in_budget(self):
        problem = make_binary_benchmark(5, "onemax", seed=3)
        run_single_es(
            problem,
            budget=20,
            target=-1.0,
            seed=10,
            lambda_=10,
            mu=1,
            center_update_kind=CenterUpdateKind.BEST,
            sufficient_statistic_kind=SufficientStatisticKind.BEST,
        )
        self.assertEqual(problem.state.evaluations, 20)

    def test_legacy_random_state_is_an_explicit_runner_option(self):
        problem = make_binary_benchmark(5, "leadingones", seed=3)
        random_state = np.random.RandomState(1993)
        state_before = random_state.get_state()[1].copy()
        run_single_es(
            problem,
            budget=10,
            target=-1.0,
            seed=10,
            lambda_=10,
            mu=1,
            center_update_kind=CenterUpdateKind.BEST,
            sufficient_statistic_kind=SufficientStatisticKind.BEST,
            random_state=random_state,
        )
        self.assertFalse(np.array_equal(state_before, random_state.get_state()[1]))

    @patch("ines.cli.run_benchmark")
    @patch(
        "sys.argv",
        [
            "ines",
            "benchmark",
            "--kind",
            "sphere",
            "--dim",
            "5",
            "--lambda",
            "12",
            "--mu",
            "3",
            "--reps",
            "1",
        ],
    )
    def test_cli_preserves_explicit_population_parameters(self, run_benchmark):
        run_benchmark.return_value.algorithm_name = "INES"
        run_benchmark.return_value.problem_id = 1
        run_benchmark.return_value.problem_name = "sphere"
        run_benchmark.return_value.dimension = 5
        run_benchmark.return_value.values = np.array([0.0])
        run_benchmark.return_value.ert = 1.0

        main()

        self.assertEqual(run_benchmark.call_args.kwargs["lambda_"], 12)
        self.assertEqual(run_benchmark.call_args.kwargs["mu"], 3)

    @patch("ines.cli.run_benchmark")
    @patch(
        "sys.argv",
        ["ines", "benchmark", "--kind", "sphere", "--dim", "100", "--reps", "1"],
    )
    def test_cli_defaults_to_paper_population(self, run_benchmark):
        run_benchmark.return_value.algorithm_name = "INES"
        run_benchmark.return_value.problem_id = 1
        run_benchmark.return_value.problem_name = "sphere"
        run_benchmark.return_value.dimension = 100
        run_benchmark.return_value.values = np.array([0.0])
        run_benchmark.return_value.ert = 1.0

        main()

        self.assertEqual(run_benchmark.call_args.kwargs["lambda_"], 10)
        self.assertEqual(run_benchmark.call_args.kwargs["mu"], 1)


if __name__ == "__main__":
    unittest.main()

