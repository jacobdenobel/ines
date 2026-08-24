import unittest

import ioh
import numpy as np

from ines import (
    IntegerNaturalEvolutionStrategy, 
    make_quadratic_benchmark, 
)


class TestAskTellINES(unittest.TestCase):
    def test_ask_sphere(self):
        problem = make_quadratic_benchmark(2, "sphere")
        es = IntegerNaturalEvolutionStrategy.from_problem(problem, seed=10, lambda_=100_000)
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
        np.testing.assert_array_equal(expected_z_prime, X[:, idx, None] - (X[:, idx, None] - expected_z_prime))

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
        

if __name__ == "__main__":
    unittest.main()