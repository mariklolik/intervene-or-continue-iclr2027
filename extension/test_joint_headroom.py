import importlib
import importlib.util
import itertools
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


class JointHeadroomTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("joint_headroom"))
        return importlib.import_module("joint_headroom")

    def moments(self, q, mass):
        q, mass = np.asarray(q), np.asarray(mass)
        d = q[:, 1:] - q[:, :1]
        second = np.array([mass @ ((q[:, b]-q[:, a])**2) for a, b in itertools.combinations(range(4), 2)])
        return mass @ d, second, mass @ np.maximum(d.max(axis=1), 0)

    def test_identical_effects_remove_redundant_arms(self):
        module = self.module()
        mu, second, value = self.moments([[.5, 0, 0, 0], [.5, 1, 1, 1]], [.5, .5])
        result = module.population_bounds(mu, second)
        self.assertAlmostEqual(result["H_interval"][1], .25)
        self.assertAlmostEqual(value, .25)
        self.assertAlmostEqual(result["upper_components"]["control_star"], .75)

    def test_shared_control_opposing_effects_are_point_identified(self):
        module = self.module()
        mu, second, value = self.moments([[.5, 1, 0, .5], [.5, 0, 1, .5]], [.5, .5])
        result = module.population_bounds(mu, second)
        np.testing.assert_allclose(result["H_interval"], [.5, .5], atol=1e-14)
        self.assertEqual(value, .5)

    def test_centered_bound_has_exact_four_atom_witness(self):
        module = self.module()
        q = [[.5, 0, 0, 0], [0, .5, 0, 0], [0, 0, .5, 0], [0, 0, 0, .5]]
        mu, second, value = self.moments(q, [.25]*4)
        result = module.population_bounds(mu, second)
        self.assertAlmostEqual(value, .375)
        self.assertAlmostEqual(result["H_interval"][1], value)
        self.assertGreater(result["upper_components"]["tree"], value)

    def test_directed_tree_orientation_handles_asymmetric_means(self):
        module = self.module()
        self.assertEqual(len(module.rooted_trees()), 16)
        for q, expected in [([.1, .7, .4, .9], .8), ([1, 0, .5, .25], 0)]:
            mu, second, value = self.moments([q], [1])
            result = module.population_bounds(mu, second)
            np.testing.assert_allclose(result["H_interval"], [expected, expected], atol=1e-12)
            self.assertAlmostEqual(value, expected)

    def test_finite_grid_linear_programs_certify_witness_bounds(self):
        module = self.module()
        q = np.array(list(itertools.product([0, .5, 1], repeat=4)))
        d = q[:, 1:] - q[:, :1]
        s = np.array([(q[:, b]-q[:, a])**2 for a, b in itertools.combinations(range(4), 2)])
        constraints = np.vstack([np.ones(len(q)), d.T, s])
        target = np.maximum(0, d.max(axis=1))
        cases = [([[.5, 0, 0, 0], [.5, 1, 1, 1]], [.5, .5], "upper"),
                 ([[.5, 1, 0, .5], [.5, 0, 1, .5]], [.5, .5], "both"),
                 ([[.5, 0, 0, 0], [0, .5, 0, 0], [0, 0, .5, 0], [0, 0, 0, .5]], [.25]*4, "upper")]
        for points, mass, exact in cases:
            mu, second, _ = self.moments(points, mass)
            bound = module.population_bounds(mu, second)["H_interval"]
            for sign, index in [(1, 0), (-1, 1)]:
                fit = linprog(sign*target, A_eq=constraints, b_eq=np.r_[1, mu, second], bounds=(0, None), method="highs")
                self.assertTrue(fit.success)
                value = sign*fit.fun
                self.assertLessEqual(bound[0], value+1e-9)
                self.assertGreaterEqual(bound[1], value-1e-9)
                if index == 1 or exact == "both":
                    self.assertAlmostEqual(bound[index], value)

    def test_random_feasible_distributions_and_rectangles_cover_truth(self):
        module = self.module()
        rng = np.random.default_rng(713)
        for _ in range(80):
            q, mass = rng.random((15, 4)), rng.dirichlet(np.ones(15))
            mu, second, value = self.moments(q, mass)
            point = module.population_bounds(mu, second)
            self.assertLessEqual(point["H_interval"][0], value+1e-12)
            self.assertGreaterEqual(point["H_interval"][1], value-1e-12)
            old_lower = max(0, mu.max(), ((second[:3]+mu)/2).max())
            old_upper = min(1, ((np.sqrt(second[:3])+mu)/2).sum(), np.sqrt(second[:3].sum()))
            self.assertGreaterEqual(point["H_interval"][0], old_lower-1e-12)
            self.assertLessEqual(point["H_interval"][1], old_upper+1e-12)
            width = rng.random(9)*.2
            rectangle = np.column_stack([np.r_[mu, second]-width, np.r_[mu, second]+width])
            interval = module.propagate_rectangle(rectangle[:3], rectangle[3:])["H_interval"]
            self.assertLessEqual(interval[0], value+1e-12)
            self.assertGreaterEqual(interval[1], value-1e-12)

    def test_full_q_moments_do_not_determine_headroom(self):
        module = self.module()
        first = np.array([[.5, .25, .5, .5], [.5, .75, .5, .5]])
        second = np.array([[.5, 0, .5, .5], [.5, .5, .5, .5], [.5, 1, .5, .5]])
        weights = [np.array([.5, .5]), np.array([.125, .75, .125])]
        np.testing.assert_allclose(weights[0] @ first, weights[1] @ second)
        np.testing.assert_allclose(np.einsum('n,ni,nj->ij', weights[0], first, first), np.einsum('n,ni,nj->ij', weights[1], second, second))
        results = [self.moments(q, p) for q, p in zip([first, second], weights)]
        self.assertEqual(results[0][2], .125)
        self.assertEqual(results[1][2], .0625)
        np.testing.assert_allclose(module.population_bounds(*results[0][:2])["H_interval"], module.population_bounds(*results[1][:2])["H_interval"])
        tables = np.array(list(itertools.product([0, 1], repeat=8))).reshape(-1, 2, 4)
        laws = [np.array([p @ np.prod(np.where(table[None, :, :], q[:, None, :], 1-q[:, None, :]), axis=(1, 2)) for table in tables]) for q, p in zip([first, second], weights)]
        np.testing.assert_allclose(laws[0], laws[1])

    def test_cross_arm_moments_use_different_rounds_and_preserve_negatives(self):
        module = self.module()
        y = np.array([[[0, 1, 0, 1], [1, 0, 1, 0]]])
        result = module.analyze(y)
        np.testing.assert_array_equal(result["raw_task_moments"], [[0, 0, 0, -1, 0, -1, -1, 0, -1]])
        self.assertIsNone(result["descriptive"]["H_interval"])
        self.assertIsNotNone(result["confidence"]["H_interval"])

    def test_nine_coordinate_radius_uses_tasks_and_no_tree_penalty(self):
        module = self.module()
        result = module.analyze(np.zeros((134, 2, 4), int))
        self.assertEqual(result["rectangle"]["family_size"], 9)
        self.assertAlmostEqual(result["rectangle"]["radius"], np.sqrt(2*np.log(360)/134))
        self.assertEqual(result["n_tasks"], 134)
        self.assertEqual(result["confidence"]["coverage_at_least"], .95)
        self.assertLess(result["confidence"]["H_interval"][1], .81)

    def test_observable_envelope_respects_two_ranges_and_tail_allocation(self):
        module = self.module()
        y = np.tile([[[0, 1, 0, 0], [1, 0, 0, 0]]], (100, 1, 1))
        result = module.analyze(y)
        self.assertIn("observable", result)
        direct = result["observable"]
        radius = np.sqrt(np.log(40)/200)
        self.assertEqual(direct["V_mean"], -.5)
        self.assertEqual(direct["O_mean"], .5)
        self.assertAlmostEqual(direct["V_lower"], -.5-2*radius)
        self.assertAlmostEqual(direct["O_upper"], .5+radius)
        np.testing.assert_allclose(direct["H_interval"], [0, .5+radius])

    def test_combined_interval_uses_stricter_component_alphas(self):
        module = self.module()
        result = module.analyze(np.zeros((134, 2, 4), int))
        self.assertIn("combined", result)
        combined = result["combined"]
        self.assertEqual(combined["moment_alpha"], .025)
        self.assertEqual(combined["observable_alpha"], .025)
        self.assertEqual(combined["coverage_at_least"], .95)
        self.assertAlmostEqual(combined["moment_rectangle"]["radius"], np.sqrt(2*np.log(720)/134))
        self.assertAlmostEqual(combined["H_interval"][1], np.sqrt(np.log(80)/268))
        self.assertGreater(combined["H_interval"][1], result["observable"]["H_interval"][1])
        self.assertLess(combined["H_interval"][1], result["confidence"]["H_interval"][1])

    def test_incompatible_component_has_no_repaired_combined_interval(self):
        module = self.module()
        y = np.tile([[[0, 1, 0, 0], [1, 0, 0, 0]]], (10000, 1, 1))
        result = module.analyze(y)
        self.assertIn("observable", result)
        self.assertIsNotNone(result["observable"]["H_interval"])
        self.assertIsNone(result["combined"]["H_interval"])
        self.assertEqual(result["combined"]["status"], "incompatible_component")
        empty = module.analyze(np.empty((0, 2, 4)))
        self.assertIsNone(empty["observable"]["H_interval"])
        self.assertIsNone(empty["combined"]["H_interval"])

    def test_invalid_inputs_and_empty_pair_regions_fail_explicitly(self):
        module = self.module()
        for mu, s in [([0]*3, [-.1]*6), ([1, -1, 0], [1]*6)]:
            self.assertIsNone(module.population_bounds(mu, s)["H_interval"])
        with self.assertRaises(ValueError):
            module.population_bounds([0]*3, [0]*5)
        with self.assertRaises(ValueError):
            module.analyze(np.zeros((2, 2, 4)), task_ids=["x", "x"])
        with self.assertRaises(ValueError):
            module.analyze(np.ones((2, 2, 4))*2)
        empty = module.analyze(np.empty((0, 2, 4)))
        self.assertEqual(empty["status"], "no_data")
        self.assertIsNone(empty["confidence"]["H_interval"])

    def test_cli_keeps_strata_separate_and_rejects_bad_schema_or_overwrite(self):
        module = self.module()
        with tempfile.TemporaryDirectory() as directory:
            path, out = Path(directory)/"rows.json", Path(directory)/"out.json"
            row = {"task_id":"t", "model":"m", "env":"alfworld", "split":"dev", "config_sha256":"c", "scaffold":"s", "Y":[[0]*4]*2}
            path.write_text(json.dumps([row, {**row, "config_sha256":"other"}]))
            command = [sys.executable, module.__file__, "--rows", str(path), "--out", str(out)]
            self.assertEqual(subprocess.run(command, capture_output=True).returncode, 0)
            result = json.loads(out.read_text())
            self.assertEqual(len(result["strata"]), 2)
            self.assertTrue(all(s["n_tasks"] == 1 for s in result["strata"]))
            content = path.read_bytes()
            self.assertNotEqual(subprocess.run(command[:-1]+[str(path)], capture_output=True).returncode, 0)
            self.assertEqual(path.read_bytes(), content)
            for invalid in [{}, [{}], [{**row, "Y":[[0, 0]]}]]:
                path.write_text(json.dumps(invalid))
                self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
