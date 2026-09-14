import hashlib
import importlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.optimize import linprog


class HeadroomTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("headroom"))
        return importlib.import_module("headroom")

    def test_sharp_witnesses_for_feasible_scalar_moments(self):
        h = self.module()
        for mu in np.linspace(-1, 1, 17):
            for fraction in np.linspace(0, 1, 11):
                second = mu * mu + fraction * (1 - mu * mu)
                bounds = h.population_bounds(mu, second)
                if second == 0:
                    witnesses = [(np.array([0]), np.array([1]))] * 2
                else:
                    if second >= abs(mu):
                        lower = (np.array([-1, 0, 1]), np.array([(second-mu)/2, 1-second, (second+mu)/2]))
                    else:
                        lower = (np.array([0, second/mu]), np.array([1-mu*mu/second, mu*mu/second]))
                    upper = (np.array([-np.sqrt(second), np.sqrt(second)]), np.array([(1-mu/np.sqrt(second))/2, (1+mu/np.sqrt(second))/2]))
                    witnesses = [lower, upper]
                self.assertTrue(bounds["feasible"])
                for (support, mass), endpoint in zip(witnesses, ["lower", "upper"]):
                    self.assertTrue(np.all(np.abs(support) <= 1 + 1e-12))
                    self.assertTrue(np.all(mass >= -1e-12))
                    self.assertAlmostEqual(mass.sum(), 1)
                    self.assertAlmostEqual(mass @ support, mu)
                    self.assertAlmostEqual(mass @ (support**2), second)
                    self.assertAlmostEqual(mass @ np.maximum(support, 0), bounds[endpoint])

    def test_independent_grid_linear_program_matches_sharp_bounds(self):
        h = self.module()
        support = np.linspace(-1, 1, 17)
        equality = np.array([np.ones(len(support)), support, support**2])
        for mu, second in [(0, 0), (.25, .25), (-.25, .25), (.5, .25), (-.5, .25), (.75, 1), (0, .0625), (0, 1)]:
            limits = h.population_bounds(mu, second)
            for sign, endpoint in [(1, "lower"), (-1, "upper")]:
                result = linprog(sign*np.maximum(support, 0), A_eq=equality, b_eq=[1, mu, second], bounds=(0, None), method="highs")
                self.assertTrue(result.success)
                self.assertAlmostEqual(sign*result.fun, limits[endpoint], places=8)

    def test_infeasible_observed_moments_are_never_repaired(self):
        h = self.module()
        for mu, second in [(0, -.1), (.5, .1), (1.1, 1), (0, 1.01)]:
            bounds = h.population_bounds(mu, second)
            self.assertFalse(bounds["feasible"])
            self.assertEqual(bounds["raw_mu"], mu)
            self.assertEqual(bounds["raw_S"], second)
            self.assertIsNone(bounds["lower"])
            self.assertIsNone(bounds["upper"])

    def test_exact_region_extrema_match_dense_feasible_grid(self):
        h = self.module()
        for mu_limits, s_limits in [([-.2, .5], [.6, .9]), ([-.8, -.3], [.1, .8]), ([.3, .8], [0, .6]), ([-2, 2], [-1, 2]), ([0, 0], [0, 0])]:
            result = h.optimize_region(mu_limits, s_limits)
            self.assertTrue(result["feasible"])
            mu = np.linspace(*result["mu_range"], 701)
            second = np.linspace(*result["S_range"], 701)
            m, s = np.meshgrid(mu, second)
            feasible = s >= m*m
            lower = np.maximum.reduce([np.zeros_like(m), m, (s+m)/2])
            upper = (np.sqrt(s)+m)/2
            self.assertLessEqual(result["lower"], lower[feasible].min()+1e-10)
            self.assertGreaterEqual(result["upper"], upper[feasible].max()-1e-10)
            self.assertLess(lower[feasible].min()-result["lower"], .004)
            self.assertLess(result["upper"]-upper[feasible].max(), .004)
            for key, endpoint in [("lower_witness", "lower"), ("upper_witness", "upper")]:
                witness_mu, witness_s = result[key]
                self.assertLessEqual(witness_mu*witness_mu, witness_s+1e-12)
                self.assertAlmostEqual(h.population_bounds(witness_mu, witness_s)[endpoint], result[endpoint])

    def test_empty_regions_are_flagged(self):
        h = self.module()
        for mu_limits, s_limits in [([.9, 1], [0, .1]), ([-1, 1], [-.5, -.1]), ([2, 3], [0, 1]), ([0, 1], [1.1, 2])]:
            self.assertFalse(h.optimize_region(mu_limits, s_limits)["feasible"])
        result = h.propagate_rectangle([[.9, 1]]*3, [[0, .1]]*3)
        self.assertEqual(result["status"], "incompatible_confidence_region")
        self.assertIsNone(result["H_interval"])

    def test_six_moment_hoeffding_radius_and_task_unit(self):
        h = self.module()
        values = np.zeros((100, 6))
        rectangle = h.simultaneous_rectangle(values)
        radius = np.sqrt(2*np.log(240)/100)
        self.assertEqual(rectangle["n_tasks"], 100)
        self.assertEqual(rectangle["family_size"], 6)
        self.assertAlmostEqual(rectangle["radius"], radius)
        np.testing.assert_allclose(rectangle["lower"], -radius)
        np.testing.assert_allclose(rectangle["upper"], radius)
        self.assertAlmostEqual(12*np.exp(-100*radius*radius/2), .05)
        with self.assertRaises(ValueError):
            h.simultaneous_rectangle(np.zeros((100, 8)))

    def test_harmful_large_signal_has_zero_population_headroom(self):
        h = self.module()
        self.assertEqual(h.population_bounds(-1, 1)["upper"], 0)
        exact = h.propagate_rectangle([[-1, -1]]*3, [[1, 1]]*3)
        self.assertEqual(exact["H_interval"], [0, 0])
        uncertain = h.propagate_rectangle([[-1, 1]]*3, [[0, 1]]*3)
        self.assertEqual(uncertain["H_interval"], [0, 1])
        self.assertTrue(uncertain["vacuous"])

    def test_analysis_preserves_raw_negative_products_and_denominators(self):
        y = np.tile([[[1, 0, 1, 1], [0, 1, 0, 0]]], (1000, 1, 1))
        result = self.module().analyze(y)
        self.assertEqual(result["raw_Y"], y.tolist())
        self.assertEqual(result["raw_mu"], [0, 0, 0])
        self.assertEqual(result["raw_S"], [-1, 0, 0])
        self.assertEqual(result["n_tasks"], 1000)
        self.assertEqual(result["arm_denominators"]["A3"], {"tasks": 1000, "round_observations": 2000})
        self.assertIsNone(result["descriptive"]["A1"]["lower"])
        self.assertEqual(result["confidence"]["status"], "incompatible_confidence_region")

    def test_empty_data_and_duplicate_task_ids(self):
        h = self.module()
        result = h.analyze(np.empty((0, 2, 4)))
        self.assertEqual(result["status"], "no_data")
        self.assertIsNone(result["raw_mu"])
        self.assertIsNone(result["confidence"]["H_interval"])
        with self.assertRaises(ValueError):
            h.analyze(np.ones((2, 2, 4)), task_ids=["same", "same"])

    def test_infeasible_point_does_not_suppress_feasible_uncertainty_region(self):
        result = self.module().analyze(np.array([[[1, 0, 1, 1], [0, 1, 0, 0]]]))
        self.assertEqual(result["raw_S"], [-1, 0, 0])
        self.assertIsNone(result["descriptive"]["A1"]["upper"])
        self.assertEqual(result["confidence"]["H_interval"], [0, 1])
        self.assertTrue(result["confidence"]["vacuous"])

    def test_cli_separates_configuration_strata_and_keeps_revision(self):
        h = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path, output = root / "rows.json", root / "headroom.json"
            base = {"task_id": "same_task", "model": "actor", "model_revision": "revision", "env": "alfworld", "split": "test", "scaffold": "grounded-react-v1", "Y": [[1, 0, 0, 0], [1, 0, 0, 0]]}
            path.write_text(json.dumps([{**base, "config_sha256": digest} for digest in ["one", "two"]]))
            command = [sys.executable, "-B", h.__file__, "--rows", str(path), "--out", str(output)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual(len(saved["strata"]), 2)
            for stratum in saved["strata"]:
                self.assertEqual(stratum["n_tasks"], 1)
                self.assertEqual(stratum["descriptive_H_interval"], [0, 0])
                self.assertEqual(stratum["provenance"]["model_revisions"], ["revision"])
            command[-1] = str(path)
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(len(json.loads(path.read_text())), 2)

    def test_cli_no_data_and_input_hash(self):
        h = self.module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            path = root / "rows.json"
            path.write_text("[]\n")
            output = root / "headroom.json"
            result = subprocess.run([sys.executable, "-B", h.__file__, "--rows", str(path), "--out", str(output)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads(output.read_text())
            self.assertEqual(saved["status"], "no_data")
            self.assertEqual(saved["strata"], [])
            self.assertEqual(saved["input_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
