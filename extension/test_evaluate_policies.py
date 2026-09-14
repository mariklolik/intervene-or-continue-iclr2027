import copy
import importlib
import importlib.util
import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

from policies import METHODS, digest


class EvaluationTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec("evaluate_policies"))
        return importlib.import_module("evaluate_policies")

    def fixture(self, all_strata=False):
        strata = [("Qwen3-8B", "alfworld")]
        if all_strata:
            strata = [(m, e) for m in ["Qwen3-8B", "Qwen2.5-7B-Instruct"] for e in ["alfworld", "scienceworld"]]
        rows, dev, audits, receipts = [], [], [], []
        for model, env in strata:
            context = {"model": model, "env": env, "model_revision": "revision", "scaffold": "grounded-react-v1", "prefix_only": {"features": {"steps": 4}, "recent_text": "room"}}
            dev.append({**context, "task_id": "dev", "split": "dev", "config_sha256": "dev-shard", "Y": [[1, 0, 0, 0]]*2})
            cells = [{"round": r, "arm": f"A{a}", "local_call_events": [{"input_tokens": a+1, "output_tokens": 2*(a+1), "wall_s": .5, "usage_known": True}]} for r in range(2) for a in range(4)]
            rows.append({**context, "task_id": "test", "split": "test", "config_sha256": "test-shard", "Y": [[0, 1, 0, 0], [1, 0, 1, 0]], "cells": cells})
            audits.append({"model": model, "phase": "test", "config_sha256": "test-shard", "tasks": [{"task_id": "test", "env": env, "split": "test", "state": "complete", "checkpoint_eligible": True}], "usage": {"requests": 99}})
            receipts.append({"model": model, "env": env, "dev_task_ids": ["dev"], "best_fixed_arm": 0, "random_rate": .6})
        dev.sort(key=lambda r: (r["model"], r["env"], r["task_id"], digest(r["prefix_only"])))
        manifest = {"dev_data_sha256": digest(dev), "source_sha256": "code", "bundle_sha256": "bundle", "strata": receipts}
        probabilities = {m: [[1, 0, 0, 0] for _ in rows] for m in METHODS}
        probabilities["REPEATED"] = [[0, 1, 0, 0] for _ in rows]
        probabilities["RATE_RANDOM"] = [[.4, .2, .2, .2] for _ in rows]
        predictions = {"rows": [{k: r[k] for k in ["model", "env", "split", "task_id"]} for r in rows], "training_data_sha256": digest(dev), "policy_source_sha256": "code", "policies": {m: {"probabilities": p} for m, p in probabilities.items()}}
        return rows, predictions, manifest, audits, dev

    def test_hand_computed_weighted_values_and_paired_sign(self):
        h = self.module()
        y = np.array([[[0, 1, 0, 0], [1, 0, 1, 0]], [[0, 1, 0, 0], [0, 1, 0, 0]]])
        p = np.array([[.4, .2, .2, .2], [0, 1, 0, 0]])
        np.testing.assert_allclose(h.policy_values(y, p), [.4, 1])
        difference = h.policy_values(y, p)-h.policy_values(y, [[1, 0, 0, 0]]*2)
        np.testing.assert_allclose(difference, [-.1, 1])
        result = h.contrast(difference, (-1, 1), 3)
        self.assertEqual(result["n_tasks"], 2)
        np.testing.assert_allclose(result["finite_frame_bounds"], [-.1/3, 1.9/3])
        self.assertEqual(len(result["bootstrap_means"]), 10000)
        with self.assertRaises(ValueError):
            h.policy_values(y.reshape(2, 4, 2), p)

    def test_holm_and_bounded_gate_are_hand_computable(self):
        h = self.module()
        np.testing.assert_allclose(h.holm([.01, .04, .03, .5]), [.04, .09, .09, .5])
        result = h.contrast(np.ones(100), (-1, 1), 100)
        self.assertAlmostEqual(result["p_bounded"], np.exp(-50))
        self.assertAlmostEqual(result["simultaneous_margin"], np.sqrt(2*np.log(71/.05)/100))
        self.assertGreater(result["simultaneous_lower"], 0)
        self.assertTrue(result["bootstrap_degenerate"])

    def test_family_has_exactly_71_slots_even_with_missing_strata(self):
        result = self.module().evaluate(*self.fixture())
        self.assertEqual(len(result["family"]), 71)
        self.assertEqual(len({r["id"] for r in result["family"]}), 71)
        self.assertEqual(len(result["strata"]), 4)
        self.assertEqual(sum(len(s["contrasts"]) for s in result["strata"]), 68)
        self.assertEqual(sum(r["n_tasks"] == 0 for r in result["family"]), 54)
        self.assertIsNone(result["equal_stratum_aggregate"])

    def test_betting_refines_zero_heavy_signed_contrasts(self):
        h = self.module()
        result = h.contrast([1]*20+[0]*114, (-1, 1), 134)
        self.assertLess(result["hoeffding_simultaneous_lower"], 0)
        self.assertGreater(result["simultaneous_lower"], 0)
        self.assertLess(result["p_confirmatory"], .05/71)
        self.assertEqual(result["confirmatory_method"], "fixed_fraction_mixture_betting")
        gap = h.contrast([0, .5, 1], (0, 1), 3)
        self.assertEqual(gap["p_confirmatory"], gap["p_bounded"])

    def test_betting_matches_analytic_all_success_wealth(self):
        h = self.module()
        fractions = np.arange(1, 101)/100
        expected = np.mean((1+fractions)**10)
        result = h.bounded_betting(np.ones(10), alpha=.05)
        self.assertAlmostEqual(result["p_value"], 1/expected)
        lower = result["lower"]
        self.assertGreaterEqual(np.mean((1+fractions*(1-lower)/(1+lower))**10), 20-1e-10)
        self.assertEqual(h.bounded_betting([-1]*20)["lower"], -1)
        self.assertIsNone(h.bounded_betting([])["lower"])
        self.assertEqual(h.bounded_betting([])["p_value"], 1)
        for values in ([2], [float("nan")], [[0, 1]]):
            with self.assertRaises(ValueError):
                h.bounded_betting(values)

    def test_betting_type_one_error_under_heterogeneous_average_null(self):
        h = self.module()
        probabilities = np.array([.2, .8, .3, .7, .5, .5])
        rejection = 0
        for code in range(64):
            bits = np.array([(code >> i) & 1 for i in range(6)])
            probability = np.prod(np.where(bits, probabilities, 1-probabilities))
            rejection += probability*(h.bounded_betting(2*bits-1)["p_value"] <= .1)
        self.assertLessEqual(rejection, .1+1e-12)

    def test_full_frame_sensitivity_penalizes_unknown_outcomes(self):
        h = self.module()
        rows, predictions, manifest, audits, dev = self.fixture()
        for task_id, eligible, state in [("missing", True, "incomplete_or_invalid"), ("early", False, "terminal_before_checkpoint"), ("unknown", None, "not_started")]:
            audits[0]["tasks"].append({"task_id": task_id, "env": "alfworld", "split": "test", "state": state, "checkpoint_eligible": eligible})
        result = h.evaluate(rows, predictions, manifest, audits, dev)["strata"][0]
        sensitivity = result["contrasts"]["REPEATED_vs_CONTINUE"]["planned_frame_sensitivity"]
        self.assertEqual(sensitivity["raw_task_values"], [0, -1, 0, -1])
        self.assertEqual(sensitivity["mean"], -.5)
        self.assertEqual(sensitivity["n_tasks"], 4)
        self.assertFalse(result["policy_gate"]["full_frame_robustness_met"])
        original = h.bounded_betting([1]*20+[0]*114)
        censored = h.bounded_betting([-1]*5+[1]*15+[0]*114)
        self.assertGreaterEqual(censored["p_value"], original["p_value"])
        self.assertLessEqual(censored["lower"], original["lower"])

    def test_omitting_a_complete_audit_task_is_not_silent_missingness(self):
        rows, predictions, manifest, audits, dev = self.fixture()
        predictions["rows"] = []
        for policy in predictions["policies"].values():
            policy["probabilities"] = []
        with self.assertRaises(ValueError):
            self.module().evaluate([], predictions, manifest, audits, dev)

    def test_expected_suffix_cost_and_unknown_usage_are_explicit(self):
        rows, predictions, manifest, audits, dev = self.fixture()
        rows[0]["cells"][0]["local_call_events"][0]["usage_known"] = False
        result = self.module().evaluate(rows, predictions, manifest, audits, dev)
        stratum = next(s for s in result["strata"] if s["n_tasks"])
        cost = stratum["policies"]["RATE_RANDOM"]["suffix_cost"]
        self.assertAlmostEqual(cost["means"]["input_tokens"], 2.2)
        self.assertAlmostEqual(cost["means"]["output_tokens"], 4.4)
        self.assertAlmostEqual(cost["means"]["requests"], 1)
        self.assertAlmostEqual(cost["means"]["unknown_usage_requests"], .2)
        self.assertEqual(result["collection_usage_by_audit"][0]["usage"]["requests"], 99)

    def test_leakage_alignment_config_and_context_errors_are_rejected(self):
        for defect in ["dev_overlap", "order", "config", "revision", "probability", "training_hash", "baseline", "random_rate", "duplicate_audit"]:
            with self.subTest(defect=defect):
                rows, predictions, manifest, audits, dev = self.fixture()
                if defect == "dev_overlap": rows[0]["task_id"] = "dev"
                if defect == "order": predictions["rows"][0]["task_id"] = "wrong"
                if defect == "config": rows[0]["config_sha256"] = "wrong"
                if defect == "revision": rows[0]["model_revision"] = "wrong"
                if defect == "probability": predictions["policies"]["REPEATED"]["probabilities"][0] = [-1, 2, 0, 0]
                if defect == "training_hash": manifest["dev_data_sha256"] = "wrong"
                if defect == "baseline": predictions["policies"]["CONTINUE"]["probabilities"][0] = [0, 1, 0, 0]
                if defect == "random_rate": predictions["policies"]["RATE_RANDOM"]["probabilities"][0] = [.7, .1, .1, .1]
                if defect == "duplicate_audit": audits.append(copy.deepcopy(audits[0]))
                with self.assertRaises(ValueError):
                    self.module().evaluate(rows, predictions, manifest, audits, dev)

    def test_missingness_is_distinct_from_population_uncertainty(self):
        rows, predictions, manifest, audits, dev = self.fixture()
        audits[0]["tasks"] += [{"task_id": "missing", "env": "alfworld", "split": "test", "state": "incomplete", "checkpoint_eligible": True}, {"task_id": "unknown", "env": "alfworld", "split": "test", "state": "not_started", "checkpoint_eligible": None}, {"task_id": "terminal", "env": "alfworld", "split": "test", "state": "terminal_before_checkpoint", "checkpoint_eligible": False}]
        result = self.module().evaluate(rows, predictions, manifest, audits, dev)
        s = next(s for s in result["strata"] if s["n_tasks"])
        self.assertEqual(s["frame"]["confirmed_eligible"], 2)
        self.assertEqual(s["frame"]["unknown_eligibility"], 1)
        self.assertEqual(s["contrasts"]["REPEATED_vs_CONTINUE"]["finite_frame_bounds"], [-.5, .5])
        self.assertEqual(s["policies"]["REPEATED"]["value"]["mean"], .5)

    def test_equal_stratum_aggregate_does_not_double_shared_tasks(self):
        result = self.module().evaluate(*self.fixture(all_strata=True))
        aggregate = result["equal_stratum_aggregate"]
        self.assertEqual(aggregate["weights"], [.25]*4)
        self.assertEqual(aggregate["distinct_environment_task_ids"], 2)
        self.assertAlmostEqual(aggregate["policy_values"]["RATE_RANDOM"], .4)
        self.assertEqual(aggregate["contrasts"]["REPEATED_vs_CONTINUE"]["mean"], 0)
        self.assertFalse(any(s["policy_gate"]["statistical_conditions_met"] for s in result["strata"]))

    def test_no_data_has_no_zero_estimates(self):
        rows, predictions, manifest, audits, dev = self.fixture()
        predictions["rows"] = []
        for policy in predictions["policies"].values():
            policy["probabilities"] = []
        result = self.module().evaluate([], predictions, manifest, [], dev)
        self.assertTrue(all(r["p_holm"] == 1 for r in result["family"]))
        self.assertTrue(all(s["policies"]["CONTINUE"]["value"]["mean"] is None for s in result["strata"]))
        self.assertIsNone(result["equal_stratum_aggregate"])

    def test_cli_verifies_frozen_bundle_and_writes_no_data_receipt(self):
        h = self.module()
        rows, predictions, manifest, audits, dev = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            audits[0]["tasks"][0].update(state="not_started", checkpoint_eligible=None)
            source_hash = hashlib.sha256(Path(importlib.import_module("policies").__file__).read_bytes()).hexdigest()
            bundle = {"strata": {}, "training_tasks": [], "training_data_sha256": manifest["dev_data_sha256"], "policy_source_sha256": source_hash}
            joblib.dump(bundle, root/"policies.joblib")
            manifest.update(source_sha256=source_hash, bundle_sha256=hashlib.sha256((root/"policies.joblib").read_bytes()).hexdigest())
            for name, data in [("manifest", manifest), ("dev", dev), ("rows", []), ("audit", audits[0])]:
                (root/f"{name}.json").write_text(json.dumps(data))
            command = [sys.executable, "-B", h.__file__, "--rows", str(root/"rows.json"), "--dev-rows", str(root/"dev.json"), "--audits", str(root/"audit.json"), "--model", str(root), "--out", str(root/"result.json")]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            saved = json.loads((root/"result.json").read_text())
            self.assertEqual(saved["family_size"], 71)
            self.assertEqual(len(saved["inputs"]), 5)
            self.assertTrue(all(r["mean"] is None for r in saved["family"]))
            (root/"policies.joblib").write_bytes(b"corrupt")
            command[-1] = str(root/"second.json")
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Frozen bundle hash mismatch", result.stderr)


if __name__ == "__main__":
    unittest.main()
