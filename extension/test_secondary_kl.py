import copy
import hashlib
import itertools
import json
import math
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from decimal import Decimal, localcontext
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

from evaluate_policies import STRATA
from joint_headroom import analyze
from secondary_kl import analyze_documents, envelope, kl_limits


def documents(n=2, early=1, missing=1):
    evaluation = {"strata": [], "inputs": []}
    original = {"strata": [], "evaluation_sha256": "e" * 64}
    receipt = {"schema_version": 1, "status": "PASS", "audits": [], "strata": [], "expected_frame": [], "source_sha256": "f" * 64, "inputs": []}
    for actor, model in enumerate(dict(STRATA)):
        for shard in range(2):
            config = hashlib.sha256(f"{model}:{shard}".encode()).hexdigest()
            audit = {"path": f"audit-{actor}-{shard}.json", "sha256": hashlib.sha256(config.encode()).hexdigest(), "model": model, "config_sha256": config}
            receipt["audits"].append(audit)
            evaluation["inputs"].append({key: audit[key] for key in ("path", "sha256")})
    for model, env in STRATA:
        tasks = []
        for index in range(n + early + missing):
            state, eligible = ("complete", True) if index < n else ("terminal_before_checkpoint", False) if index < n + early else ("not_started", None)
            config = hashlib.sha256(f"{model}:{index % 2}".encode()).hexdigest()
            task = {"task_id": f"{env}-{index}", "split": "test", "config_sha256": config, "state": state, "checkpoint_eligible": eligible}
            tasks.append(task)
            receipt["expected_frame"].append({**{key: task[key] for key in ("task_id", "split", "config_sha256")}, "model": model, "env": env, "master_config_sha256": "a" * 64, "task_sha256": "b" * 64})
        identifiers = [task["task_id"] for task in tasks[:n]]
        y = np.zeros((n, 2, 4), dtype=int)
        frame = {"tasks": [{**task, "model": model, "env": env} for task in tasks], "confirmed_eligible": n, "early_terminal": early, "unknown_eligibility": missing}
        configs = sorted({task["config_sha256"] for task in tasks[:n]})
        row = {"model": model, "env": env, "n_tasks": n, "task_ids": identifiers, "raw_Y": y.tolist(), "test_config_sha256": configs, "frame": frame}
        evaluation["strata"].append(row)
        original["strata"].append({"model": model, "env": env, "test_config_sha256": configs, "frame": copy.deepcopy(frame), "diagnostics": {"n_tasks": n, "task_ids": identifiers, "raw_Y": y.tolist()}, "headroom": analyze(y, task_ids=identifiers)})
        receipt["strata"].append({"model": model, "env": env, "n_tasks": len(tasks), "tasks": tasks})
    for audit in receipt["audits"]:
        tasks = [task for row in receipt["strata"] if row["model"] == audit["model"] for task in row["tasks"] if task["config_sha256"] == audit["config_sha256"]]
        audit.update(planned_tasks=len(tasks), accepted_tasks=sum(task["state"] == "complete" for task in tasks), state_counts=dict(Counter(task["state"] for task in tasks)))
    return evaluation, original, receipt


class KLBoundsTest(unittest.TestCase):
    def test_boundary_formulas_and_task_count(self):
        for n in (1, 84, 122, 134, 1000):
            low, high = kl_limits(0, n, .025)
            expected = -math.expm1(math.log(.025) / n)
            self.assertEqual(low, 0)
            self.assertGreaterEqual(high, expected - 1e-16)
            self.assertAlmostEqual(high, expected, places=13)
            low, high = kl_limits(1, n, .025)
            self.assertEqual(high, 1)
            self.assertLessEqual(low, math.exp(math.log(.025) / n) + 1e-16)
        result = envelope(np.zeros(134), np.zeros(134))
        self.assertEqual(result["n_tasks"], 134)
        self.assertLess(result["H_interval"][1], .03)
        self.assertGreater(kl_limits(0, 121, .025)[1], .03)
        self.assertGreater(kl_limits(0, 134, .025)[1], .02)
        self.assertLess(kl_limits(0, 268, .025)[1], .02)
        self.assertGreater(.98 ** 134, .025)

    def test_interior_roots_monotonicity_and_quadratic_domination(self):
        for n in (5, 134):
            previous = [0, 0]
            for x in (0, .01, .1, .3, .5, .8, .99, 1):
                limits = kl_limits(x, n, .025)
                radius = math.sqrt(math.log(40) / (2 * n))
                self.assertGreaterEqual(limits[0] + 1e-12, max(0, x - radius))
                self.assertLessEqual(limits[1], min(1, x + radius) + 1e-12)
                self.assertTrue(all(a <= b for a, b in zip(previous, limits)))
                previous = limits
                if 0 < x < 1:
                    with localcontext() as context:
                        context.prec = 100
                        dx, one = Decimal.from_float(x), Decimal(1)
                        threshold = -Decimal.from_float(.025).ln() / n
                        for endpoint in limits:
                            if 0 < endpoint < 1:
                                dm = Decimal.from_float(endpoint)
                                divergence = dx * (dx / dm).ln() + (one - dx) * ((one - dx) / (one - dm)).ln()
                                self.assertGreaterEqual(divergence, threshold)
                    def objective(m):
                        return x * math.log(x / m) + (1 - x) * math.log((1 - x) / (1 - m)) - math.log(40) / n
                    high = np.nextafter(1., 0.)
                    expected = [brentq(objective, 1e-100, x, xtol=5e-15), 1. if objective(high) <= 0 else brentq(objective, x, high, xtol=5e-15)]
                    np.testing.assert_allclose(limits, expected, atol=1e-12, rtol=0)

    def test_heterogeneous_independent_task_tail_coverage(self):
        probabilities = [.02, .1, .25, .4, .8, .95]
        mean = np.mean(probabilities)
        limits = {k: kl_limits(k / 6, 6, .05) for k in range(7)}
        failures = [0., 0.]
        for y in itertools.product((0, 1), repeat=6):
            probability = math.prod(p if value else 1 - p for value, p in zip(y, probabilities))
            low, high = limits[sum(y)]
            failures[0] += probability * (low > mean)
            failures[1] += probability * (high < mean)
        self.assertTrue(all(value <= .05 for value in failures))

    def test_pessimistic_missingness_is_pathwise_conservative(self):
        v, o = np.array([-.5, 1, 0, .5]), np.array([.5, 1, 0, .5])
        base = envelope(v, o)["H_interval"]
        for mask in itertools.product((False, True), repeat=4):
            missing = np.array(mask)
            limits = envelope(np.where(missing, -1, v), np.where(missing, 1, o))["H_interval"]
            self.assertLessEqual(limits[0], base[0])
            self.assertGreaterEqual(limits[1], base[1])
        self.assertEqual(envelope([-1] * 4, [1] * 4)["H_interval"], [0, 1])

    def test_heterogeneous_nonbernoulli_task_coverage(self):
        probabilities = [[.8, .1, .1], [.1, .2, .7], [.3, .4, .3]]
        mean = np.mean([p[1] / 2 + p[2] for p in probabilities])
        limits = {k: kl_limits(k / 6, 3, .05) for k in range(7)}
        failures = [0., 0.]
        for outcomes in itertools.product(range(3), repeat=3):
            probability = math.prod(p[value] for value, p in zip(outcomes, probabilities))
            low, high = limits[sum(outcomes)]
            failures[0] += probability * (low > mean)
            failures[1] += probability * (high < mean)
        self.assertTrue(all(value <= .05 for value in failures))

    def test_invalid_inputs_and_no_data(self):
        for values in ((math.nan, 4, .05), (-.1, 4, .05), (.2, 0, .05), (.2, 2.5, .05), (.2, 4, 0), (.2, 4, math.nan), (None, 4, .05), ("bad", 4, .05), (.2, 4, None)):
            with self.subTest(values=values), self.assertRaises(ValueError):
                kl_limits(*values)
        for v, o in (([0], []), ([math.inf], [0]), ([0], [1.1]), ([[0]], [[0]])):
            with self.assertRaises(ValueError):
                envelope(v, o)
        self.assertEqual(envelope([], [])["status"], "no_data")
        self.assertIsNone(envelope([], [])["H_interval"])


class SecondaryContractTest(unittest.TestCase):
    def test_separate_families_full_frame_and_preserved_inputs(self):
        evaluation, original, receipt = documents()
        before = copy.deepcopy((evaluation, original, receipt))
        result = analyze_documents(evaluation, original, receipt, "e" * 64)
        self.assertEqual((evaluation, original, receipt), before)
        for row, old in zip(result["strata"], original["strata"]):
            self.assertEqual(row["complete_case"]["alpha"], .05)
            self.assertEqual(row["complete_case_stricter"]["alpha"], .025)
            self.assertEqual(row["combined"]["moment_alpha"], .025)
            self.assertEqual(row["combined"]["observable_alpha"], .025)
            self.assertEqual(row["original_headroom"]["combined"], old["headroom"]["combined"])
            self.assertEqual(row["full_planned_frame"]["n_tasks"], 4)
            self.assertEqual(row["full_planned_frame"]["raw_O"], [0, 0, 0, 1])
            self.assertEqual(row["full_planned_frame"]["raw_V"], [0, 0, 0, -1])
        self.assertFalse(result["cross_stratum_joint_coverage"])

    def test_rejects_incomplete_or_inconsistent_frame_and_analysis(self):
        def mutations(evaluation, original, receipt):
            return [lambda: receipt.update(status="FAIL"), lambda: receipt["audits"].pop(), lambda: evaluation["inputs"][0].update(sha256="0" * 64), lambda: receipt["expected_frame"].pop(), lambda: receipt["strata"][0]["tasks"].append(receipt["strata"][0]["tasks"][0]), lambda: evaluation["strata"][0]["frame"].update(early_terminal=2), lambda: evaluation["strata"][0]["frame"]["tasks"][-1].update(checkpoint_eligible=False), lambda: original.update(evaluation_sha256="0" * 64), lambda: original["strata"][0]["headroom"]["raw_Y"][0][0].__setitem__(0, 1), lambda: original["strata"][0]["headroom"]["combined"].update(moment_alpha=.05), lambda: evaluation["strata"][0]["task_ids"].__setitem__(1, evaluation["strata"][0]["task_ids"][0]), lambda: evaluation["strata"][0]["raw_Y"][0][0].__setitem__(0, math.nan)]
        for index in range(12):
            evaluation, original, receipt = documents()
            mutations(evaluation, original, receipt)[index]()
            with self.subTest(index=index), self.assertRaises(ValueError):
                analyze_documents(evaluation, original, receipt, "e" * 64)

    def test_receipt_audit_denominators_are_enforced_and_order_is_irrelevant(self):
        for name, value in (("planned_tasks", 999), ("accepted_tasks", 999), ("state_counts", {})):
            evaluation, original, receipt = documents()
            receipt["audits"][0][name] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                analyze_documents(evaluation, original, receipt, "e" * 64)
        evaluation, original, receipt = documents()
        receipt["strata"].reverse()
        for row in receipt["strata"]:
            row["tasks"].reverse()
        self.assertEqual(len(analyze_documents(evaluation, original, receipt, "e" * 64)["strata"]), 4)

    def test_no_complete_blocks_full_missing_and_empty_intersection(self):
        evaluation, original, receipt = documents(n=0, early=0, missing=4)
        result = analyze_documents(evaluation, original, receipt, "e" * 64)
        self.assertEqual(result["strata"][0]["complete_case"]["status"], "no_data")
        self.assertEqual(result["strata"][0]["full_planned_frame"]["H_interval"], [0, 1])
        evaluation, original, receipt = documents(n=134, early=0, missing=0)
        original["strata"][0]["headroom"]["combined"]["moment_confidence"]["H_interval"] = [.9, 1]
        result = analyze_documents(evaluation, original, receipt, "e" * 64)
        self.assertEqual(result["strata"][0]["combined"]["status"], "empty_intersection")
        self.assertIsNone(result["strata"][0]["combined"]["H_interval"])

    def test_cli_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation, original, receipt = documents()
            data = json.dumps(evaluation).encode()
            original["evaluation_sha256"] = hashlib.sha256(data).hexdigest()
            (root / "evaluation.json").write_bytes(data)
            for name, value in (("analysis", original), ("frame", receipt)):
                (root / f"{name}.json").write_text(json.dumps(value))
            command = [sys.executable, str(Path(__file__).with_name("secondary_kl.py")), "--evaluation", str(root / "evaluation.json"), "--analysis", str(root / "analysis.json"), "--frame-validation", str(root / "frame.json"), "--out", str(root / "secondary.json")]
            subprocess.run(command, check=True, capture_output=True, timeout=20)
            result = json.loads((root / "secondary.json").read_text())
            self.assertEqual(result["inputs"]["evaluation_sha256"], hashlib.sha256(data).hexdigest())
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
