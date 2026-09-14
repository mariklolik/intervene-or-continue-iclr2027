import hashlib
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


class AnalysisTests(unittest.TestCase):
    def analysis(self):
        self.assertIsNotNone(importlib.util.find_spec("analysis"), "The prospective analysis module is not implemented")
        return importlib.import_module("analysis")

    def fixture(self, root):
        task = {"task_id": "t1", "split": "test", "env": "alfworld", "task_spec": {"game_file": "game", "task_type": "pick"}, "checkpoint_step": 1, "seed": 17}
        config = {"model": "actor", "model_revision": "revision", "rounds": 2, "arms": {"A0": None, "A1": "warning", "A2": "replan", "A3": "rollback"}, "tasks": [task]}
        path = root / "config.json"
        path.write_text(json.dumps(config))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        directory = root / "raw" / "t1"
        directory.mkdir(parents=True)
        prefix_step = {"step": 0, "state_hash": "h1", "prev_state_hash": "h0", "done": False, "parsed_action": "look", "observation": "room", "prompt_chars": 20, "reward": 0}
        episode = {"env": "alfworld", "task_spec": task["task_spec"], "success": True, "step_limit": 50, "failure": None, "suspended": False, "restore_hash_ok": None, "steps": [prefix_step, {"step": 1, "done": True, "observation": "SECRET FUTURE", "reward": 99}], "actions": ["look", "SECRET FUTURE"], "transcript": [{"step": 0, "kind": "observation", "text": "start"}, {"step": 0, "kind": "action", "text": "look"}, {"step": 1, "kind": "observation", "text": "room"}, {"step": 1, "kind": "action", "text": "SECRET FUTURE"}], "local_call_events": []}
        common = {"task_id": "t1", "split": "test", "model": "actor", "config_sha256": digest}
        (directory / "baseline.json").write_text(json.dumps({**common, "arm": "BASELINE", "round": -1, "seed": 17, "episode": episode}))
        for round_id, arm in itertools.product(range(2), config["arms"]):
            restored = int(arm != "A3")
            seed = int.from_bytes(hashlib.sha256(f"17:{round_id}:{arm}".encode()).digest()[:4], "big")
            request_seed = int.from_bytes(hashlib.sha256(f"{seed}:0:actor".encode()).digest()[:4], "big") % 2147483647
            ep = {"env": "alfworld", "task_spec": task["task_spec"], "success": bool(round_id), "failure": None, "suspended": False, "restore_hash_ok": True, "segment_start_step": restored, "step_limit": 50 - int(arm == "A3"), "interventions_injected": [] if arm == "A0" else [restored], "steps": [{"step": restored, "prev_state_hash": "h0" if arm == "A3" else "h1"}], "transcript": [] if arm == "A0" else [{"step": restored, "kind": "overseer", "text": config["arms"][arm]}], "local_call_events": [{"ok": True, "usage_known": True, "input_tokens": 2, "output_tokens": 3, "wall_s": 0.5, "model": "actor", "tag": "actor", "seed": request_seed}]}
            record = {**common, "arm": arm, "round": round_id, "seed": seed, "checkpoint_step": 1, "restored_step": restored, "episode": ep}
            (directory / f"round{round_id}-{arm}.json").write_text(json.dumps(record))
        return path, root / "raw", directory

    def test_independent_null_exact_enumeration_exposes_hindsight_gap(self):
        a = self.analysis()
        y = np.array(list(itertools.product([0, 1], repeat=8))).reshape(-1, 2, 4)
        result = a.diagnostics(y)
        self.assertAlmostEqual(result["same_round_opportunity"].mean(), 0.4375)
        self.assertAlmostEqual(result["cross_selected_uplift"].mean(), 0.0)
        self.assertAlmostEqual(result["gap"].mean(), 0.4375)
        np.testing.assert_allclose(result["products"].mean(axis=0), [0, 0, 0, 0])

    def test_wrong_round_ordering_and_negative_products_are_visible(self):
        r = self.analysis().diagnostics(np.array([[[0, 1, 0, 0], [1, 0, 1, 0]]]))
        np.testing.assert_array_equal(r["selected_actions"], [[1, 0]])
        self.assertEqual(r["same_round_opportunity"].item(), 0.5)
        self.assertEqual(r["cross_selected_uplift"].item(), -0.5)
        self.assertEqual(r["gap"].item(), 1)
        self.assertEqual(r["products"][0, 1], -1)

    def test_shared_baseline_creates_spurious_squared_moment(self):
        y = np.array(list(itertools.product([0, 1], repeat=8))).reshape(-1, 2, 4)
        y[:, 1, 0] = y[:, 0, 0]
        np.testing.assert_allclose(self.analysis().diagnostics(y)["products"].mean(axis=0), [0, .25, .25, .25])

    def test_continue_wins_ties_and_stable_benefit_has_zero_gap(self):
        a = self.analysis()
        ties = a.diagnostics(np.ones((3, 2, 4), dtype=int))
        np.testing.assert_array_equal(ties["selected_actions"], np.zeros((3, 2), dtype=int))
        r = a.diagnostics(np.array([[[0, 1, 0, 0], [0, 1, 0, 0]]]))
        self.assertEqual(r["cross_selected_uplift"].item(), 1)
        self.assertEqual(r["gap"].item(), 0)

    def test_bootstrap_unit_is_task_and_degenerate_interval_is_bounded(self):
        a = self.analysis()
        result = a.summarize_tasks([0, 1], (0, 1))
        self.assertEqual(result["n_tasks"], 2)
        self.assertEqual(result["mean"], 0.5)
        self.assertEqual(result, a.summarize_tasks([0, 1], (0, 1)))
        zero = a.summarize_tasks(np.zeros(100), (0, 1))
        self.assertTrue(zero["bootstrap_degenerate"])
        self.assertEqual(zero["interval_method"], "bounded_hoeffding")
        self.assertGreater(zero["interval_95"][1], .13)
        self.assertLess(zero["interval_95"][1], .14)
        with self.assertRaises(ValueError):
            a.summarize_tasks([[0, 1], [0, 1]], (0, 1))

    def test_empty_and_invalid_arrays_do_not_manufacture_estimates(self):
        a = self.analysis()
        self.assertIsNone(a.summarize_tasks([], (0, 1))["mean"])
        self.assertEqual(a.diagnostics(np.empty((0, 2, 4)))["gap"].shape, (0,))
        for y in [np.ones((2, 4)), np.ones((2, 2, 4)) * .5]:
            with self.assertRaises(ValueError):
                a.diagnostics(y)

    def test_complete_loader_retains_raw_outcomes_and_prefix_only_features(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            accepted, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(accepted[0]["Y"], [[0, 0, 0, 0], [1, 1, 1, 1]])
            prefix = accepted[0]["prefix_only"]
            self.assertNotIn("SECRET FUTURE", json.dumps(prefix))
            self.assertNotIn("success", prefix)
            self.assertEqual(prefix["features"]["last_prompt_len"], 20)
            self.assertEqual(prefix["features"]["cum_reward"], 0)
            self.assertEqual(audit["usage"]["requests"], 8)

    def test_hidden_state_features_are_not_admitted(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            accepted, audit = self.analysis().load_blocks(config, raw)
            prefix = accepted[0]["prefix_only"]
            for key in ["n_state_unchanged", "rate_state_unchanged", "stall_state_unchanged_run", "n_irreversible", "last_irreversible"]:
                self.assertNotIn(key, prefix["features"])
            path = directory / "baseline.json"
            record = json.loads(path.read_text())
            record["episode"]["steps"][0].update(state_changed=False, reversible=False)
            path.write_text(json.dumps(record))
            changed, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(prefix["features"], changed[0]["prefix_only"]["features"])

    def test_failed_attempt_usage_is_counted_even_when_final_cell_is_accepted(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            attempt = json.loads((directory / "round0-A1.json").read_text())
            attempt["episode"]["failure"] = {"reason": "cli_call_failed"}
            attempt["episode"]["local_call_events"] = [{"ok": False, "usage_known": False, "input_tokens": 5, "output_tokens": 7, "wall_s": 2}]
            (directory / "round0-A1.attempt1.json").write_text(json.dumps(attempt))
            accepted, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(len(accepted), 1)
            self.assertEqual(audit["usage"]["requests"], 9)
            self.assertEqual(audit["usage"]["input_tokens"], 21)
            self.assertEqual(audit["usage"]["output_tokens"], 31)
            self.assertEqual(audit["usage"]["unknown_usage_requests"], 1)
            self.assertEqual(len(audit["episode_inventory"]), 10)

    def test_missing_fresh_control_blocks_entire_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            (directory / "round1-A0.json").unlink()
            accepted, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(accepted, [])
            self.assertIn("round1-A0.json:missing", audit["tasks"][0]["reasons"])
            self.assertEqual(audit["arm_denominators"]["A0"]["valid_cells"], 1)
            self.assertEqual(audit["arm_denominators"]["A0"]["confirmed_eligible_cells"], 2)
            self.assertEqual(audit["arm_denominators"]["A1"]["accepted_block_cells"], 0)

    def test_wrong_model_restore_delivery_and_seed_block_task(self):
        for change in ["model", "restore", "delivery", "seed", "message", "budget", "hash", "response_model", "request_seed", "task_spec"]:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temporary:
                config, raw, directory = self.fixture(Path(temporary))
                path = directory / "round1-A3.json"
                record = json.loads(path.read_text())
                if change == "model": record["model"] = "wrong"
                if change == "restore": record["episode"]["restore_hash_ok"] = False
                if change == "delivery": record["episode"]["interventions_injected"] = []
                if change == "seed": record["seed"] = 17
                if change == "message": record["episode"]["transcript"][0]["text"] = "wrong"
                if change == "budget": record["episode"]["step_limit"] = 50
                if change == "hash": record["config_sha256"] = "wrong"
                if change == "response_model": record["episode"]["local_call_events"][0]["model"] = "wrong"
                if change == "request_seed": record["episode"]["local_call_events"][0]["seed"] = 17
                if change == "task_spec": record["episode"]["task_spec"] = {"game_file": "wrong"}
                path.write_text(json.dumps(record))
                accepted, audit = self.analysis().load_blocks(config, raw)
                self.assertEqual(accepted, [])
                self.assertTrue(audit["tasks"][0]["reasons"])

    def test_invalid_baseline_environment_is_excluded_and_corrupt_file_is_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            path = directory / "baseline.json"
            record = json.loads(path.read_text())
            record["episode"]["env"] = "scienceworld"
            path.write_text(json.dumps(record))
            (directory / "round0-A2.attempt1.json").write_text("{broken")
            accepted, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(accepted, [])
            self.assertEqual(audit["tasks"][0]["state"], "baseline_incomplete_or_invalid")
            self.assertEqual(len(audit["file_issues"]), 1)
            self.assertEqual(audit["usage"]["requests"], 8)

    def test_early_terminal_and_no_data_are_enumerated(self):
        with tempfile.TemporaryDirectory() as temporary:
            config, raw, directory = self.fixture(Path(temporary))
            path = directory / "baseline.json"
            record = json.loads(path.read_text())
            record["episode"]["steps"][0]["done"] = True
            path.write_text(json.dumps(record))
            accepted, audit = self.analysis().load_blocks(config, raw)
            self.assertEqual(accepted, [])
            self.assertEqual(audit["tasks"][0]["state"], "terminal_before_checkpoint")
            self.assertEqual(audit["usage"]["requests"], 8)
            accepted, audit = self.analysis().load_blocks(config, Path(temporary) / "absent")
            self.assertEqual(accepted, [])
            self.assertEqual(audit["tasks"][0]["state"], "not_started")

    def test_cli_no_data_has_outputs_and_no_zero_estimate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config, raw, directory = self.fixture(root)
            result = subprocess.run([sys.executable, "-B", str(Path(__file__).with_name("analysis.py")), "--config", str(config), "--raw", str(root / "absent"), "--out", str(root / "out")], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            metrics = json.loads((root / "out/metrics.json").read_text())
            self.assertEqual(metrics["status"], "no_complete_blocks")
            self.assertEqual(json.loads((root / "out/rows.json").read_text()), [])
            self.assertIsNone(metrics["strata"]["actor|alfworld|test"]["diagnostics"]["gap"]["mean"])


if __name__ == "__main__":
    unittest.main()
