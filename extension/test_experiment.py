import importlib.util
import tempfile
import json
from pathlib import Path
import unittest


class PrefixTest(unittest.TestCase):
    def test_mismatched_restoration_closes_before_a_suffix_can_run(self):
        from experiment import checked_restore
        from src.env_adapters import BaseAdapter

        class Fixture(BaseAdapter):
            latest = None

            def __init__(self, task_spec):
                super().__init__(task_spec)
                Fixture.latest = self
                self.closed = False

            def reset(self):
                self.actions = []
                self.steps = 0

            def step(self, action):
                self.actions.append(action)
                self.steps += 1

            def state_hash(self):
                return "|".join(self.actions)

            def close(self):
                self.closed = True

        checkpoint = {"env_name": "fixture", "task_spec": {}, "actions": ["look"], "step_index": 1, "state_hash": "look"}
        restored = checked_restore(Fixture, checkpoint)
        self.assertEqual(restored.actions, ["look"])
        self.assertFalse(restored.closed)
        with self.assertRaisesRegex(RuntimeError, "Replay mismatch"):
            checked_restore(Fixture, {**checkpoint, "state_hash": "different"})
        self.assertTrue(Fixture.latest.closed)

    def test_grounded_scaffold_corrects_executed_command_syntax_without_changing_scienceworld(self):
        from experiment import configure_scaffold
        from src.agent import ENV_RULES
        science = ENV_RULES["scienceworld"]
        configure_scaffold("grounded-action-v1")
        self.assertIn("move <object> to <receptacle>", ENV_RULES["alfworld"])
        self.assertNotIn("put <object> in/on <receptacle>", ENV_RULES["alfworld"])
        self.assertEqual(ENV_RULES["scienceworld"], science)
        with self.assertRaises(ValueError):
            configure_scaffold("unknown")
        from src import agent
        configure_scaffold("grounded-react-v1")
        prompt = agent.build_prompt("alfworld", "goal", [], ["look"], 0, 50)
        self.assertIn("THOUGHT:", prompt)
        self.assertNotIn("exactly one line", prompt)
        self.assertEqual(agent.parse_action("THOUGHT: Observe the room.\nACTION: look"), ("look", False))
        configure_scaffold("legacy-action")
        self.assertNotIn("THOUGHT:", agent.build_prompt("alfworld", "goal", [], ["look"], 0, 50))

    def test_resume_rejects_changed_configuration_and_preserves_failed_attempt(self):
        import experiment
        self.assertTrue(hasattr(experiment, "load_episode"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "baseline.json"
            common = {"config_sha256": "original", "model": "actor", "task_id": "task"}
            saved = {**common, "episode": {"failure": {"reason": "cli_call_failed"}, "suspended": False}}
            path.write_text(json.dumps(saved))
            with self.assertRaises(ValueError):
                experiment.load_episode(path, {**common, "config_sha256": "changed"})
            self.assertIsNone(experiment.load_episode(path, common))
            self.assertEqual(json.loads(path.with_suffix(".attempt1.json").read_text()), saved)
            path.write_text(json.dumps(saved))
            self.assertIsNone(experiment.load_episode(path, common))
            self.assertEqual(json.loads(path.with_suffix(".attempt2.json").read_text()), saved)

    def test_prefix_has_no_future_and_rollback_injects_at_restored_step(self):
        self.assertIsNotNone(importlib.util.find_spec("experiment"))
        from experiment import fork_payload

        baseline = {
            "env": "alfworld", "task_spec": {"game_file": "game"},
            "actions": ["look", "open cabinet 1", "FUTURE"],
            "steps": [
                {"state_hash": "h1", "prev_state_hash": "h0", "transcript_len_before": 1, "done": False},
                {"state_hash": "h2", "prev_state_hash": "h1", "transcript_len_before": 3, "done": False},
                {"state_hash": "h3", "prev_state_hash": "h2", "transcript_len_before": 5, "done": True},
            ],
            "transcript": [{"kind": "observation", "text": "start", "step": 0}, {"kind": "action", "text": "look", "step": 0}, {"kind": "observation", "text": "room", "step": 1}, {"kind": "action", "text": "open cabinet 1", "step": 1}, {"kind": "observation", "text": "opened", "step": 2}, {"kind": "action", "text": "FUTURE", "step": 2}],
        }
        normal = fork_payload(baseline, 2, "A0")
        self.assertEqual(normal["restore_from"]["actions"], ["look", "open cabinet 1"])
        self.assertEqual(normal["expected_state_hash"], "h2")
        self.assertEqual(len(normal["prior_transcript"]), 5)
        rollback = fork_payload(baseline, 2, "A3")
        self.assertEqual(rollback["restore_from"]["actions"], ["look"])
        self.assertEqual(rollback["restore_from"]["step_index"], 1)
        self.assertEqual(rollback["expected_state_hash"], "h1")
        self.assertEqual(len(rollback["prior_transcript"]), 3)
        self.assertIsNone(fork_payload(baseline, 3, "A0"))


if __name__ == "__main__":
    unittest.main()
