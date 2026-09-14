import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


class ProtocolTests(unittest.TestCase):
    def test_candidate_selection_includes_safe_fallbacks(self):
        freeze = module(ROOT / "controller/freeze_predictions.py", "freeze_predictions_test")
        direct = {"strata": [{"model": "m", "env": "e", "selected": {"oof_utility": .6, "oof_firing_rate": .2}}]}
        arm = {"strata": [{"model": "m", "env": "e", "mean_utility": [.5, .4, .55, .3], "best_fixed_arm": 2, "policies": {"REPEATED": {"oof_utility": .52, "oof_firing_rate": .8}, "SINGLE_1": {"oof_utility": .58, "oof_firing_rate": .4}}}]}
        result = freeze.candidate_selection(direct, arm, "m", "e")
        self.assertEqual({row["name"] for row in result["candidates"]}, {"CONTINUE", "BEST_FIXED", "DIRECT_ADVANTAGE", "SINGLE_1"})
        self.assertEqual(result["selected"]["name"], "DIRECT_ADVANTAGE")
        self.assertEqual(result["arm_outcome_parameterization_control"], "REPEATED")

    def test_panel_exposure_collects_goal_and_variation(self):
        freeze = module(ROOT / "extension/freeze_panel.py", "freeze_panel_test")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "exposure.json"
            path.write_text(json.dumps({"tasks": [{"task_spec": {"game_file": "/x/train/goal-7/trial/game.tw-pddl"}}, {"task_spec": {"task_name": "boil", "variation": 21}}]}))
            goals, science, records = freeze.exposure([path])
        self.assertEqual(goals, {"goal-7"})
        self.assertEqual(science, {("boil", 21)})
        self.assertEqual(len(records), 1)

    def test_group_bootstrap_preserves_task_weighting(self):
        evaluate = module(ROOT / "controller/evaluate_confirmation.py", "evaluate_confirmation_test")
        result = evaluate.bootstrap(np.array([1., 0., 0.]), ["a", "a", "b"], 9, draws=2000)
        self.assertAlmostEqual(result["mean"], 1 / 3)
        self.assertEqual(result["n_groups"], 2)
        self.assertGreaterEqual(result["group_sign_flip_p"], 0)
        self.assertLessEqual(result["group_sign_flip_p"], 1)

    def test_cell_costs_align_rounds_and_arms(self):
        evaluate = module(ROOT / "controller/evaluate_confirmation.py", "evaluate_confirmation_cost_test")
        rows = [{"cells": [
            {"round": 1, "arm": "A2", "local_call_events": [{"input_tokens": 7}, {"input_tokens": 5}]},
            {"round": 0, "arm": "A0", "local_call_events": [{"input_tokens": 3}]},
        ]}]
        costs = evaluate.cell_costs(rows, "input_tokens")
        self.assertEqual(costs.shape, (1, 2, 4))
        self.assertEqual(costs[0, 1, 2], 12)
        self.assertEqual(costs[0, 0, 0], 3)


if __name__ == "__main__":
    unittest.main()
