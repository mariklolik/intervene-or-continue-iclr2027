import copy
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


class WithoutOutcomes(dict):
    def __getitem__(self, key):
        if key == "Y":
            raise AssertionError("Holdout outcomes were accessed")
        return super().__getitem__(key)

    def get(self, key, default=None):
        if key == "Y":
            raise AssertionError("Holdout outcomes were accessed")
        return super().get(key, default)


class PolicyTests(unittest.TestCase):
    cache = {}

    def module(self):
        self.assertTrue((ROOT / "policies.py").exists(), "CPU policies are not implemented")
        return importlib.import_module("policies")

    def rows(self, n=64, constant=False):
        rows = []
        for i in range(n):
            failure = i % 2
            values = [1, 1, 1, 1] if constant else [1 - failure, failure, 0, 0]
            rows.append({"task_id": f"task{i:03}", "split": "dev", "model": "actor", "env": "alfworld", "Y": [values[:], values[:]], "prefix_only": {"features": {"stalled": failure, "steps_so_far": 4}, "recent_text": f"unique{i:03} room {'stalled' if failure else 'progress'}"}})
        return rows

    def holdout(self):
        rows = self.rows(2)
        for i, row in enumerate(rows):
            row.update(task_id=f"new{i}", split="test", Y=[[0, 0, 0, 0], [0, 0, 0, 0]])
        return rows

    def trained(self, constant=False, text=False):
        key = constant, text
        if key not in self.cache:
            rows = self.rows(8 if constant else 64, constant)
            ignored = WithoutOutcomes(self.holdout()[0])
            self.cache[key] = self.module().fit(rows + [ignored], text=text)
        return self.cache[key]

    def test_fresh_holdout_labels_and_suffixes_cannot_change_predictions(self):
        bundle, manifest = self.trained()
        rows = self.holdout()
        expected = self.module().predict(bundle, rows)
        for row in rows:
            row["Y"] = [[1, 0, 1, 0], [0, 1, 0, 1]]
            row["cells"] = [{"success": True, "reward": 1000}]
            row["baseline_future"] = "secret success reward 1000"
        self.assertEqual(expected, self.module().predict(bundle, [WithoutOutcomes(r) for r in rows]))
        np.testing.assert_array_equal(expected["policies"]["REPEATED"]["probabilities"], [[1, 0, 0, 0], [0, 1, 0, 0]])
        self.assertEqual(manifest["dev_records"], 64)

    def test_task_folds_and_text_dictionary_exclude_validation_tasks(self):
        bundle, manifest = self.trained(text=True)
        stratum = manifest["strata"][0]
        for method in stratum["policies"].values():
            for fold in method.get("folds", []):
                self.assertFalse(set(fold["train_task_ids"]) & set(fold["validation_task_ids"]))
                for task_id in fold["validation_task_ids"]:
                    self.assertNotIn("unique" + task_id[4:], fold["text_vocabulary"])
        rows = self.rows(16)
        duplicated = [copy.deepcopy(row) for row in rows for _ in range(2)]
        for train, valid in self.module().task_folds(duplicated):
            self.assertFalse({duplicated[i]["task_id"] for i in train} & {duplicated[i]["task_id"] for i in valid})

    def test_constant_success_and_missing_pair_classes_preserve_continue_ties(self):
        bundle, manifest = self.trained(constant=True)
        result = self.module().predict(bundle, self.holdout())
        for name, policy in result["policies"].items():
            with self.subTest(name=name):
                np.testing.assert_array_equal(policy["probabilities"], [[1, 0, 0, 0], [1, 0, 0, 0]])
                self.assertEqual(policy["firing_rate"], 0)
                self.assertTrue(np.isfinite(policy["predicted_values"]).all())

    def test_identical_observable_text_does_not_fit_degenerate_svd(self):
        rows = self.rows(8)
        for row in rows:
            row["prefix_only"]["recent_text"] = "room door"
        with warnings.catch_warnings(record=True) as observed:
            warnings.simplefilter("always")
            transform = self.module().Features(True).fit(rows)
            matrix = transform.transform(self.holdout())
        self.assertTrue(np.isfinite(matrix).all())
        self.assertFalse([w for w in observed if issubclass(w.category, RuntimeWarning)])

    def test_rate_random_uses_fixed_oof_rate_and_expected_weights(self):
        bundle, manifest = self.trained()
        result = self.module().predict(bundle, self.holdout())
        random = result["policies"]["RATE_RANDOM"]
        np.testing.assert_allclose(random["probabilities"], [[.5, 1 / 6, 1 / 6, 1 / 6]] * 2)
        self.assertEqual(random["firing_rate"], .5)
        repeated = self.holdout()[1:]
        self.assertEqual(self.module().predict(bundle, repeated)["policies"]["RATE_RANDOM"]["firing_rate"], .5)
        self.assertEqual(result["policies"]["BEST_FIXED"]["probabilities"], [[1, 0, 0, 0]] * 2)

    def test_round_one_cannot_tune_or_train_single_round_zero(self):
        bundle, original_manifest = self.trained()
        changed = self.rows()
        for row in changed:
            row["Y"][1] = [0, 0, 1, 1]
        alternate, manifest = self.module().fit(changed)
        before = self.module().predict(bundle, self.holdout())["policies"]["SINGLE_0"]
        after = self.module().predict(alternate, self.holdout())["policies"]["SINGLE_0"]
        self.assertEqual(before, after)
        self.assertEqual(original_manifest["strata"][0]["policies"]["SINGLE_0"]["target_sha256"], manifest["strata"][0]["policies"]["SINGLE_0"]["target_sha256"])

    def test_half_data_records_exact_task_budget_and_constant_ties(self):
        bundle, manifest = self.trained()
        policies = manifest["strata"][0]["policies"]
        half = policies["REPEATED_HALF"]
        self.assertEqual(len(half["task_ids"]), 32)
        self.assertEqual(half["target_rounds"], [0, 1])
        self.assertEqual(half["round_task_count"], policies["SINGLE_0"]["round_task_count"])
        self.assertTrue(set(half["task_ids"]) < set(policies["REPEATED"]["task_ids"]))
        for fold in half["folds"]:
            self.assertTrue(set(fold["train_task_ids"] + fold["validation_task_ids"]) <= set(half["task_ids"]))

    def test_relabelled_development_tasks_and_unknown_strata_are_rejected(self):
        bundle, manifest = self.trained()
        renamed = self.rows(1)
        renamed[0]["split"] = "test"
        with self.assertRaises(ValueError):
            self.module().predict(bundle, renamed)
        unknown = self.holdout()
        unknown[0]["model"] = "untrained_actor"
        with self.assertRaises(ValueError):
            self.module().predict(bundle, unknown)
        with self.assertRaises(ValueError):
            self.module().fit(self.rows(7))

    def test_duplicate_task_blocks_cannot_silently_reweight_training(self):
        rows = self.rows(8, constant=True)
        with self.assertRaises(ValueError):
            self.module().fit(rows + [copy.deepcopy(rows[0])])

    def test_actor_and_environment_models_remain_separate_with_missing_classes(self):
        rows = self.rows(8, constant=True)
        science, other = copy.deepcopy(rows), copy.deepcopy(rows)
        for row in science:
            row.update(env="scienceworld", Y=[[0, 1, 0, 0]] * 2)
        for row in other:
            row.update(model="other_actor", Y=[[0, 0, 1, 0]] * 2)
        bundle, manifest = self.module().fit(rows + science + other)
        prediction_rows = [self.holdout()[0] for _ in range(3)]
        prediction_rows[1]["env"] = "scienceworld"
        prediction_rows[2]["model"] = "other_actor"
        result = self.module().predict(bundle, prediction_rows)
        for method in ["BEST_FIXED", "REPEATED", "PAIRWISE", "RF_LCB"]:
            np.testing.assert_array_equal(result["policies"][method]["probabilities"], [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
        self.assertEqual(len(manifest["strata"]), 3)

    def test_serialized_cli_ignores_labels_and_refuses_overwrite(self):
        bundle, manifest = self.trained(constant=True)
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            development = directory / "dev.json"
            development.write_text(json.dumps(self.rows(8, constant=True)))
            subprocess.run([sys.executable, str(ROOT / "policies.py"), "fit", "--rows", str(development), "--out", str(directory / "model")], check=True, capture_output=True, text=True)
            with self.assertRaises(FileExistsError):
                self.module().save(bundle, manifest, directory / "model")
            rows = self.holdout()
            source = directory / "rows.json"
            source.write_text(json.dumps(rows))
            output = directory / "prediction.json"
            command = [sys.executable, str(ROOT / "policies.py"), "predict", "--rows", str(source), "--model", str(directory / "model"), "--out", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            first = json.loads(output.read_text())
            self.assertEqual(first, self.module().predict(bundle, rows))
            for row in rows:
                row.pop("Y")
            source.write_text(json.dumps(rows))
            output.unlink()
            subprocess.run(command, check=True, capture_output=True, text=True)
            self.assertEqual(first, json.loads(output.read_text()))
            receipt = json.loads((directory / "model/manifest.json").read_text())
            self.assertEqual(len(receipt["bundle_sha256"]), 64)


if __name__ == "__main__":
    unittest.main()
