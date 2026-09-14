import copy
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class MakeConfigsTest(unittest.TestCase):
    def setUp(self):
        self.assertIsNotNone(importlib.util.find_spec("make_configs"))
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.data = self.root / "alfworld"
        for split in ("valid_seen", "valid_unseen"):
            for family in ("pick_and_place_simple", "look_at_obj_in_light"):
                for variation in range(4):
                    path = self.data / "json_2.1.1" / split / family / f"trial_{split}_{variation}"
                    path.mkdir(parents=True)
                    (path / "game.tw-pddl").write_text('{"solvable":true}')
                    (path / "traj_data.json").write_text(json.dumps({"task_type": family}))
        self.models = ("Qwen3-8B", "Qwen2.5-7B-Instruct")
        self.bases = []
        for model in self.models:
            self.bases.append(self.write(model + ".json", {
                "phase": "pilot-deliberation-final-interface", "model": model,
                "temperature": 0.7, "rounds": 0, "arms": {},
                "tasks": [{"task_id": "pilot-must-not-survive"}],
                "scaffold": "grounded-react-v1", "max_tokens": 256,
                "system_prompt": "THOUGHT then ACTION",
            }))
        self.arms = self.write("arms.json", {
            "phase": "pilot-exploratory", "model": self.models[0],
            "temperature": 0.7, "rounds": 2, "tasks": [],
            "arms": {"A0": None, "A1": "check", "A2": "replan", "A3": "rollback"},
        })
        self.readiness = self.write("readiness.json", {
            "models": {"Qwen/" + model: {"path": "/cache/snapshots/" + digit * 40,
                "all_indexed_weights_present": True} for model, digit in zip(self.models, "ab")},
            "sglang": {"image": "registry/sglang@sha256:" + "c" * 64},
            "cpu": {"packages": {"alfworld": "0.4.2", "textworld": "1.7.0"}},
            "assets": [{"path": "/data/json_2.1.2_tw-pddl.zip", "sha256": "d" * 64,
                "source": "https://github.com/alfworld/alfworld/releases/download/0.4.0/json_2.1.2_tw-pddl.zip"}],
        })
        self.catalog = {family: {
            "ids": {"train": [0, 1], "dev": [10, 11, 12], "test": [20, 21, 22]},
            "counts": {"train": 2, "dev": 3, "test": 3},
            "excluded_by_legacy_sampler": family == "grow-plant",
        } for family in ("boil", "melt", "grow-plant")}
        self.scienceworld = self.write("scienceworld.json", {
            "scienceworld_version": "1.2.3", "catalog": self.catalog,
            "source_sha256": {"/scienceworld/scienceworld.jar": "e" * 64},
            "probe": {"passed": True},
        })
        self.counts = {phase: {"alfworld": 5, "scienceworld": 5} for phase in ("dev", "test")}

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def build(self, counts=None):
        from make_configs import build_configs

        return build_configs(self.bases, self.arms, self.readiness, self.scienceworld,
                             self.data, counts if counts is not None else self.counts)

    def test_official_splits_unique_tasks_actor_parity_and_balanced_shards(self):
        configs = self.build()
        self.assertEqual(len(configs), 12)
        phase_ids = {}
        for phase, source_split, variations in (("dev", "valid_seen", {10, 11, 12}),
                                                 ("test", "valid_unseen", {20, 21, 22})):
            full = configs[f"main-{phase}-Qwen3-8B.json"]
            self.assertEqual(full["tasks"], configs[f"main-{phase}-Qwen2.5-7B-Instruct.json"]["tasks"])
            self.assertEqual(len(full["tasks"]), 10)
            phase_ids[phase] = {task["task_id"] for task in full["tasks"]}
            self.assertEqual(len(phase_ids[phase]), 10)
            shards = [configs[f"main-{phase}-Qwen3-8B-shard{i}.json"]["tasks"] for i in range(2)]
            self.assertEqual([len(shard) for shard in shards], [5, 5])
            self.assertFalse({task["task_id"] for task in shards[0]} & {task["task_id"] for task in shards[1]})
            self.assertEqual({task["task_id"] for shard in shards for task in shard}, phase_ids[phase])
            for env in ("alfworld", "scienceworld"):
                sizes = [sum(task["env"] == env for task in shard) for shard in shards]
                self.assertEqual(sorted(sizes), [2, 3])
            for task in full["tasks"]:
                self.assertIn("identity", task)
                if task["env"] == "alfworld":
                    self.assertIn("/" + source_split + "/", task["task_spec"]["game_file"])
                    self.assertIn(task["checkpoint_step"], (4, 8))
                else:
                    self.assertIn(task["task_spec"]["variation"], variations)
                    self.assertNotEqual(task["task_spec"]["task_name"], "grow-plant")
                    self.assertIn(task["checkpoint_step"], (8, 16))
        self.assertFalse(phase_ids["dev"] & phase_ids["test"])

    def test_deterministic_seeds_frozen_interface_and_provenance(self):
        first = self.build()
        self.assertEqual(first, self.build())
        config = first["main-dev-Qwen3-8B.json"]
        self.assertEqual(config["model_revision"], "a" * 40)
        self.assertEqual(config["image"], "registry/sglang@sha256:" + "c" * 64)
        self.assertEqual(config["scaffold"], "grounded-react-v1")
        self.assertEqual(config["system_prompt"], "THOUGHT then ACTION")
        self.assertEqual(config["max_tokens"], 256)
        self.assertEqual(config["rounds"], 2)
        self.assertEqual(config["arms"], {"A0": None, "A1": "check", "A2": "replan", "A3": "rollback"})
        self.assertEqual(config["sampling"]["seed"], 260909)
        self.assertEqual(first["main-test-Qwen3-8B.json"]["sampling"]["seed"], 260910)
        self.assertEqual(config["provenance"]["readiness_sha256"], hashlib.sha256(self.readiness.read_bytes()).hexdigest())
        self.assertEqual(config["datasets"]["scienceworld"]["version"], "1.2.3")
        self.assertEqual(len({task["seed"] for task in config["tasks"]}), 10)
        reduced = self.build({phase: {"alfworld": 3, "scienceworld": 3} for phase in self.counts})
        previous = {task["task_id"]: task for task in config["tasks"]}
        for task in reduced["main-dev-Qwen3-8B.json"]["tasks"]:
            self.assertEqual(task["seed"], previous[task["task_id"]]["seed"])
            self.assertEqual(task["checkpoint_step"], previous[task["task_id"]]["checkpoint_step"])

    def test_shortfalls_invalid_counts_and_unknown_schema_fail(self):
        for env, count in (("alfworld", 9), ("scienceworld", 7), ("alfworld", -1), ("scienceworld", True)):
            with self.subTest(env=env, count=count):
                counts = copy.deepcopy(self.counts)
                counts["dev"][env] = count
                with self.assertRaises(ValueError):
                    self.build(counts)
        original = json.loads(self.bases[0].read_text())
        for change in ({"schema_version": 99}, {"scaffold": "unknown"}, {"new_unrecognized_field": True}, {"model": "unknown"}):
            with self.subTest(change=change):
                self.bases[0].write_text(json.dumps({**original, **change}))
                with self.assertRaises(ValueError):
                    self.build()
        self.bases[0].write_text(json.dumps(original))

    def test_catalog_overlap_duplicates_and_missing_split_fail(self):
        original = json.loads(self.scienceworld.read_text())
        for ids in ({"train": [0], "dev": [1], "test": [1]},
                    {"train": [0], "dev": [1, 1], "test": [2]},
                    {"train": [0], "test": [2]}):
            data = copy.deepcopy(original)
            data["catalog"]["boil"]["ids"] = ids
            self.scienceworld.write_text(json.dumps(data))
            with self.assertRaises(ValueError):
                self.build()

    def test_malformed_catalogue_and_missing_provenance_fail_explicitly(self):
        original = json.loads(self.scienceworld.read_text())
        for change in ({"catalog": {"boil": None}}, {"source_sha256": {"/scienceworld/scienceworld.jar": "unknown"}}):
            self.scienceworld.write_text(json.dumps({**original, **change}))
            with self.assertRaises(ValueError):
                self.build()

    def test_zero_environment_counts_and_cli_emit_only_fixture_configs(self):
        counts = {"dev": {"alfworld": 0, "scienceworld": 2},
                  "test": {"alfworld": 2, "scienceworld": 0}}
        configs = self.build(counts)
        self.assertEqual({task["env"] for task in configs["main-dev-Qwen3-8B.json"]["tasks"]}, {"scienceworld"})
        self.assertEqual({task["env"] for task in configs["main-test-Qwen3-8B.json"]["tasks"]}, {"alfworld"})
        output = self.root / "cli-output"
        command = [sys.executable, str(Path(__file__).with_name("make_configs.py")),
                   "--arms-config", str(self.arms), "--readiness", str(self.readiness),
                   "--scienceworld-readiness", str(self.scienceworld),
                   "--alfworld-data", str(self.data), "--out", str(output)]
        for base in self.bases:
            command.extend(["--base-config", str(base)])
        for phase, row in counts.items():
            for env, count in row.items():
                command.extend([f"--{phase}-{env}", str(count)])
        result = subprocess.run(command, text=True, capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["counts"], counts)
        self.assertEqual({path.name for path in output.iterdir()}, set(configs))
        for name, config in configs.items():
            self.assertEqual(json.loads((output / name).read_text()), config)

    def test_cross_split_same_alfworld_variant_fails(self):
        path = self.data / "json_2.1.1" / "valid_unseen"
        for trial in list(path.glob("*/*")):
            trial.rename(trial.with_name(trial.name.replace("valid_unseen", "valid_seen")))
        with self.assertRaisesRegex(ValueError, "overlap"):
            self.build()

    def test_frozen_outputs_cannot_be_overwritten(self):
        from make_configs import write_configs

        configs = self.build()
        output = self.root / "configs"
        write_configs(configs, output)
        before = {path.name: path.read_bytes() for path in output.iterdir()}
        with self.assertRaises(FileExistsError):
            write_configs(configs, output)
        self.assertEqual(before, {path.name: path.read_bytes() for path in output.iterdir()})


if __name__ == "__main__":
    unittest.main()
