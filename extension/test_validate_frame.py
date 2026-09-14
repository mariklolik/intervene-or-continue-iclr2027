import copy
import hashlib
import importlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
ARMS = ["A0", "A1", "A2", "A3"]


class FrameTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.masters, self.shards, self.audits = [], [], []
        self.context = {"context_length": 32768, "configs": {}}
        self.scope = {"configs": {}, "selected_scienceworld_specs": [{"task_name": "boil", "variation": 1}]}
        tasks = [{"task_id": f"t{i}", "env": "scienceworld" if i == 1 else "alfworld", "split": "test", "task_spec": {"task_name": "boil", "variation": 1} if i == 1 else {"game_file": f"game{i}"}, "identity": {"id": i}, "seed": i + 1, "checkpoint_step": 4, "worker_shard": i % 2} for i in range(6)]
        for actor in ["actor1", "actor2"]:
            master = {"model": actor, "model_revision": actor + "revision", "phase": "test", "scaffold": "grounded-react-v1", "rounds": 2, "arms": {a: "" if a == "A0" else a for a in ARMS}, "runtime_contract": {"context_length": 32768}, "worker_shard": None, "worker_shards": 2, "tasks": tasks}
            self.masters.append(self.config(master, f"main-test-{actor}.json"))
            for shard in range(2):
                config = {**master, "worker_shard": shard, "tasks": [t for t in tasks if t["worker_shard"] == shard]}
                path = self.config(config, f"main-test-{actor}-shard{shard}.json")
                self.shards.append(path)
                audit_tasks = [self.task(t, ["complete", "terminal_before_checkpoint", "incomplete_or_invalid", "not_started", "baseline_incomplete_or_invalid", "complete"][int(t["task_id"][1:])]) for t in config["tasks"]]
                audit = {"model": actor, "model_revision": master["model_revision"], "phase": "test", "config_sha256": self.sha(path), "planned_tasks": len(audit_tasks), "accepted_tasks": sum(t["state"] == "complete" for t in audit_tasks), "tasks": audit_tasks}
                self.audits.append(self.write(f"audit-{actor}-{shard}.json", {"ingestion": audit}))
        self.context_path = self.write("context.json", self.context)
        self.scope_path = self.write("scope.json", self.scope)

    def sha(self, path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value, sort_keys=True))
        return path

    def config(self, config, name):
        parent = hashlib.sha256(name.encode()).hexdigest()
        config = {**config, "context_repair_parent_config_sha256": parent}
        path = self.write(name, config)
        counts = {env: sum(t["env"] == env for t in config["tasks"]) for env in {t["env"] for t in config["tasks"]}}
        self.context["configs"][name] = {"sha256": self.sha(path), "parent_sha256": parent, "tasks_unchanged": True, "task_count": len(config["tasks"])}
        self.scope["configs"][name] = {"sha256": parent, "task_ids": [t["task_id"] for t in config["tasks"]], "counts": counts}
        return path

    def task(self, task, state):
        eligible = True if state in {"complete", "incomplete_or_invalid"} else False if state == "terminal_before_checkpoint" else None
        cells = [{"round": r, "arm": a, "path": f"{task['task_id']}/round{r}-{a}.json", "present": True, "valid": True, "reasons": []} for r in range(2) for a in ARMS] if eligible is True else []
        reasons = []
        if state == "incomplete_or_invalid":
            cells[-1].update(valid=False, reasons=["failure"])
            reasons = ["round1-A3.json:failure"]
        if state == "not_started":
            reasons = ["baseline.json:missing"]
        if state == "baseline_incomplete_or_invalid":
            reasons = ["baseline_failure_or_suspension_or_provenance"]
        return {k: task[k] for k in ["task_id", "env", "split"]} | {"state": state, "checkpoint_eligible": eligible, "valid_cells": sum(c["valid"] for c in cells), "cells": cells, "reasons": reasons, "missing_cells": []}

    def module(self):
        self.assertTrue((ROOT / "validate_frame.py").exists(), "Exact final-frame validator is missing")
        return importlib.import_module("validate_frame")

    def validate(self, **kwargs):
        args = {"master_paths": self.masters, "shard_paths": self.shards, "context_receipt_path": self.context_path, "scope_receipt_path": self.scope_path, "audit_paths": self.audits}
        return self.module().validate_frame(**(args | kwargs))

    def mutate(self, path, change, refresh=False):
        value = json.loads(path.read_text())
        change(value)
        self.write(path.name, value)
        if refresh:
            self.context["configs"][path.name]["sha256"] = self.sha(path)
            self.write("context.json", self.context)

    def continue_value(self, value):
        for path in self.masters + self.shards:
            self.mutate(path, lambda c: c["arms"].update(A0=value), refresh=True)
        for config, audit in zip(self.shards, self.audits):
            self.mutate(audit, lambda a: a["ingestion"].update(config_sha256=self.sha(config)))

    def test_null_continue_matches_the_qualified_configuration_schema(self):
        self.continue_value(None)
        try:
            result = self.validate()
        except ValueError as error:
            self.fail(f"Valid JSON null no-intervention configuration rejected: {error}")
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["expected_frame"]), 12)

    def test_nonempty_and_other_false_valued_continue_types_are_rejected(self):
        for value in ["WARNING", False, 0, [], {}]:
            with self.subTest(value=value):
                self.continue_value(value)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_valid_mixed_states_preserve_every_planned_identity(self):
        result = self.validate()
        self.assertEqual(result["status"], "PASS")
        self.assertEqual(len(result["expected_frame"]), 12)
        self.assertEqual(len(result["audits"]), 4)
        self.assertEqual(sum(s["n_tasks"] for s in result["strata"]), 12)
        self.assertEqual({t["state"] for s in result["strata"] for t in s["tasks"]}, {"complete", "terminal_before_checkpoint", "incomplete_or_invalid", "not_started", "baseline_incomplete_or_invalid"})
        self.assertTrue(all(len(t["task_sha256"]) == 64 for t in result["expected_frame"]))

    def test_omitted_entire_audit_or_shard_is_rejected(self):
        for kwargs in [{"audit_paths": self.audits[:-1]}, {"shard_paths": self.shards[:-1]}, {"master_paths": self.masters[:-1]}, {"audit_paths": self.audits[:3] + self.audits[:1]}]:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.validate(**kwargs)

    def test_altered_master_bytes_and_parent_hash_are_rejected(self):
        self.mutate(self.masters[0], lambda c: c.update(scaffold="changed"))
        with self.assertRaises(ValueError):
            self.validate()
        self.mutate(self.masters[0], lambda c: c.update(context_repair_parent_config_sha256="0" * 64), refresh=True)
        with self.assertRaises(ValueError):
            self.validate()

    def test_changed_shard_task_spec_is_rejected_even_if_config_receipt_rehashed(self):
        self.mutate(self.shards[0], lambda c: c["tasks"][0]["task_spec"].update(game_file="replacement"), refresh=True)
        with self.assertRaises(ValueError):
            self.validate()

    def test_empty_duplicate_extra_or_reordered_shard_tasks_are_rejected(self):
        original = self.shards[0].read_text()
        changes = [lambda c: c.update(tasks=[]), lambda c: c["tasks"].append(copy.deepcopy(c["tasks"][0])), lambda c: c["tasks"][0].update(task_id="extra"), lambda c: c["tasks"].reverse(), lambda c: c["tasks"][0].update(split="dev")]
        for change in changes:
            with self.subTest(change=change):
                self.shards[0].write_text(original)
                self.mutate(self.shards[0], change, refresh=True)
                with self.assertRaises(ValueError):
                    self.validate()

    def test_incorrect_audit_hash_model_order_or_task_set_is_rejected(self):
        original = self.audits[0].read_text()
        changes = [lambda a: a.update(config_sha256="0" * 64), lambda a: a.update(model="wrong"), lambda a: a["tasks"].reverse(), lambda a: a["tasks"].pop(), lambda a: a["tasks"].append(copy.deepcopy(a["tasks"][0])), lambda a: a["tasks"][0].update(task_id="extra")]
        for change in changes:
            with self.subTest(change=change):
                self.audits[0].write_text(original)
                self.mutate(self.audits[0], lambda value: change(value["ingestion"]))
                with self.assertRaises(ValueError):
                    self.validate()

    def test_incorrect_state_eligibility_and_cell_combinations_are_rejected(self):
        original = self.audits[0].read_text()
        changes = [lambda t: t.update(checkpoint_eligible=None), lambda t: t.update(checkpoint_eligible=1), lambda t: t.update(state="outside_requested_split"), lambda t: t.update(state="incomplete_or_invalid"), lambda t: t.update(valid_cells=7), lambda t: t["cells"][0].update(present=False), lambda t: t.update(missing_cells=["round0-A0.json"])]
        for change in changes:
            with self.subTest(change=change):
                self.audits[0].write_text(original)
                self.mutate(self.audits[0], lambda value: change(value["ingestion"]["tasks"][0]))
                with self.assertRaises(ValueError):
                    self.validate()

    def test_early_and_unknown_states_cannot_be_relabelled_eligible(self):
        for path, index in [(self.audits[1], 0), (self.audits[1], 1), (self.audits[0], 2)]:
            original = path.read_text()
            self.mutate(path, lambda v: v["ingestion"]["tasks"][index].update(checkpoint_eligible=True))
            with self.assertRaises(ValueError):
                self.validate()
            path.write_text(original)

    def test_legitimate_missing_cell_stays_in_the_eligible_frame(self):
        value = json.loads(self.audits[0].read_text())
        task = value["ingestion"]["tasks"][1]
        task["cells"][-1].update(present=False, reasons=["missing"])
        task.update(reasons=["round1-A3.json:missing"], missing_cells=["round1-A3.json"])
        self.write(self.audits[0].name, value)
        result = self.validate()
        found = next(t for s in result["strata"] if s["model"] == "actor1" for t in s["tasks"] if t["task_id"] == "t2")
        self.assertEqual(found["state"], "incomplete_or_invalid")
        self.assertIs(found["checkpoint_eligible"], True)

    def test_scope_specification_and_summary_count_mismatches_are_rejected(self):
        self.mutate(self.scope_path, lambda s: s["selected_scienceworld_specs"][0].update(variation=2))
        with self.assertRaises(ValueError):
            self.validate()
        self.write("scope.json", self.scope)
        self.mutate(self.audits[0], lambda a: a["ingestion"].update(accepted_tasks=0))
        with self.assertRaises(ValueError):
            self.validate()

    def test_cli_emits_hashed_receipt_and_refuses_overwrite(self):
        self.module()
        output = self.root / "receipt.json"
        command = [sys.executable, str(ROOT / "validate_frame.py"), "--masters", *map(str, self.masters), "--shards", *map(str, self.shards), "--context-receipt", str(self.context_path), "--scope-receipt", str(self.scope_path), "--audits", *map(str, self.audits), "--out", str(output)]
        first = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        receipt = json.loads(output.read_text())
        self.assertEqual(receipt["source_sha256"], self.sha(ROOT / "validate_frame.py"))
        self.assertEqual(len(receipt["inputs"]), 12)
        self.assertNotEqual(subprocess.run(command, capture_output=True, text=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
