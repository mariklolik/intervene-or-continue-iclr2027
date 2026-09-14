import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from analysis import ARMS
from policies import digest

TASK_KEYS = ["task_id", "env", "split"]
GRID = [(r, a) for r in range(2) for a in ARMS]
EARLY = {"terminal_before_checkpoint": False, "not_started": None, "baseline_incomplete_or_invalid": None}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path, role):
    content = Path(path).read_bytes()
    value = json.loads(content)
    require(isinstance(value, dict), f"Expected JSON object: {path}")
    return value, {"path": str(path), "sha256": hashlib.sha256(content).hexdigest(), "role": role}


def validate_task(task):
    state, eligible = task["state"], task["checkpoint_eligible"]
    cells, reasons, missing = task["cells"], task["reasons"], task["missing_cells"]
    require(isinstance(cells, list) and isinstance(reasons, list) and isinstance(missing, list), "Invalid task audit lists")
    names = {f"round{r}-{a}.json" for r, a in GRID}
    require(len(missing) == len(set(missing)) and set(missing) <= names, "Invalid missing-cell names")
    require(type(task["valid_cells"]) is int, "Valid-cell count must be an integer")
    if state in EARLY:
        require(eligible is EARLY[state] and not cells and task["valid_cells"] == 0, "Invalid early/unknown eligibility or cells")
        require(not reasons if state == "terminal_before_checkpoint" else bool(reasons), "Invalid early/unknown reasons")
        if state == "not_started":
            require(reasons == ["baseline.json:missing"], "Invalid not-started reason")
        return
    require(state in {"complete", "incomplete_or_invalid"} and eligible is True, "Invalid state/eligibility combination")
    require([(c["round"], c["arm"]) for c in cells] == GRID, "Eligible task requires the exact ordered eight-cell grid")
    expected_reasons, expected_missing = [], []
    for cell in cells:
        name = f"round{cell['round']}-{cell['arm']}.json"
        require(cell["path"] == f"{task['task_id']}/{name}", "Cell task/path mismatch")
        require(type(cell["present"]) is bool and type(cell["valid"]) is bool and isinstance(cell["reasons"], list), "Invalid cell flags")
        require(cell["valid"] == (not cell["reasons"]) and (cell["present"] or not cell["valid"]), "Inconsistent cell validity")
        expected_reasons.extend(f"{name}:{reason}" for reason in cell["reasons"])
        if not cell["present"]:
            require(cell["reasons"] == ["missing"], "Absent cell must have the missing reason")
            expected_missing.append(name)
    count = sum(c["valid"] for c in cells)
    require(task["valid_cells"] == count and reasons == expected_reasons and missing == expected_missing, "Task/cell summary mismatch")
    require((state == "complete") == (count == 8 and not reasons), "Incorrect complete/incomplete state")


def validate_frame(master_paths, shard_paths, context_receipt_path, scope_receipt_path, audit_paths):
    require(len(master_paths) == 2 and len(shard_paths) == 4 and len(audit_paths) == 4, "Exactly two masters, four shards and four audits required")
    all_paths = list(master_paths) + list(shard_paths) + [context_receipt_path, scope_receipt_path] + list(audit_paths)
    require(len({Path(p).resolve() for p in all_paths}) == 12, "Duplicate input paths")
    context, context_input = read(context_receipt_path, "context_receipt")
    scope, scope_input = read(scope_receipt_path, "scope_receipt")
    config_paths = list(master_paths) + list(shard_paths)
    names = [Path(p).name for p in config_paths]
    require(len(set(names)) == 6 and set(names) == set(context["configs"]) == set(scope["configs"]), "Exact six frozen config names required")
    inputs, configs = [context_input, scope_input], []
    for index, path in enumerate(config_paths):
        config, source = read(path, "master" if index < 2 else "shard")
        name = Path(path).name
        frozen, original = context["configs"][name], scope["configs"][name]
        require(source["sha256"] == frozen["sha256"], f"Qualified config hash mismatch: {name}")
        require(frozen["parent_sha256"] == original["sha256"] == config["context_repair_parent_config_sha256"], "Scope/context parent hash mismatch")
        require(frozen["tasks_unchanged"] is True and config["runtime_contract"]["context_length"] == context["context_length"], "Context task-preservation contract mismatch")
        tasks = config["tasks"]
        ids = [t["task_id"] for t in tasks]
        require(tasks and len(ids) == len(set(ids)) and ids == original["task_ids"], "Empty, duplicate, changed or reordered frozen tasks")
        require(len(tasks) == frozen["task_count"] and dict(Counter(t["env"] for t in tasks)) == original["counts"], "Frozen task counts mismatch")
        require(config["phase"] == "test" and config["rounds"] == 2 and set(config["arms"]) == set(ARMS) and config["arms"]["A0"] in (None, ""), "Unexpected final design")
        require(config.get("model_revision") and config.get("scaffold") and config["worker_shards"] == 2, "Missing qualified model/scaffold/shard metadata")
        require(all(t["split"] == "test" and t["env"] in {"alfworld", "scienceworld"} and t["worker_shard"] in {0, 1} for t in tasks), "Changed task split/environment/shard")
        for task in tasks:
            require(all(k in task for k in ["identity", "task_spec", "seed", "checkpoint_step"]), "Incomplete task identity/specification")
        inputs.append(source)
        configs.append((config, source))
    masters, shards = configs[:2], configs[2:]
    require(len({c["model"] for c, _ in masters}) == 2 and all(c.get("worker_shard") is None for c, _ in masters), "Two distinct actor masters required")
    require(masters[0][0]["tasks"] == masters[1][0]["tasks"], "Actor master task frames differ")
    selected_specs = scope["selected_scienceworld_specs"]
    master_specs = [t["task_spec"] for t in masters[0][0]["tasks"] if t["env"] == "scienceworld"]
    require(Counter(map(digest, master_specs)) == Counter(map(digest, selected_specs)), "Selected ScienceWorld specifications changed")
    expected, by_shard = [], {}
    for master, master_source in masters:
        children = [(c, s) for c, s in shards if c["model"] == master["model"]]
        require(len(children) == 2 and {c["worker_shard"] for c, _ in children} == {0, 1}, "Missing or duplicate actor shard")
        for child, source in children:
            subset = [t for t in master["tasks"] if t["worker_shard"] == child["worker_shard"]]
            require(child["tasks"] == subset and subset, "Shard union/spec/order differs from master")
            excluded = {"tasks", "worker_shard", "context_repair_parent_config_sha256"}
            require({k: v for k, v in child.items() if k not in excluded} == {k: v for k, v in master.items() if k not in excluded}, "Master/shard configuration differs")
            by_shard[source["sha256"]] = (child, source)
        for task in master["tasks"]:
            source = next(s for c, s in children if c["worker_shard"] == task["worker_shard"])
            expected.append({"model": master["model"], **{k: task[k] for k in TASK_KEYS}, "config_sha256": source["sha256"], "master_config_sha256": master_source["sha256"], "task_sha256": digest(task)})
    require(len(by_shard) == 4, "Shard configuration hashes must be unique")
    audited, seen, audit_receipts = {}, set(), []
    for path in audit_paths:
        value, source = read(path, "audit")
        audit = value.get("ingestion", value)
        config_hash = audit["config_sha256"]
        require(config_hash in by_shard and config_hash not in seen, "Missing, duplicate or unfrozen audit shard")
        seen.add(config_hash)
        config, config_source = by_shard[config_hash]
        require(all(audit[k] == config[k] for k in ["model", "model_revision", "phase"]), "Audit model/revision/phase mismatch")
        tasks = audit["tasks"]
        require([{k: t[k] for k in TASK_KEYS} for t in tasks] == [{k: t[k] for k in TASK_KEYS} for t in config["tasks"]], "Audit planned identities/order differ from shard")
        require(audit["planned_tasks"] == len(tasks), "Audit planned count mismatch")
        for task in tasks:
            validate_task(task)
            key = (config["model"], task["env"], task["task_id"])
            require(key not in audited, "Duplicate audited task identity")
            audited[key] = {k: task[k] for k in ["task_id", "split", "state", "checkpoint_eligible"]} | {"config_sha256": config_hash}
        accepted = sum(t["state"] == "complete" for t in tasks)
        require(audit["accepted_tasks"] == accepted, "Audit accepted count mismatch")
        if "arm_denominators" in audit:
            for arm in ARMS:
                cells = [c for t in tasks for c in t["cells"] if c["arm"] == arm]
                expected_counts = {"potential_cells": 2 * len(tasks), "confirmed_eligible_cells": 2 * sum(t["checkpoint_eligible"] is True for t in tasks), "valid_cells": sum(c["valid"] for c in cells), "present_cells_at_eligible_checkpoints": sum(c["present"] for c in cells), "accepted_block_cells": 2 * accepted}
                require(all(audit["arm_denominators"][arm][k] == v for k, v in expected_counts.items()), "Audit arm denominator mismatch")
        inputs.append(source)
        audit_receipts.append({k: source[k] for k in ["path", "sha256"]} | {"model": config["model"], "config_sha256": config_hash, "planned_tasks": len(tasks), "accepted_tasks": accepted, "state_counts": dict(Counter(t["state"] for t in tasks))})
    require(seen == set(by_shard), "Every frozen shard must have its audit")
    require(set(audited) == {(t["model"], t["env"], t["task_id"]) for t in expected}, "Audit/master frame mismatch")
    strata = []
    for model, env in sorted({(t["model"], t["env"]) for t in expected}):
        tasks = [audited[(model, env, t["task_id"])] for t in expected if (t["model"], t["env"]) == (model, env)]
        strata.append({"model": model, "env": env, "n_tasks": len(tasks), "tasks": tasks})
    return {"schema_version": 1, "status": "PASS", "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "inputs": inputs, "audits": audit_receipts, "strata": strata, "expected_frame": expected, "scope": "Complete accounting of the caller-supplied frozen task frame; missing/failed/early tasks retained; not a task completion or model qualification certificate"}


def main():
    parser = argparse.ArgumentParser()
    for name, count in [("masters", 2), ("shards", 4), ("audits", 4)]:
        parser.add_argument(f"--{name}", nargs=count, type=Path, required=True)
    for name in ["context-receipt", "scope-receipt", "out"]:
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    result = validate_frame(args.masters, args.shards, args.context_receipt, args.scope_receipt, args.audits)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        stream.write(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
