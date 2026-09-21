import argparse
import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path


TASK_TYPES = [
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
]


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def strings(value):
    if isinstance(value, dict):
        for item in value.values():
            yield from strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from strings(item)
    elif isinstance(value, str):
        yield value


def task_specs(value):
    if isinstance(value, dict):
        if isinstance(value.get("task_spec"), dict):
            yield value["task_spec"]
        for item in value.values():
            yield from task_specs(item)
    elif isinstance(value, list):
        for item in value:
            yield from task_specs(item)


def exposure(paths: list[Path]) -> tuple[set[str], set[tuple[str, int]], list[dict]]:
    goals, science, records = set(), set(), []
    for path in paths:
        content = path.read_bytes()
        value = json.loads(content)
        for text in strings(value):
            if text.endswith("/game.tw-pddl"):
                goals.add(Path(text).parts[-3])
        for spec in task_specs(value):
            if "task_name" in spec and "variation" in spec:
                science.add((spec["task_name"], int(spec["variation"])))
        records.append({"path": str(path), "sha256": hashlib.sha256(content).hexdigest()})
    return goals, science, records


def alf_candidates(data_root: Path, excluded_goals: set[str], count, seed: int, split: str = "holdout", source: str = "train") -> list[dict]:
    by_type = defaultdict(list)
    root = data_root / "json_2.1.1" / source
    for path in sorted(root.glob("*/*/game.tw-pddl")):
        goal = path.parts[-3]
        if goal in excluded_goals or "movable" in goal or "Sliced" in goal:
            continue
        try:
            if not json.loads(path.read_text()).get("solvable", False):
                continue
            task_type = json.loads((path.parent / "traj_data.json").read_text())["task_type"]
        except (OSError, ValueError, KeyError):
            continue
        if task_type not in TASK_TYPES:
            continue
        relative = path.relative_to(root)
        floorplan = re.search(r"-(\d+)$", goal)
        if floorplan is None:
            continue
        identity = {"game": "/".join(relative.parts[:2])}
        task_id = digest({"env": "alfworld", "identity": identity})[:16]
        by_type[task_type].append({
            "task_id": task_id,
            "split": split,
            "env": "alfworld",
            "task_spec": {"game_file": str(path), "task_type": task_type},
            "identity": identity,
            "source_split": source,
            "group_id": f"alf-floorplan-{floorplan.group(1)}",
        })
    if isinstance(count, dict):
        quota = {task_type: int(count.get(task_type, 0)) for task_type in TASK_TYPES}
    else:
        if count % len(TASK_TYPES):
            raise ValueError("ALFWorld count must divide evenly across task types")
        quota = dict.fromkeys(TASK_TYPES, count // len(TASK_TYPES))
    selected = []
    for task_type in TASK_TYPES:
        ranked = sorted(by_type[task_type], key=lambda row: digest([seed, task_type, row["identity"]]))
        if len(ranked) < quota[task_type]:
            raise ValueError(f"ALFWorld shortfall for {task_type}: {len(ranked)} < {quota[task_type]}")
        selected.extend(ranked[:quota[task_type]])
    return selected


def science_candidates(reserved: dict, excluded: set[tuple[str, int]], count: int, seed: int, split: str = "holdout") -> list[dict]:
    candidates = []
    for source in reserved["tasks"]:
        if source.get("env") != "scienceworld":
            continue
        spec = source["task_spec"]
        key = (spec["task_name"], int(spec["variation"]))
        if key in excluded:
            continue
        task = {**source, "split": split, "group_id": f"science-family-{spec['task_name']}"}
        task["source_split"] = "test"
        candidates.append(task)
    ranked = sorted(candidates, key=lambda row: digest([seed, row["task_spec"]]))
    if len(ranked) < count:
        raise ValueError(f"ScienceWorld shortfall: {len(ranked)} < {count}")
    return ranked[:count]


def finalize(tasks: list[dict], seed: int, shards: int) -> list[dict]:
    ordered = sorted(tasks, key=lambda row: digest([seed, row["env"], row["task_id"]]))
    for index, task in enumerate(ordered):
        choice = int(digest([task["task_id"], "checkpoint"])[0], 16) % 2
        task["checkpoint_step"] = ((4, 8) if task["env"] == "alfworld" else (8, 16))[choice]
        task["seed"] = int(digest([seed, task["task_id"], "actor_seed"])[:8], 16) % 2147483647
        task["worker_shard"] = index % shards
    random.Random(seed).shuffle(ordered)
    if len({task["task_id"] for task in ordered}) != len(ordered):
        raise ValueError("Duplicate task IDs")
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--prior-reserved-frame", type=Path, required=True)
    parser.add_argument("--exclude-config", type=Path, action="append", default=[])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--alfworld", type=int, default=384)
    parser.add_argument("--alfworld-per-type", type=json.loads, default=None)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--source-split", default="train")
    parser.add_argument("--scienceworld", type=int, default=57)
    parser.add_argument("--shards", type=int, default=8)
    parser.add_argument("--seed", type=int, default=270914)
    args = parser.parse_args()
    if args.out.exists() or args.shards < 1:
        raise ValueError("Output must be new and shard count positive")
    excluded_goals, excluded_science, exposure_records = exposure(args.exclude_config)
    reserved_bytes = args.prior_reserved_frame.read_bytes()
    reserved = json.loads(reserved_bytes)
    tasks = finalize(
        alf_candidates(args.data_root, excluded_goals, args.alfworld_per_type or args.alfworld, args.seed, args.split, args.source_split)
        + science_candidates(reserved, excluded_science, args.scienceworld, args.seed, args.split),
        args.seed,
        args.shards,
    )
    template_bytes = args.template.read_bytes()
    template = json.loads(template_bytes)
    config = {key: template[key] for key in ["schema_version", "model", "model_path", "model_revision", "temperature", "scaffold", "max_tokens", "system_prompt", "arms", "datasets"]}
    config.update(phase=args.split, rounds=2, tasks=tasks, worker_shards=args.shards, sampling={
        "seed": args.seed,
        "counts": {"alfworld": args.alfworld_per_type or args.alfworld, "scienceworld": args.scienceworld},
        "task_set_sha256": digest(tasks),
        "exclusion_scope": "Prior configured or recorded task identities and ALFWorld goal directories from the supplied exposure packet",
    })
    args.out.mkdir(parents=True)
    master = args.out / "panel-master.json"
    master.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    shard_records = []
    for shard in range(args.shards):
        shard_config = {**config, "worker_shard": shard, "tasks": [task for task in tasks if task["worker_shard"] == shard]}
        path = args.out / f"panel-shard{shard}.json"
        path.write_text(json.dumps(shard_config, indent=2, sort_keys=True) + "\n")
        shard_records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "tasks": len(shard_config["tasks"])})
    manifest = {
        "status": "FROZEN_BEFORE_BASELINE_OR_ARM_GENERATION",
        "seed": args.seed,
        "counts": {"alfworld": args.alfworld_per_type or args.alfworld, "scienceworld": args.scienceworld},
        "tasks_sha256": digest(tasks),
        "groups": {env: len({task["group_id"] for task in tasks if task["env"] == env}) for env in ["alfworld", "scienceworld"]},
        "template_sha256": hashlib.sha256(template_bytes).hexdigest(),
        "prior_reserved_frame_sha256": hashlib.sha256(reserved_bytes).hexdigest(),
        "exposure_inputs": exposure_records,
        "excluded_alfworld_goals": len(excluded_goals),
        "excluded_scienceworld_variations": len(excluded_science),
        "shards": shard_records,
    }
    (args.out / "panel-freeze.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
