import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from freeze_panel import digest


def retask(task: dict, rule: str, window: int) -> dict:
    task = {key: value for key, value in task.items() if key != "checkpoint_window"}
    task["checkpoint_rule"] = rule
    if rule == "event":
        task["checkpoint_window"] = [2, window]
    return task


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint-rule", required=True, choices=["scheduled", "event"])
    parser.add_argument("--event-window", type=int, default=24)
    args = parser.parse_args()
    master = json.loads((args.source / "panel-master.json").read_text())
    tasks = [retask(task, args.checkpoint_rule, args.event_window) for task in master["tasks"]]
    config = {**master, "tasks": tasks,
              "sampling": {**master["sampling"], "checkpoint_rule": args.checkpoint_rule, "task_set_sha256": digest(tasks)}}
    args.out.mkdir(parents=True)
    (args.out / "panel-master.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    records = []
    for shard in range(master["worker_shards"]):
        path = args.out / f"panel-shard{shard}.json"
        path.write_text(json.dumps({**config, "worker_shard": shard,
                                    "tasks": [task for task in tasks if task["worker_shard"] == shard]},
                                   indent=2, sort_keys=True) + "\n")
        records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (args.out / "panel-manifest.json").write_text(json.dumps({
        "status": "MIRRORED_BEFORE_BASELINE_OR_ARM_GENERATION",
        "source_panel": str(args.source), "source_task_set_sha256": master["sampling"]["task_set_sha256"],
        "checkpoint_rule": args.checkpoint_rule, "tasks_sha256": digest(tasks), "shards": records,
    }, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
