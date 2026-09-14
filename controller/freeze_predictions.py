import argparse
import hashlib
import json
import sys
from pathlib import Path

import joblib


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "extension"), str(ROOT / "controller"), str(ROOT)]
from analysis import prefix_metadata
from direct_advantage import predict as predict_direct
from policies import predict as predict_arm_outcome


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_selection(direct_manifest: dict, arm_manifest: dict, model: str, env: str) -> dict:
    direct = next(row for row in direct_manifest["strata"] if (row["model"], row["env"]) == (model, env))
    arm = next(row for row in arm_manifest["strata"] if (row["model"], row["env"]) == (model, env))
    strongest_name, strongest = max(arm["policies"].items(), key=lambda item: (round(item[1]["oof_utility"], 12), -item[1]["oof_firing_rate"], item[0]))
    fixed_index = arm["best_fixed_arm"]
    candidates = [
        {"name": "CONTINUE", "oof_utility": arm["mean_utility"][0], "oof_firing_rate": 0.0},
        {"name": "BEST_FIXED", "oof_utility": arm["mean_utility"][fixed_index], "oof_firing_rate": 1.0, "arm": fixed_index},
        {"name": "DIRECT_ADVANTAGE", **{key: direct["selected"][key] for key in ["oof_utility", "oof_firing_rate"]}},
        {"name": strongest_name, **{key: strongest[key] for key in ["oof_utility", "oof_firing_rate"]}},
    ]
    selected = max(candidates, key=lambda row: (round(row["oof_utility"], 12), -row["oof_firing_rate"], row["name"] == "CONTINUE", row["name"]))
    return {"candidates": candidates, "selected": selected, "strongest_matched_comparator": strongest_name, "arm_outcome_parameterization_control": "REPEATED"}


def load_prefixes(config_paths: list[Path], raw: Path) -> list[dict]:
    rows = []
    seen = set()
    for config_path in config_paths:
        content = config_path.read_bytes()
        config_hash = hashlib.sha256(content).hexdigest()
        config = json.loads(content)
        for task in config["tasks"]:
            key = (config["model"], task["env"], task["task_id"])
            if key in seen:
                raise ValueError("Duplicate task across shards")
            seen.add(key)
            path = raw / config_path.stem / task["task_id"] / "baseline.json"
            if not path.exists():
                raise ValueError(f"Missing baseline: {task['task_id']}")
            record = json.loads(path.read_text())
            expected = {"task_id": task["task_id"], "split": task["split"], "model": config["model"], "config_sha256": config_hash, "arm": "BASELINE", "round": -1, "seed": task["seed"]}
            if any(record.get(name) != value for name, value in expected.items()):
                raise ValueError(f"Baseline metadata mismatch: {task['task_id']}")
            episode = record["episode"]
            if episode.get("failure") or episode.get("suspended") or episode.get("task_spec") != task["task_spec"] or episode.get("env") != task["env"]:
                raise ValueError(f"Invalid baseline: {task['task_id']}")
            step = task["checkpoint_step"]
            if step < 1 or len(episode["steps"]) < step or episode["steps"][step - 1]["done"]:
                continue
            rows.append({
                "model": config["model"],
                "env": task["env"],
                "split": task["split"],
                "task_id": task["task_id"],
                "model_revision": config["model_revision"],
                "scaffold": config["scaffold"],
                "config_sha256": config_hash,
                "group_id": task["group_id"],
                "prefix_only": prefix_metadata(episode, step),
            })
    return sorted(rows, key=lambda row: (row["env"], row["task_id"]))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--direct-model", type=Path, required=True)
    parser.add_argument("--direct-manifest", type=Path, required=True)
    parser.add_argument("--arm-model", type=Path, required=True)
    parser.add_argument("--arm-manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    rows = load_prefixes(args.config, args.raw)
    direct_manifest = json.loads(args.direct_manifest.read_text())
    arm_manifest = json.loads(args.arm_manifest.read_text())
    direct = predict_direct(joblib.load(args.direct_model), rows)
    arm = predict_arm_outcome(joblib.load(args.arm_model), rows)
    if direct["rows"] != arm["rows"]:
        raise ValueError("Prediction row mismatch")
    policies = {
        "CONTINUE": arm["policies"]["CONTINUE"]["probabilities"],
        "BEST_FIXED": arm["policies"]["BEST_FIXED"]["probabilities"],
        "DIRECT_ADVANTAGE": direct["probabilities"],
        "ARM_OUTCOME": arm["policies"]["REPEATED"]["probabilities"],
    }
    selections = {}
    for env in sorted({row["env"] for row in rows}):
        selection = candidate_selection(direct_manifest, arm_manifest, rows[0]["model"], env)
        selections[env] = selection
        comparator = selection["strongest_matched_comparator"]
        policies[f"MATCHED_{env.upper()}"] = arm["policies"][comparator]["probabilities"]
    payload = {
        "rows": [{key: row[key] for key in ["model", "env", "split", "task_id", "config_sha256", "group_id"]} for row in rows],
        "policies": policies,
        "selection": selections,
        "training_data_sha256": direct["training_data_sha256"],
    }
    args.out.mkdir(parents=True)
    (args.out / "prefixes.json").write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n")
    (args.out / "predictions.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    paths = args.config + [args.direct_model, args.direct_manifest, args.arm_model, args.arm_manifest, Path(__file__), ROOT / "controller/direct_advantage.py", ROOT / "extension/policies.py"]
    freeze = {
        "status": "FROZEN_AFTER_BASELINES_BEFORE_ANY_ARM_OUTCOME_GENERATION",
        "eligible_prefixes": len(rows),
        "policy_set": sorted(policies),
        "selection": selections,
        "inputs": [{"path": str(path), "sha256": file_hash(path)} for path in paths],
        "prefixes_sha256": file_hash(args.out / "prefixes.json"),
        "predictions_sha256": file_hash(args.out / "predictions.json"),
    }
    (args.out / "prediction-freeze.json").write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
