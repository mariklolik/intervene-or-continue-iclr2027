import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "extension"), str(ROOT)]
from analysis import ARMS, load_blocks


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap(values: np.ndarray, groups: list[str], seed: int, draws: int = 10000) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or len(x) != len(groups) or not len(x):
        raise ValueError("Nonempty aligned task values and groups required")
    rng = np.random.default_rng(seed)
    task_means = x[rng.integers(0, len(x), size=(draws, len(x)))].mean(axis=1)
    unique = sorted(set(groups))
    indices = {group: np.flatnonzero(np.asarray(groups) == group) for group in unique}
    cluster_means = np.empty(draws)
    for draw in range(draws):
        sampled = rng.choice(unique, len(unique), replace=True)
        chosen = np.concatenate([indices[group] for group in sampled])
        cluster_means[draw] = x[chosen].mean()
    sums = np.array([x[indices[group]].sum() for group in unique])
    signs = rng.choice([-1, 1], size=(50000, len(unique)))
    null = (signs * sums).sum(axis=1) / len(x)
    p = (1 + np.sum(np.abs(null) >= abs(x.mean()) - 1e-12)) / (len(null) + 1)
    return {
        "mean": float(x.mean()),
        "n_tasks": len(x),
        "n_groups": len(unique),
        "task_bootstrap_95": np.quantile(task_means, [.025, .975]).tolist(),
        "group_bootstrap_95": np.quantile(cluster_means, [.025, .975]).tolist(),
        "group_sign_flip_p": float(p),
        "raw_task_values": x.tolist(),
    }


def holm(values: list[float]) -> list[float]:
    p = np.asarray(values, dtype=float)
    order = np.argsort(p, kind="stable")
    adjusted = np.empty(len(p))
    adjusted[order] = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    return adjusted.tolist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--prediction-freeze", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    prediction_bytes = args.predictions.read_bytes()
    predictions = json.loads(prediction_bytes)
    freeze = json.loads(args.prediction_freeze.read_text())
    if file_hash(args.predictions) != freeze["predictions_sha256"]:
        raise ValueError("Prediction freeze mismatch")
    rows, audits = [], []
    task_meta = {}
    early = {}
    for config_path in args.config:
        config = json.loads(config_path.read_text())
        accepted, audit = load_blocks(config_path, args.raw / config_path.stem)
        rows.extend(accepted)
        audits.append(audit)
        for task in config["tasks"]:
            task_meta[(task["env"], task["task_id"])] = task
        for task in audit["tasks"]:
            if task["state"] == "terminal_before_checkpoint":
                early[(task_meta[(task["env"], task["task_id"])]["env"], task["task_id"])] = float(task["baseline_success"])
    rows.sort(key=lambda row: (row["env"], row["task_id"]))
    expected = predictions["rows"]
    actual = [{key: row[key] for key in ["model", "env", "split", "task_id", "config_sha256"]} for row in rows]
    expected_core = [{key: row[key] for key in ["model", "env", "split", "task_id", "config_sha256"]} for row in expected]
    if actual != expected_core:
        raise ValueError("Complete holdout rows differ from frozen prediction rows")
    probability = {name: np.asarray(values, dtype=float) for name, values in predictions["policies"].items()}
    if any(values.shape != (len(rows), len(ARMS)) for values in probability.values()):
        raise ValueError("Policy probability shape mismatch")
    report = {"status": "COMPLETE", "domains": {}, "prediction_freeze_sha256": file_hash(args.prediction_freeze), "predictions_sha256": file_hash(args.predictions)}
    for domain_index, env in enumerate(["alfworld", "scienceworld"]):
        indices = [index for index, row in enumerate(rows) if row["env"] == env]
        subset = [rows[index] for index in indices]
        planned = [task for (task_env, _), task in task_meta.items() if task_env == env]
        complete_ids = {row["task_id"] for row in subset}
        early_ids = {task_id for task_env, task_id in early if task_env == env}
        missing = [task["task_id"] for task in planned if task["task_id"] not in complete_ids | early_ids]
        if missing:
            raise ValueError(f"Incomplete planned frame for {env}: {len(missing)}")
        y = np.asarray([row["Y"] for row in subset], dtype=float)
        policies = {
            "CONTINUE": probability["CONTINUE"][indices],
            "BEST_FIXED": probability["BEST_FIXED"][indices],
            "DIRECT_ADVANTAGE": probability["DIRECT_ADVANTAGE"][indices],
            "ARM_OUTCOME": probability["ARM_OUTCOME"][indices],
            "MATCHED_COMPARATOR": probability[f"MATCHED_{env.upper()}"][indices],
        }
        selected_name = predictions["selection"][env]["selected"]["name"]
        if selected_name in policies:
            policies["SAFE_SELECTED"] = policies[selected_name]
        elif selected_name == predictions["selection"][env]["strongest_matched_comparator"]:
            policies["SAFE_SELECTED"] = policies["MATCHED_COMPARATOR"]
        else:
            raise ValueError(f"Unknown selected safe policy: {selected_name}")
        values = {name: (y.mean(axis=1) * p).sum(axis=1) for name, p in policies.items()}
        groups = [task_meta[(env, row["task_id"])]["group_id"] for row in subset]
        early_values = np.array([early[(env, task_id)] for task_id in sorted(early_ids)])
        summaries = {}
        for name, value in values.items():
            summaries[name] = {
                "eligible_mean": float(value.mean()),
                "planned_mean": float((value.sum() + early_values.sum()) / len(planned)),
                "firing_rate_eligible": float(1 - policies[name][:, 0].mean()),
            }
        contrasts = {}
        for left, right in [
            ("DIRECT_ADVANTAGE", "CONTINUE"),
            ("DIRECT_ADVANTAGE", "MATCHED_COMPARATOR"),
            ("DIRECT_ADVANTAGE", "ARM_OUTCOME"),
            ("DIRECT_ADVANTAGE", "BEST_FIXED"),
            ("SAFE_SELECTED", "CONTINUE"),
        ]:
            contrast_values = np.r_[values[left] - values[right], np.zeros(len(early_values))]
            contrast_groups = groups + [task_meta[(env, task_id)]["group_id"] for task_id in sorted(early_ids)]
            contrasts[f"{left}_vs_{right}"] = bootstrap(contrast_values, contrast_groups, 270914 + domain_index)
        if len(y):
            chosen0 = y[:, 0].argmax(axis=1)
            chosen1 = y[:, 1].argmax(axis=1)
            gap = .5 * ((y[:, 0].max(axis=1) - y[np.arange(len(y)), 0, chosen1]) + (y[:, 1].max(axis=1) - y[np.arange(len(y)), 1, chosen0]))
        else:
            gap = np.array([])
        gap_values = np.r_[gap, np.zeros(len(early_values))]
        gap_groups = groups + [task_meta[(env, task_id)]["group_id"] for task_id in sorted(early_ids)]
        measurement = bootstrap(gap_values, gap_groups, 271014 + domain_index)
        if env == "alfworld":
            primary = [measurement, contrasts["DIRECT_ADVANTAGE_vs_CONTINUE"], contrasts["DIRECT_ADVANTAGE_vs_MATCHED_COMPARATOR"]]
            for item, adjusted in zip(primary, holm([item["group_sign_flip_p"] for item in primary])):
                item["holm_p_primary_family"] = adjusted
        report["domains"][env] = {
            "planned_tasks": len(planned),
            "eligible_tasks": len(subset),
            "early_terminal_tasks": len(early_values),
            "groups": len(set(gap_groups)),
            "policies": summaries,
            "contrasts": contrasts,
            "same_draw_selection_optimism": measurement,
            "selection": predictions["selection"][env],
        }
    args.out.mkdir(parents=True)
    (args.out / "results.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    (args.out / "ingestion-audits.json").write_text(json.dumps(audits, indent=2, sort_keys=True) + "\n")
    manifest = {
        "results_sha256": file_hash(args.out / "results.json"),
        "ingestion_audits_sha256": file_hash(args.out / "ingestion-audits.json"),
        "inputs": [{"path": str(path), "sha256": file_hash(path)} for path in args.config + [args.predictions, args.prediction_freeze, Path(__file__)]],
    }
    (args.out / "verification.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
