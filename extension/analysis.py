import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ARMS = ["A0", "A1", "A2", "A3"]
sys.dont_write_bytecode = True
SOURCE_ROOT = Path(__file__).resolve().parents[1] / "Holosophus/run-archives/intervene-or-continue-20260820T150100Z-foreground3-claude-opus-5/workspace"
if SOURCE_ROOT.exists():
    sys.path.insert(0, str(SOURCE_ROOT))


def diagnostics(outcomes: np.ndarray) -> dict:
    y = np.asarray(outcomes, dtype=float)
    if y.ndim != 3 or y.shape[1:] != (2, 4) or not np.isin(y, [0, 1]).all():
        raise ValueError("Expected binary outcomes shaped [independent task, 2 rounds, 4 arms]")
    selected = np.argmax(y, axis=2)
    best = y.max(axis=2)
    cross = np.take_along_axis(y[:, ::-1, :], selected[:, :, None], axis=2)[:, :, 0]
    baseline = y[:, :, 0]
    d = y - baseline[:, :, None]
    return {"selected_actions": selected, "same_round_value": best.mean(axis=1),
            "cross_selected_value": cross.mean(axis=1), "same_round_opportunity": (best - baseline).mean(axis=1),
            "cross_selected_uplift": (cross - baseline[:, ::-1]).mean(axis=1), "gap": (best - cross).mean(axis=1),
            "fixed_arm_uplift": d.mean(axis=1), "products": d[:, 0] * d[:, 1],
            "flips": (y[:, 0] != y[:, 1]).astype(float), "success": y.mean(axis=1),
            "selection_firing_rate": (selected != 0).mean(axis=1)}


def summarize_tasks(values, bounds: tuple[float, float], seed: int = 42, n_boot: int = 10000) -> dict:
    x = np.asarray(values, dtype=float)
    if x.ndim != 1 or not np.isfinite(x).all() or bounds[0] >= bounds[1] or np.any((x < bounds[0]) | (x > bounds[1])):
        raise ValueError("Supply one finite bounded scalar per independent task")
    result = {"n_tasks": len(x), "raw_task_values": x.tolist(), "mean": None, "interval_95": None,
              "interval_method": "no_data", "bootstrap_95": None, "bounded_95": None, "bootstrap_degenerate": None,
              "seed": seed, "n_boot": n_boot, "assumption": "Independent sampled task blocks; repeated arms and rounds are not sampling units"}
    if not len(x):
        return result
    means = x[np.random.default_rng(seed).integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    bootstrap = np.quantile(means, [.025, .975]).tolist()
    width = (bounds[1] - bounds[0]) * np.sqrt(np.log(40) / (2 * len(x)))
    bounded = [float(max(bounds[0], x.mean() - width)), float(min(bounds[1], x.mean() + width))]
    degenerate = bool(np.all(means == means[0]))
    result.update(mean=float(x.mean()), bootstrap_95=bootstrap, bounded_95=bounded, bootstrap_degenerate=degenerate,
                  interval_95=bounded if degenerate else bootstrap, interval_method="bounded_hoeffding" if degenerate else "task_percentile_bootstrap")
    return result


def prefix_metadata(baseline: dict, checkpoint: int) -> dict:
    from src.step6_features import prefix_features, recent_text

    safe_keys = ["step", "parsed_action", "observation", "reward", "admissible_n", "action_admissible", "invalid_output", "parse_retries", "obs_hash", "prompt_chars", "usage"]
    steps = [{**{k: v for k, v in step.items() if k in safe_keys}, "prompt_len": step.get("prompt_chars", step.get("prompt_len", 0))} for step in baseline["steps"][:checkpoint]]
    episode = {"env": baseline["env"], "steps": steps}
    transcript = [entry for entry in baseline.get("transcript", []) if entry["step"] < checkpoint or (entry["step"] == checkpoint and entry["kind"] == "observation")]
    features = prefix_features(episode, checkpoint)
    for key in ["n_state_unchanged", "rate_state_unchanged", "stall_state_unchanged_run", "n_irreversible", "last_irreversible"]:
        features.pop(key)
    return {"checkpoint_step": checkpoint, "features": features, "recent_text": recent_text(episode, checkpoint), "steps": steps, "actions": baseline.get("actions", [])[:checkpoint], "transcript": transcript, "signal_contract": "Public observations/actions, prior usage and public env.step rewards shared by every controller; hidden state hashes and reversibility excluded"}


def inventory(raw: Path) -> tuple[dict, dict]:
    records, entries, issues = {}, [], []
    usage = dict.fromkeys(["requests", "input_tokens", "output_tokens", "wall_s", "failed_requests", "unknown_usage_requests"], 0)
    for path in sorted(raw.rglob("*.json")):
        name = str(path.relative_to(raw))
        try:
            content = path.read_bytes()
            record = json.loads(content)
            episode = record["episode"]
        except (ValueError, KeyError, TypeError, OSError) as error:
            issues.append({"path": name, "error": str(error), "usage_status": "unrecoverable_from_this_record"})
            continue
        records[name] = record
        events = episode.get("local_call_events")
        entry = {"path": name, "sha256": hashlib.sha256(content).hexdigest(), "task_id": record.get("task_id"), "arm": record.get("arm"), "round": record.get("round"), "attempt_archive": ".attempt" in path.name, "failure": episode.get("failure"), "suspended": episode.get("suspended"), "restore_hash_ok": episode.get("restore_hash_ok"), "success": episode.get("success"), "interventions_injected": episode.get("interventions_injected"), "requests": len(events) if isinstance(events, list) else None}
        entries.append(entry)
        if not isinstance(events, list):
            issues.append({"path": name, "error": "local_call_events missing", "usage_status": "unknown"})
            continue
        for event in events:
            usage["requests"] += 1
            usage["failed_requests"] += int(not event.get("ok", False))
            usage["unknown_usage_requests"] += int(not event.get("usage_known", False))
            for key in ["input_tokens", "output_tokens", "wall_s"]:
                usage[key] += event.get(key, 0) or 0
        declared = episode.get("local_usage", {}).get("requests")
        if declared is not None and declared != len(events):
            issues.append({"path": name, "error": "local_usage request count differs from event ledger", "declared": declared, "recorded": len(events)})
    logs = {}
    for name in ["attempts.jsonl", "progress.jsonl"]:
        path = raw / name
        logs[name] = []
        if path.exists():
            for number, line in enumerate(path.read_text().splitlines(), 1):
                try:
                    logs[name].append(json.loads(line))
                except ValueError:
                    issues.append({"path": name, "line": number, "error": "invalid JSONL", "raw_line": line})
    return records, {"episode_inventory": entries, "usage": usage, "file_issues": issues, "orchestration_logs": logs,
                     "usage_caveat": "Recorded local_call_events summed across every file and attempt; missing files/process-killed unsaved calls remain unknown. This is not allocated GPU time."}


def validate_cell(record: dict, task: dict, config: dict, digest: str, baseline: dict, round_id: int, arm: str) -> list[str]:
    restored = task["checkpoint_step"] - int(arm == "A3")
    seed = int.from_bytes(hashlib.sha256(f"{task['seed']}:{round_id}:{arm}".encode()).digest()[:4], "big")
    expected = {"task_id": task["task_id"], "split": task["split"], "model": config["model"], "config_sha256": digest, "round": round_id, "arm": arm, "seed": seed, "checkpoint_step": task["checkpoint_step"], "restored_step": restored}
    reasons = [f"metadata_{key}" for key, value in expected.items() if record.get(key) != value]
    episode = record["episode"]
    for key in ["failure", "suspended", "censored", "excluded"]:
        if episode.get(key): reasons.append(key)
    if episode.get("restore_hash_ok") is not True: reasons.append("restore_hash_not_verified")
    if episode.get("env") != task["env"]: reasons.append("environment_mismatch")
    if episode.get("task_spec") != task["task_spec"]: reasons.append("task_spec_mismatch")
    if episode.get("segment_start_step") != restored: reasons.append("segment_start_mismatch")
    if episode.get("step_limit") != baseline["step_limit"] - int(arm == "A3"): reasons.append("remaining_budget_mismatch")
    if not isinstance(episode.get("success"), bool): reasons.append("success_not_binary")
    state = baseline["steps"][restored - 1]["state_hash"] if restored else baseline["steps"][0]["prev_state_hash"]
    if not episode.get("steps") or episode["steps"][0].get("prev_state_hash") != state: reasons.append("initial_state_mismatch_or_no_execution")
    injected = episode.get("interventions_injected")
    if injected != ([] if arm == "A0" else [restored]): reasons.append("assigned_intervention_not_delivered")
    messages = [e for e in episode.get("transcript", []) if e.get("kind") == "overseer"]
    if arm != "A0" and not any(e.get("step") == restored and e.get("text") == config["arms"][arm] for e in messages): reasons.append("intervention_text_unverified")
    if arm == "A0" and messages: reasons.append("continue_received_intervention")
    events = episode.get("local_call_events", [])
    if not events: reasons.append("no_fresh_call_evidence")
    if any(e.get("model", config["model"]) != config["model"] for e in events): reasons.append("response_model_mismatch")
    for index, event in enumerate(events):
        expected_seed = int.from_bytes(hashlib.sha256(f"{seed}:{index}:{event.get('tag', '')}".encode()).digest()[:4], "big") % 2147483647
        if event.get("seed") != expected_seed: reasons.append(f"request_{index}_seed_mismatch")
    return reasons


def load_blocks(config_path: Path, raw_dir: Path, split: str | None = None) -> tuple[list[dict], dict]:
    config_bytes = Path(config_path).read_bytes()
    config = json.loads(config_bytes)
    digest = hashlib.sha256(config_bytes).hexdigest()
    if config.get("rounds") != 2 or set(config.get("arms", {})) != set(ARMS) or config["arms"]["A0"]:
        raise ValueError("The registered design requires two rounds and fresh A0,A1,A2,A3")
    tasks = config["tasks"]
    if len({t["task_id"] for t in tasks}) != len(tasks): raise ValueError("Duplicate task identifiers")
    records, audit = inventory(Path(raw_dir))
    accepted, task_audit = [], []
    for task in tasks:
        task_id, checkpoint = task["task_id"], task["checkpoint_step"]
        item = {"task_id": task_id, "split": task["split"], "env": task["env"], "state": "not_started", "reasons": [], "missing_cells": [], "valid_cells": 0, "cells": [], "checkpoint_eligible": None}
        task_audit.append(item)
        if split is not None and task["split"] != split:
            item["state"] = "outside_requested_split"
            continue
        baseline_record = records.get(f"{task_id}/baseline.json")
        for round_id in range(2):
            for arm in ARMS:
                if f"{task_id}/round{round_id}-{arm}.json" not in records: item["missing_cells"].append(f"round{round_id}-{arm}.json")
        if baseline_record is None:
            item["reasons"].append("baseline.json:missing")
            continue
        baseline = baseline_record["episode"]
        common = {"task_id": task_id, "split": task["split"], "model": config["model"], "config_sha256": digest, "arm": "BASELINE", "round": -1, "seed": task["seed"]}
        item["reasons"] += [f"baseline.json:metadata_{k}" for k, v in common.items() if baseline_record.get(k) != v]
        item["reasons"] += [f"baseline.json:{k}_mismatch" for k in ["env", "task_spec"] if baseline.get(k) != task[k]]
        if baseline.get("failure") or baseline.get("suspended") or item["reasons"]:
            item["state"] = "baseline_incomplete_or_invalid"
            item["reasons"].append("baseline_failure_or_suspension_or_provenance")
            continue
        if checkpoint < 1 or len(baseline["steps"]) < checkpoint or baseline["steps"][checkpoint - 1]["done"]:
            item.update(state="terminal_before_checkpoint", baseline_success=baseline.get("success"), checkpoint_eligible=False)
            continue
        item["checkpoint_eligible"] = True
        cells, y = [], []
        for round_id in range(2):
            outcome_round = []
            for arm in ARMS:
                name = f"round{round_id}-{arm}.json"
                record = records.get(f"{task_id}/{name}")
                reasons = ["missing"] if record is None else validate_cell(record, task, config, digest, baseline, round_id, arm)
                item["reasons"] += [f"{name}:{reason}" for reason in reasons]
                item["cells"].append({"round": round_id, "arm": arm, "path": f"{task_id}/{name}", "present": record is not None, "valid": not reasons, "reasons": reasons})
                if reasons: continue
                item["valid_cells"] += 1
                episode = record["episode"]
                outcome_round.append(int(episode["success"]))
                cells.append({"round": round_id, "arm": arm, "seed": record["seed"], "path": f"{task_id}/{name}", "local_call_events": episode["local_call_events"], "success": episode["success"], "interventions_injected": episode["interventions_injected"]})
            y.append(outcome_round)
        item["state"] = "complete" if item["valid_cells"] == 8 and not item["reasons"] else "incomplete_or_invalid"
        if item["state"] == "complete":
            accepted.append({"task_id": task_id, "split": task["split"], "env": task["env"], "model": config["model"], "model_revision": config.get("model_revision"), "config_sha256": digest, "scaffold": config.get("scaffold", "legacy-action"), "arms": ARMS, "Y": y, "cells": cells, "prefix_only": prefix_metadata(baseline, checkpoint)})
    selected_tasks = [t for t in task_audit if t["state"] != "outside_requested_split"]
    denominators = {}
    for arm in ARMS:
        cells = [c for t in selected_tasks for c in t["cells"] if c["arm"] == arm]
        denominators[arm] = {"potential_cells": 2 * len(selected_tasks), "confirmed_eligible_cells": 2 * sum(t["checkpoint_eligible"] is True for t in selected_tasks), "valid_cells": sum(c["valid"] for c in cells), "present_cells_at_eligible_checkpoints": sum(c["present"] for c in cells), "accepted_block_cells": 2 * len(accepted), "all_episode_attempt_records": sum(e["arm"] == arm for e in audit["episode_inventory"])}
    expected_names = {"baseline.json"} | {f"round{r}-{a}.json" for r in range(2) for a in ARMS}
    unexpected = [p for p in records if p.split("/")[0] not in {t["task_id"] for t in tasks} or (Path(p).name not in expected_names and ".attempt" not in Path(p).name)]
    audit.update(tasks=task_audit, config_sha256=digest, config_path=str(config_path), phase=config.get("phase"), model=config["model"], model_revision=config.get("model_revision"), provenance_status="revision_declared_in_hashed_config" if config.get("model_revision") else "exact_model_revision_unavailable", provenance_caveat="Metadata/config consistency does not itself certify served weight hashes or conditional stochastic independence", unexpected_episode_paths=unexpected, arm_denominators=denominators, accepted_tasks=len(accepted), planned_tasks=len(tasks))
    return accepted, audit


def summarize_blocks(rows: list[dict], seed: int = 42) -> dict:
    y = np.array([row["Y"] for row in rows], dtype=int).reshape(-1, 2, 4)
    d = diagnostics(y)
    summaries = {key: summarize_tasks(d[key], bounds, seed) for key, bounds in {"gap": (0, 1), "same_round_opportunity": (0, 1), "same_round_value": (0, 1), "cross_selected_value": (0, 1), "cross_selected_uplift": (-1, 1), "selection_firing_rate": (0, 1)}.items()}
    per_arm = {}
    for index, arm in enumerate(ARMS):
        harm = ((y[:, :, index] == 0) & (y[:, :, 0] == 1)).sum(axis=1)
        recovery = ((y[:, :, index] == 1) & (y[:, :, 0] == 0)).sum(axis=1)
        per_arm[arm] = {key: summarize_tasks(d[key][:, index], (0, 1) if key in ["flips", "success"] else (-1, 1), seed) for key in ["fixed_arm_uplift", "products", "flips", "success"]}
        per_arm[arm].update(round_observations=2 * len(rows), independent_tasks=len(rows), harm_count=int(harm.sum()), at_risk_rounds=int((y[:, :, 0] == 1).sum()), recovery_count=int(recovery.sum()), baseline_failure_rounds=int((y[:, :, 0] == 0).sum()), per_task_harm=harm.tolist(), per_task_recovery=recovery.tolist())
    return {"n_tasks": len(rows), "task_ids": [r["task_id"] for r in rows], "diagnostics": summaries, "per_arm": per_arm, "selected_actions": d["selected_actions"].tolist(), "raw_Y": y.tolist(), "primary_identity": "gap = 0.5 * [(max(Y0)-Y0[argmax(Y1)]) + (max(Y1)-Y1[argmax(Y0)])] >= 0 pathwise", "causal_boundary": "Selection instability is not causal benefit or uniquely diagnosed winner's curse; products are signed uncentered squared-effect moments conditional on independent rounds and separate fresh A0.", "multiplicity": "Intervals are marginal; secondary-stratum and learned-policy confirmatory claims require the separately declared Holm family."}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    accepted, audit = load_blocks(args.config, args.raw, args.split)
    strata = {}
    for model, env, split in sorted({(audit["model"], t["env"], t["split"]) for t in audit["tasks"] if t["state"] != "outside_requested_split"}):
        subset = [r for r in accepted if (r["model"], r["env"], r["split"]) == (model, env, split)]
        summary = summarize_blocks(subset, args.seed)
        tasks = [t for t in audit["tasks"] if (t["env"], t["split"]) == (env, split)]
        eligible = sum(t["checkpoint_eligible"] is True for t in tasks)
        total = sum(summary["diagnostics"]["gap"]["raw_task_values"])
        summary["missingness_sensitivity"] = {"confirmed_eligible_tasks": eligible, "complete_tasks": len(subset), "unknown_eligibility_tasks": sum(t["checkpoint_eligible"] is None for t in tasks), "finite_frame_gap_bounds": [total / eligible, (total + eligible - len(subset)) / eligible] if eligible else None, "interpretation": "Worst-case [0,1] gaps for incomplete confirmed-eligible tasks; descriptive finite frame, not a population interval or missing-at-random assertion"}
        strata[f"{model}|{env}|{split}"] = summary
    metrics = {"status": "complete_blocks_available" if accepted else "no_complete_blocks", "ingestion": audit, "strata": strata, "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, value in [("metrics.json", metrics), ("rows.json", accepted)]:
        (args.out / name).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
