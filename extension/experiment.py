import argparse
import hashlib
import json
import multiprocessing
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from local_client import LocalClient
from src import agent
from src.agent import ALFWORLD_RULES, ENV_RULES, TranscriptEntry, run_episode
from src.env_adapters import BaseAdapter
from src.rollout_driver import append_jsonl, write_json_atomic

BASE_BUILD_PROMPT = agent.build_prompt
BASE_RESTORE = BaseAdapter.from_checkpoint.__func__


def checked_restore(adapter: type, checkpoint: dict):
    expected = checkpoint["state_hash"] if isinstance(checkpoint, dict) else checkpoint.state_hash
    environment = BASE_RESTORE(adapter, checkpoint)
    actual = environment.state_hash()
    if actual != expected:
        environment.close()
        raise RuntimeError(f"Replay mismatch before generation: expected {expected}, got {actual}")
    return environment


def deliberation_prompt(*args, **kwargs) -> str:
    return BASE_BUILD_PROMPT(*args, **kwargs).replace("Reply with exactly one line: ACTION: <your next action>", "Reply with one short THOUGHT: <assess the current state and next decision>, followed by ACTION: <one exact next command> on its own final line.")


def configure_scaffold(name: str) -> None:
    if name not in ("legacy-action", "grounded-action-v1", "grounded-react-v1"):
        raise ValueError(f"Unknown scaffold: {name}")
    agent.build_prompt = deliberation_prompt if name == "grounded-react-v1" else BASE_BUILD_PROMPT
    BaseAdapter.from_checkpoint = classmethod(checked_restore)
    ENV_RULES["alfworld"] = ALFWORLD_RULES
    if name != "legacy-action":
        ENV_RULES["alfworld"] = ALFWORLD_RULES.replace("put <object> in/on <receptacle>", "move <object> to <receptacle>") + " You can carry only one object at a time. Choose an exact command from the current valid-action list; do not invent a different spelling. If an action fails, check the latest observation and valid commands before your next action."


def fork_payload(baseline: dict, step: int, arm: str) -> dict | None:
    steps = baseline["steps"]
    if step < 1 or len(steps) < step or steps[step - 1]["done"]:
        return None
    restored = step - int(arm == "A3")
    state = steps[restored - 1]["state_hash"] if restored else steps[0]["prev_state_hash"]
    end = steps[restored]["transcript_len_before"] if restored < len(steps) else len(baseline["transcript"])
    return {
        "restore_from": {"env_name": baseline["env"], "task_spec": baseline["task_spec"], "actions": baseline["actions"][:restored], "step_index": restored, "state_hash": state},
        "prior_transcript": [TranscriptEntry(**entry) for entry in baseline["transcript"][:end]],
        "expected_state_hash": state,
    }


def load_episode(path: Path, common: dict) -> dict | None:
    if not path.exists():
        return None
    saved = json.loads(path.read_text())
    if any(saved.get(key) != value for key, value in common.items()):
        raise ValueError(f"Configuration mismatch in {path}")
    episode = saved["episode"]
    if episode.get("failure") or episode.get("suspended"):
        number = 1
        while path.with_suffix(f".attempt{number}.json").exists():
            number += 1
        archived = path.with_suffix(f".attempt{number}.json")
        path.rename(archived)
        return None
    return episode


def record_episode(path: Path, metadata: dict, episode: dict, client: LocalClient) -> None:
    for step in episode.get("steps", []):
        prompt = step.pop("prompt", "")
        step["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
        step["prompt_chars"] = len(prompt)
    episode["local_call_events"] = client.events
    episode["local_usage"] = {key: sum(event[key] for event in client.events) for key in ("input_tokens", "output_tokens", "wall_s")}
    episode["local_usage"]["requests"] = client.n_requests
    episode["local_usage"]["unknown_usage_requests"] = sum(not event["usage_known"] for event in client.events)
    episode.pop("n_claude_requests", None)
    write_json_atomic(str(path), {**metadata, "episode": episode})


def run_task(task: dict, config: dict, endpoint: str, output: str, deadline: float, baseline_only: bool = False) -> dict:
    configure_scaffold(config.get("scaffold", "legacy-action"))
    directory = Path(output) / task["task_id"]
    directory.mkdir(parents=True, exist_ok=True)
    baseline_path = directory / "baseline.json"
    seed = task["seed"]
    client_options = {key: config[key] for key in ("system_prompt", "max_tokens") if key in config}
    common = {"task_id": task["task_id"], "split": task["split"], "model": config["model"], "config_sha256": config["config_sha256"]}
    baseline = load_episode(baseline_path, common)
    if baseline is None:
        cli = LocalClient(endpoint, config["model"], seed, deadline, config["temperature"], **client_options)
        baseline = run_episode(task["env"], task["task_spec"], cli, deadline=deadline)
        record_episode(baseline_path, {**common, "arm": "BASELINE", "round": -1, "seed": seed}, baseline, cli)
    if baseline.get("failure") or baseline.get("suspended"):
        return {**common, "state": "baseline_incomplete"}
    step = task["checkpoint_step"]
    if fork_payload(baseline, step, "A0") is None:
        return {**common, "state": "terminal_before_checkpoint", "success": baseline["success"]}
    if baseline_only:
        return {**common, "state": "baseline_ready"}
    cells = [(round_id, arm) for round_id in range(config["rounds"]) for arm in config["arms"]]
    random.Random(seed).shuffle(cells)
    completed = 0
    for round_id, arm in cells:
        path = directory / f"round{round_id}-{arm}.json"
        saved = load_episode(path, common)
        if saved is not None:
            completed += int(not saved.get("failure") and not saved.get("suspended") and saved.get("restore_hash_ok") is True)
            continue
        if time.time() >= deadline:
            break
        payload = fork_payload(baseline, step, arm)
        restored = payload["restore_from"]["step_index"]
        text = config["arms"][arm]
        interventions = {restored: text} if text else {}
        cell_seed = int.from_bytes(hashlib.sha256(f"{seed}:{round_id}:{arm}".encode()).digest()[:4], "big")
        cli = LocalClient(endpoint, config["model"], cell_seed, deadline, config["temperature"], **client_options)
        episode = run_episode(task["env"], task["task_spec"], cli, max_steps=baseline["step_limit"] - int(arm == "A3"), interventions=interventions, deadline=deadline, **payload)
        metadata = {**common, "arm": arm, "round": round_id, "seed": cell_seed, "checkpoint_step": step, "restored_step": restored}
        record_episode(path, metadata, episode, cli)
        completed += int(not episode.get("failure") and not episode.get("suspended") and episode.get("restore_hash_ok") is True)
    return {**common, "state": "complete" if completed == len(cells) else "incomplete", "completed_cells": completed, "planned_cells": len(cells)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-seconds", type=int, required=True)
    parser.add_argument("--split")
    parser.add_argument("--baseline-only", action="store_true")
    args = parser.parse_args()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    config["config_sha256"] = hashlib.sha256(config_bytes).hexdigest()
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    started = time.time()
    deadline = started + args.max_seconds
    tasks = [task for task in config["tasks"] if args.split is None or task["split"] == args.split]
    append_jsonl(str(output / "attempts.jsonl"), {"event": "start", "time": started, "deadline": deadline, "config_sha256": config["config_sha256"], "tasks": len(tasks), "workers": args.workers})
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(run_task, task, config, args.endpoint, str(output), deadline, args.baseline_only): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"task_id": task["task_id"], "state": "exception", "error": f"{type(error).__name__}: {error}"}
            append_jsonl(str(output / "progress.jsonl"), {"time": time.time(), **result})
            print(json.dumps(result), flush=True)
    append_jsonl(str(output / "attempts.jsonl"), {"event": "end", "time": time.time(), "wall_seconds": time.time() - started})


if __name__ == "__main__":
    main()
