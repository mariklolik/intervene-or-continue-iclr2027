import hashlib
import json

BASE_ARMS = ["A0", "A1", "A2", "A3"]
FIXED_ROLLBACK = {"A3": 1}
PROGRESS_ROLLBACK = {"A4"}
SCHEDULED = "scheduled"
EVENT = "event"


def arm_names(config: dict) -> list[str]:
    return sorted(config["arms"])


def rounds(config: dict) -> int:
    return int(config["rounds"])


def observation_hashes(baseline: dict, step: int) -> list[str]:
    return [entry.get("obs_hash") for entry in baseline["steps"][:step]]


def restore_index(baseline: dict, step: int, arm: str) -> int:
    if arm in FIXED_ROLLBACK:
        return max(0, step - FIXED_ROLLBACK[arm])
    if arm in PROGRESS_ROLLBACK:
        hashes = observation_hashes(baseline, step)
        for index in range(len(hashes) - 1, 0, -1):
            if hashes[index] != hashes[index - 1]:
                return index
        return max(0, step - 1)
    return step


def event_step(baseline: dict, floor: int, ceiling: int) -> int | None:
    steps = baseline["steps"]
    previous = None
    for index, step in enumerate(steps[:ceiling]):
        repeated = step.get("obs_hash") is not None and step.get("obs_hash") == previous
        stuck = bool(step.get("invalid_output")) or step.get("action_admissible") is False
        previous = step.get("obs_hash")
        if index + 1 >= floor and (repeated or stuck) and not step["done"]:
            return index + 1
    return None


def checkpoint_index(task: dict, baseline: dict) -> int | None:
    if task.get("checkpoint_rule", SCHEDULED) == SCHEDULED:
        return task["checkpoint_step"]
    window = task.get("checkpoint_window", [2, task["checkpoint_step"]])
    return event_step(baseline, window[0], window[1])


def cell_seed(seed: int, round_id: int, arm: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}:{round_id}:{arm}".encode()).digest()[:4], "big")


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
