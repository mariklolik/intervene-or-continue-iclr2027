"""Step-6 feature construction: non-leaking, deployable-at-fork features.

A supervisor standing at the fork of prefix ``<task>@k`` has seen exactly the
first ``k`` executed steps of the trajectory (indices 0..k-1) and nothing else.
Every feature here is a pure function of ``steps[0:k]`` plus the environment
identifier.  Nothing that depends on the *future* of the trajectory or on its
outcome may enter.

HARD EXCLUSION LIST (leakage) -- see ``EXCLUDED_FEATURES`` below for the
machine-readable version with justifications.

Source of truth: ``data/processed/baseline_episodes_full.jsonl`` (88 baseline
episodes, each with a per-step record carrying ``parsed_action``,
``observation``, ``reward``, ``state_changed``, ``reversible``,
``admissible_n``, ``action_admissible``, ``invalid_output``, ``parse_retries``,
``obs_hash``, ``prompt_len`` and token usage).  READ-ONLY.

CPU-only, seed 42, zero LLM calls, no network, no model downloads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42

ROOT = Path(__file__).resolve().parent.parent
EPISODES_FULL = ROOT / "data" / "processed" / "baseline_episodes_full.jsonl"
PREFIX_FRAME = ROOT / "tables" / "step5_prefix_frame.csv"

# --------------------------------------------------------------------------
# Leakage exclusion list.  Each entry: column -> why it may never be a feature.
# --------------------------------------------------------------------------
EXCLUDED_FEATURES: dict[str, str] = {
    "norm_step_index": (
        "step_index / baseline_length -- the denominator is the eventual length "
        "of the baseline episode, which is unknowable at the fork. Leaky."
    ),
    "baseline_length": (
        "total length of the eventual baseline episode; a real-time supervisor "
        "cannot know how long the agent will keep going. Leaky."
    ),
    "checkpoint_position": (
        "early/middle/late is derived from norm_step_index, i.e. from the same "
        "unknowable denominator. Offline stratifier only, never a model input."
    ),
    "y_continue": "the outcome being predicted (baseline success). Target, not feature.",
    "y_a": "post-intervention outcome of an arm. Future/outcome.",
    "tau_a": "y_a - y_continue. Outcome contrast.",
    "harm": "derived from y_a and y_continue. Outcome.",
    "recovery": "derived from y_a and y_continue. Outcome.",
    "final_score": "terminal score of the episode. Outcome.",
    "n_steps": "total steps of the episode, i.e. its eventual length. Future.",
    "termination_reason": "how the episode ended. Future.",
    "step_limit_reached": "whether the episode hit the step cap. Future.",
    "arm / arm_name / intervention_text": "identifies the counterfactual arm, not the fork state.",
    "risk_family_loto": (
        "leave-one-task-out family failure rate; computed in Step 5 and found to "
        "have zero variance everywhere. Excluded for degeneracy, not leakage."
    ),
    "split": "experimental bookkeeping, not an observable of the fork state.",
    "task_key / prefix_id": "identifiers; would memorise clusters.",
    "steps[k:] (any per-step field at index >= k)": (
        "everything the agent does after the fork. The single largest leakage "
        "surface; the feature builder slices steps[0:k] and never looks past it."
    ),
}

# Observation strings that signal a wasted / rejected action.  Compiled once.
_FAIL_OBS_PAT = re.compile(
    r"(no known action matches|nothing happens|that's not something you|"
    r"you can't|cannot|not a valid|no such thing|i don't|invalid|"
    r"there is no way to do that|unknown action)",
    re.IGNORECASE,
)

# Coarse action-type vocabulary shared by ALFWorld and ScienceWorld.
_VERB_GROUPS: dict[str, tuple[str, ...]] = {
    "move": ("go to", "go ", "teleport", "move to"),
    "take": ("take", "pick up", "pick "),
    "put": ("put", "move ", "place"),
    "open": ("open",),
    "close": ("close",),
    "toggle": ("use", "activate", "deactivate", "turn on", "turn off", "toggle"),
    "look": ("look", "examine", "inspect"),
    "read": ("read",),
    "focus": ("focus",),
    "manip": ("clean", "heat", "cool", "slice", "mix", "pour", "connect", "wait", "dunk"),
}

_TOKEN_PAT = re.compile(r"[a-z0-9]+")


def _tok(text: str) -> list[str]:
    return _TOKEN_PAT.findall(text.lower())


def load_episodes() -> dict[str, dict]:
    """Load the 88 baseline episodes keyed by ``task_key`` (read-only)."""
    out: dict[str, dict] = {}
    with EPISODES_FULL.open() as fh:
        for line in fh:
            rec = json.loads(line)
            out[rec["task_key"]] = rec
    return out


def _safe_bool(value, default: bool = False) -> bool:
    if value is None:
        return default
    return bool(value)


def prefix_features(episode: dict, k: int) -> dict[str, float]:
    """Features observable by a supervisor after ``k`` executed steps.

    ``k >= 1`` always (Step-2 never forks before the first action).
    """
    steps = episode["steps"][:k]
    assert len(steps) == k, f"prefix requires {k} steps, episode has {len(steps)}"

    actions = [(s.get("parsed_action") or "") for s in steps]
    observations = [(s.get("observation") or "") for s in steps]
    rewards = np.array([float(s.get("reward") or 0.0) for s in steps])
    state_changed = np.array([_safe_bool(s.get("state_changed"), True) for s in steps])
    reversible = np.array([_safe_bool(s.get("reversible"), True) for s in steps])
    invalid = np.array([_safe_bool(s.get("invalid_output"), False) for s in steps])
    retries = np.array([float(s.get("parse_retries") or 0) for s in steps])
    admissible_n = np.array([float(s.get("admissible_n") or 0) for s in steps])
    act_adm = [s.get("action_admissible") for s in steps]
    obs_hash = [s.get("obs_hash") or "" for s in steps]
    prompt_len = np.array([float(s.get("prompt_len") or 0) for s in steps])
    out_tok = np.array([float((s.get("usage") or {}).get("output_tokens") or 0) for s in steps])

    feats: dict[str, float] = {}
    feats["steps_so_far"] = float(k)
    feats["log_steps_so_far"] = float(np.log1p(k))
    feats["env_scienceworld"] = 1.0 if episode["env"] == "scienceworld" else 0.0

    # --- progress signals (cumulative reward observed so far) ---------------
    cum_reward = float(rewards.sum())
    feats["cum_reward"] = cum_reward
    feats["any_reward"] = 1.0 if cum_reward > 0 else 0.0
    feats["reward_rate"] = cum_reward / k
    nz = np.flatnonzero(rewards > 0)
    feats["steps_since_reward"] = float(k - 1 - nz[-1]) if nz.size else float(k)
    feats["frac_steps_rewarded"] = float((rewards > 0).mean())

    # --- malformed / rejected agent output ---------------------------------
    feats["n_invalid_output"] = float(invalid.sum())
    feats["rate_invalid_output"] = float(invalid.mean())
    feats["n_parse_retries"] = float(retries.sum())

    # --- environment-state stagnation --------------------------------------
    feats["n_state_unchanged"] = float((~state_changed).sum())
    feats["rate_state_unchanged"] = float((~state_changed).mean())
    trail = 0
    for flag in reversed(state_changed.tolist()):
        if flag:
            break
        trail += 1
    feats["stall_state_unchanged_run"] = float(trail)
    feats["n_irreversible"] = float((~reversible).sum())
    feats["last_irreversible"] = 0.0 if reversible[-1] else 1.0

    # --- admissibility (ALFWorld exposes an admissible-action list) ---------
    feats["mean_admissible_n"] = float(admissible_n.mean())
    feats["last_admissible_n"] = float(admissible_n[-1])
    known = [a for a in act_adm if a is not None]
    feats["action_admissible_rate"] = float(np.mean([bool(a) for a in known])) if known else -1.0
    feats["n_action_inadmissible"] = float(sum(1 for a in known if a is False))

    # --- repetition / loop indicators --------------------------------------
    uniq_actions = len(set(actions))
    feats["distinct_action_ratio"] = uniq_actions / k
    counts = pd.Series(actions).value_counts()
    feats["max_action_repeat"] = float(counts.iloc[0]) if len(counts) else 0.0
    feats["n_repeated_actions"] = float(k - uniq_actions)
    feats["n_consecutive_repeats"] = float(
        sum(1 for i in range(1, k) if actions[i] == actions[i - 1])
    )
    feats["last_action_seen_before"] = 1.0 if actions[-1] in set(actions[:-1]) else 0.0

    # --- observation novelty ------------------------------------------------
    uniq_obs = len(set(obs_hash))
    feats["distinct_obs_ratio"] = uniq_obs / k
    feats["n_repeated_obs"] = float(k - uniq_obs)
    feats["last_obs_seen_before"] = 1.0 if obs_hash[-1] in set(obs_hash[:-1]) else 0.0

    failed = np.array([bool(_FAIL_OBS_PAT.search(o)) for o in observations])
    feats["n_failed_obs"] = float(failed.sum())
    feats["rate_failed_obs"] = float(failed.mean())
    feats["last_obs_failed"] = 1.0 if failed[-1] else 0.0
    trail_f = 0
    for flag in reversed(failed.tolist()):
        if not flag:
            break
        trail_f += 1
    feats["stall_failed_obs_run"] = float(trail_f)

    # --- token-level novelty of the observation stream ----------------------
    seen: set[str] = set()
    novel_counts: list[float] = []
    total_tokens = 0
    for obs in observations:
        toks = _tok(obs)
        total_tokens += len(toks)
        if toks:
            novel_counts.append(sum(1 for t in toks if t not in seen) / len(toks))
        else:
            novel_counts.append(0.0)
        seen.update(toks)
    feats["novel_token_frac_last"] = float(novel_counts[-1])
    feats["novel_token_frac_mean"] = float(np.mean(novel_counts))
    feats["novel_token_frac_last3"] = float(np.mean(novel_counts[-3:]))
    feats["vocab_size_seen"] = float(len(seen))
    feats["vocab_per_step"] = len(seen) / k
    feats["token_repeat_ratio"] = float(len(seen) / total_tokens) if total_tokens else 0.0

    # --- verbosity / prompt growth -----------------------------------------
    feats["last_prompt_len"] = float(prompt_len[-1])
    feats["prompt_len_per_step"] = float(prompt_len[-1] / k)
    feats["mean_output_tokens"] = float(out_tok.mean())
    feats["last_output_tokens"] = float(out_tok[-1])
    feats["mean_obs_len"] = float(np.mean([len(o) for o in observations]))
    feats["last_obs_len"] = float(len(observations[-1]))

    # --- action-type distribution so far ------------------------------------
    lowered = [a.lower().strip() for a in actions]
    assigned = np.zeros(k, dtype=bool)
    for group, prefixes in _VERB_GROUPS.items():
        hit = np.array(
            [any(a.startswith(p) or a.split(" ")[0] == p.strip() for p in prefixes) for a in lowered]
        )
        hit &= ~assigned
        assigned |= hit
        feats[f"act_frac_{group}"] = float(hit.mean())
    feats["act_frac_other"] = float((~assigned).mean())

    return feats


def recent_text(episode: dict, k: int, n_recent: int = 3) -> str:
    """Concatenated recent (action, observation) text at the fork, for TF-IDF."""
    steps = episode["steps"][:k][-n_recent:]
    parts: list[str] = []
    for s in steps:
        parts.append(s.get("parsed_action") or "")
        parts.append(s.get("observation") or "")
    return " ".join(parts)


def build_feature_frame() -> pd.DataFrame:
    """233-row prefix-level feature frame (identifiers + labels + features)."""
    episodes = load_episodes()
    frame = pd.read_csv(PREFIX_FRAME)
    rows: list[dict] = []
    for rec in frame.to_dict("records"):
        ep = episodes[rec["task_key"]]
        k = int(rec["step_index"])
        feats = prefix_features(ep, k)
        row = {
            "prefix_id": rec["prefix_id"],
            "task_key": rec["task_key"],
            "env": rec["env"],
            "split": rec["split"],
            "task_family": rec["task_family"],
            "step_index": k,
            # kept only as an OFFLINE STRATIFIER -- never passed to any model
            "checkpoint_position": rec["checkpoint_position"],
            "y_continue": int(bool(rec["y_continue"])),
            "y_fail": int(not bool(rec["y_continue"])),
            "tau_A1": float(rec["tau_A1"]),
            "tau_A2": float(rec["tau_A2"]),
            "tau_A3": float(rec["tau_A3"]),
            "max_tau_arms": float(rec["max_tau_arms"]),
            "best_action": rec["best_action"],
            "recent_text": recent_text(ep, k),
        }
        row.update(feats)
        rows.append(row)
    return pd.DataFrame(rows)


ID_COLS = [
    "prefix_id",
    "task_key",
    "env",
    "split",
    "task_family",
    "step_index",
    "checkpoint_position",
    "y_continue",
    "y_fail",
    "tau_A1",
    "tau_A2",
    "tau_A3",
    "max_tau_arms",
    "best_action",
    "recent_text",
]


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in ID_COLS]


if __name__ == "__main__":  # pragma: no cover - manual smoke
    fr = build_feature_frame()
    cols = feature_columns(fr)
    print(fr.shape, len(cols))
    print(cols)
    print(fr.groupby("split")["y_fail"].agg(["size", "sum"]))
